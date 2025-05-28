from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# Imports from original api.py that are still needed for FastAPI routes
from livekit.protocol.room import DeleteRoomRequest
from livekit import api as livekit_api # For stop_session_endpoint

# New imports for services and config
from app.services.agent_service import BetsyTeacher
from app.services.token_service import create_token as create_livekit_token # Import the token service
from app.config import logger as app_config_logger, settings
from app.logger import ServerLogger # Import our custom logger class

# Ensure the logger instance is treated as ServerLogger for type hinting
assert isinstance(app_config_logger, ServerLogger), \
    f"Logger from app.config is not a ServerLogger instance, type: {type(app_config_logger)}"
logger: ServerLogger = app_config_logger

app = FastAPI(
    title="BETSY Server",
    description="API for the BETSY teacher agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

teacher = BetsyTeacher() # Instantiate the teacher from agent_service

# Pydantic models (kept here for now as per instructions)
class StartSessionRequest(BaseModel):
    instructions: str = Field(..., description="System prompt instructions for the teacher")
    room_name: str = Field(..., description="LiveKit room name to join")

class SessionResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[str] = None
    room_name: Optional[str] = None

# Pydantic model for the new token endpoint response
class LiveKitTokenResponse(BaseModel):
    token: str
    livekit_url: str
    room_name: str
    identity: str

@app.post("/api/livekit-token", response_model=LiveKitTokenResponse)
async def livekit_token_endpoint():
    """
    Generates a LiveKit token, a unique room name, and a unique identity.
    """
    try:
        logger.info(f"Requesting LiveKit token")
        token_data = create_livekit_token()
        logger.info(f"Token generated for room: {token_data['room_name']}, identity: {token_data['identity']}")
        return LiveKitTokenResponse(**token_data)
    except ValueError as ve:
        logger.error(f"ValueError generating LiveKit token: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error generating LiveKit token: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate LiveKit token")

@app.post("/api/start-session", response_model=SessionResponse)
async def start_session_endpoint(request: StartSessionRequest):
    try:
        logger.info(f"Starting session with instructions: {request.instructions[:50]}... (truncated)")
        logger.info(f"Agent will join room: {request.room_name}")
        
        if teacher.active and teacher.room_name == request.room_name:
            logger.warning(f"Agent already active in room {request.room_name}. Re-initializing and reconnecting.")
            await teacher.stop() 
        elif teacher.active and teacher.room_name != request.room_name:
            logger.info(f"Agent active in a different room ({teacher.room_name}). Stopping it before connecting to {request.room_name}.")
            await teacher.stop()

        await teacher.initialize_session(request.instructions)
        
        logger.info(f"Attempting to connect teacher to room: {request.room_name}")
        connection_successful = await teacher.connect_to_room(request.room_name)
        
        if not connection_successful:
            error_msg = f"Failed to connect to LiveKit room: {request.room_name}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        logger.success(f"Successfully joined room: {request.room_name}")
        
        return SessionResponse(
            success=True,
            message=f"Session started successfully in room {request.room_name}",
            session_id="global", # Assuming a single global teacher instance for now
            room_name=request.room_name
        )
    except Exception as e:
        error_msg = f"Error starting session: {e}"
        logger.error(error_msg, exc_info=True)
        if isinstance(e, HTTPException):
            raise
        return SessionResponse(
            success=False,
            message=error_msg
        )

@app.post("/api/stop-session", response_model=SessionResponse)
async def stop_session_endpoint(): 
    try:
        if not teacher.active:
            logger.warning("No active session found to stop")
            return SessionResponse(
                success=False,
                message="No active session found"
            )
        
        current_room_name = teacher.room_name 
        logger.info(f"Stopping session. Current room: {current_room_name}")
        
        await teacher.stop() 
        logger.info(f"Teacher stop process initiated for room: {current_room_name}.")

        if current_room_name: 
            try:
                # Replace os.getenv with settings
                livekit_url = settings.LIVEKIT_URL
                livekit_api_key = settings.LIVEKIT_API_KEY
                livekit_api_secret = settings.LIVEKIT_API_SECRET

                if not all([livekit_url, livekit_api_key, livekit_api_secret]):
                    logger.error("LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET not configured in settings. Cannot delete room.")
                else:
                    lk_api_client = livekit_api.LiveKitAPI(livekit_url, livekit_api_key, livekit_api_secret)
                    logger.info(f"Attempting to delete room: {current_room_name} from LiveKit server.")
                    delete_req = DeleteRoomRequest(room=current_room_name)
                    await lk_api_client.room.delete_room(delete_req)
                    logger.success(f"Successfully deleted room: {current_room_name} from LiveKit server.")
                    await lk_api_client.aclose()
            except Exception as delete_e:
                logger.error(f"Error deleting room {current_room_name} from LiveKit server: {delete_e}", exc_info=True)
        
        teacher.room_name = None # Clear the room name on the teacher instance

        return SessionResponse(
            success=True,
            message=f"Session stopped successfully (room: {current_room_name}). Room deletion attempted.",
            room_name=current_room_name 
        )
    except Exception as e:
        error_msg = f"Error stopping session: {e}"
        logger.error(error_msg, exc_info=True)
        return SessionResponse(
            success=False,
            message=error_msg
        )

@app.get("/api/status")
async def get_status():
    return {
        "active": teacher.active,
        "room_name": teacher.room_name,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# The uvicorn.run call will be in main.py, so it's not included here.