#!/usr/bin/env python3
import asyncio
import os
import types
import logging
from datetime import datetime
from typing import Optional, Dict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    Worker,
    WorkerOptions,
    JobExecutorType,
    stt,
    vad,
)
from livekit.protocol.room import DeleteRoomRequest
from livekit.plugins import openai, silero
from livekit import api as livekit_api

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment_variables():

    env_file = Path('.env.local')
    if env_file.exists():
        logger.info(f"Loading environment from {env_file.absolute()}")
        load_dotenv(env_file)
    else:
        logger.info("No .env.local found, trying .env")
        load_dotenv()

    logger.info("Checking required environment variables:")
    missing_vars = []
    for var_name in required_vars:
        value = os.getenv(var_name)
        if value:
            masked_value = value[:3] + '***' if len(value) > 3 else '***'
            logger.info(f"  {var_name}: {masked_value} (SET)")
        else:
            logger.warning(f"  {var_name}: NOT SET")
            missing_vars.append(var_name)

    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

required_vars = (
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "LIVEKIT_URL",
    "OPENAI_API_KEY"
)

load_environment_variables()

class BetsyTeacher:
    def __init__(self):
        self.agent_instructions: Optional[str] = None
        self.active: bool = False
        self.room_name: Optional[str] = None
        self._worker_task: Optional[asyncio.Task] = None

    async def _agent_entrypoint_method(self, ctx: JobContext):
        agent = None
        session = None
        session_task = None
        try:
            logger.info(f"Agent Worker (Thread): Initializing for room '{self.room_name}' with instructions: '{self.agent_instructions[:50]}...'")
            
            logger.info(f"Agent Worker (Thread): Initializing LiveKit components for room '{self.room_name}'")
            
            agent = Agent(instructions=self.agent_instructions)
            
            # Attempt to initialize Silero VAD
            vad_instance = None
            try:
                # Create an object for opts with a sample_rate attribute
                vad_opts_obj = types.SimpleNamespace(sample_rate=16000)
                vad_instance = silero.VAD(session=ctx.room, opts=vad_opts_obj)
                logger.info(f"Agent Worker (Thread): Silero VAD instantiated with ctx.room and opts: {vad_opts_obj}")
            except Exception as e_vad:
                logger.error(f"Agent Worker (Thread): Failed to instantiate silero.VAD: {e_vad}", exc_info=True)
                # If VAD fails, we can't use StreamAdapter, which will likely lead to STT issues.
                # For now, we'll let it proceed and see if AgentSession handles a None VAD gracefully
                # or if STT without StreamAdapter (and thus without VAD) works differently.
                # This is primarily to get past the StreamAdapter(vad=...) requirement if VAD init fails.

            openai_stt = openai.STT()
            if vad_instance:
                adapted_stt = stt.StreamAdapter(stt=openai_stt, vad=vad_instance)
                logger.info("Agent Worker (Thread): OpenAI STT wrapped with StreamAdapter using Silero VAD.")
            else:
                # Fallback: If VAD instantiation failed, pass STT directly.
                # This will likely re-trigger the "STT does not support streaming without VAD" error,
                # but it allows us to test VAD initialization separately.
                adapted_stt = openai_stt
                logger.warning("Agent Worker (Thread): Silero VAD instantiation failed. Passing raw OpenAI STT to AgentSession. Expecting STT streaming error.")

            session = AgentSession(
                stt=adapted_stt,
                llm=openai.LLM(model="gpt-4o"),
                tts=openai.TTS(voice="alloy"),
                vad=vad_instance, # Pass the VAD instance to AgentSession as well, if available
            )
            logger.info(f"Agent Worker (Thread): AgentSession instantiated (VAD {'provided' if vad_instance else 'not provided'}).")

            logger.info(f"Agent Worker (Thread): Connecting to room: {self.room_name}")
            await ctx.connect(auto_subscribe=True) 
            logger.info(f"Agent Worker (Thread): Connected to room: {ctx.room.name}")
            
            logger.info(f"Agent Worker (Thread): Starting agent session task in room: {self.room_name}")
            session_task = asyncio.create_task(session.start(agent=agent, room=ctx.room), name=f"AgentSession-{self.room_name}")

            await asyncio.sleep(1.0) 
            
            if ctx.room.isconnected and hasattr(session, '_started') and session._started:
                try:
                    logger.info(f"Agent Worker (Thread): Attempting to say a welcome message.")
                    await session.generate_reply(instructions="Dis bonjour et présente-toi brièvement.")
                    logger.info(f"Agent Worker (Thread): Welcome message task created/finished.")
                except RuntimeError as e_runtime: 
                    logger.error(f"Agent Worker (Thread): RuntimeError trying to say welcome message (session started: {session._started}): {e_runtime}", exc_info=True)
                except Exception as e_say:
                    logger.error(f"Agent Worker (Thread): Generic error trying to say welcome message: {e_say}", exc_info=True)
            elif hasattr(session, '_started') and not session._started :
                 logger.warning(f"Agent Worker (Thread): Session not in a 'started' state (session._started = {session._started}), cannot say welcome message.")
            elif not ctx.room.isconnected:
                 logger.warning(f"Agent Worker (Thread): Room disconnected, cannot say welcome message.")

            logger.info(f"Agent Worker (Thread): Entering keep-alive loop for room {self.room_name}.")
            while ctx.room.isconnected:
                if session_task and session_task.done():
                    logger.info(f"Agent Worker (Thread): Session task for {self.room_name} completed.")
                    try:
                        session_task.result() 
                    except asyncio.CancelledError:
                        logger.info(f"Agent Worker (Thread): Session task for {self.room_name} was cancelled.")
                    except Exception as e_session:
                        logger.error(f"Agent Worker (Thread): Exception from session_task: {e_session}", exc_info=True)
                    break 
                await asyncio.sleep(0.5)
            
            if not ctx.room.isconnected:
                logger.info(f"Agent Worker (Thread): Room {self.room_name} disconnected, stopping entrypoint.")
                if session_task and not session_task.done():
                    session_task.cancel()
                    try:
                        await session_task
                    except asyncio.CancelledError:
                        logger.info(f"Agent Worker (Thread): Session task for {self.room_name} cancelled during room disconnect.")

            logger.info(f"Agent Worker (Thread): Agent session processing finished for room: {self.room_name}")

        except Exception as e:
            logger.error(f"Agent Worker (Thread): Critical error in entrypoint for room {self.room_name}: {e}", exc_info=True)
            raise
        finally:
            logger.info(f"Agent Worker (Thread): Exiting entrypoint for room {self.room_name}.")
            if session:
                try:
                    if hasattr(session, '_started') and session._started and (not hasattr(session, '_closing_task') or (hasattr(session, '_closing_task') and session._closing_task is None)):
                         logger.info(f"Agent Worker (Thread): Explicitly closing session for {self.room_name} in finally block.")
                         await session.aclose()
                         logger.info(f"Agent Worker (Thread): Session for room {self.room_name} closed in finally block.")
                    elif hasattr(session, '_started') and not session._started:
                         logger.info(f"Agent Worker (Thread): Session for room {self.room_name} was not started, no aclose needed in finally.")
                    else:
                         logger.info(f"Agent Worker (Thread): Session for room {self.room_name} already closing or closed, no aclose needed in finally.")
                except Exception as close_e:
                    logger.error(f"Agent Worker (Thread): Error closing session for room {self.room_name} in finally: {close_e}", exc_info=True)

    async def initialize_session(self, system_prompt: str) -> None:
        logger.info(f"BetsyTeacher: Storing instructions: {system_prompt[:50]}...")
        self.agent_instructions = system_prompt

    async def connect_to_room(self, room_name: str) -> bool:
        if not self.agent_instructions:
            logger.warning("BetsyTeacher: Agent instructions not set. Call initialize_session first.")
            return False
        
        if self._worker_task and not self._worker_task.done():
            logger.warning(f"BetsyTeacher: Worker task already active for room {self.room_name}. Stopping it first.")
            await self.stop()

        self.room_name = room_name
        
        try:
            region = os.getenv("LIVEKIT_REGION")
            
            lk_url = os.getenv("LIVEKIT_URL")
            lk_api_key = os.getenv("LIVEKIT_API_KEY")
            lk_api_secret = os.getenv("LIVEKIT_API_SECRET")

            if not all([lk_url, lk_api_key, lk_api_secret]):
                logger.error("BetsyTeacher: LIVEKIT_URL, API_KEY, or API_SECRET not found. Worker will fail.")
                return False

            logger.info(f"BetsyTeacher: Preparing LiveKit Worker (Thread) for room: {self.room_name} (region: {region or 'default'}) using URL: {lk_url[:20]}...")
            
            options = WorkerOptions(
                entrypoint_fnc=self._agent_entrypoint_method,
                ws_url=lk_url, 
                api_key=lk_api_key, 
                api_secret=lk_api_secret,
                job_executor_type=JobExecutorType.THREAD
            )
            if region: 
                options.region = region
            
            worker = Worker(options)
            
            logger.info(f"BetsyTeacher: Starting LiveKit Worker (Thread) task for room {self.room_name}.")
            self._worker_task = asyncio.create_task(worker.run(), name=f"BetsyWorker-{self.room_name}")
            
            await asyncio.sleep(2) 

            if self._worker_task.done():
                try:
                    exc = self._worker_task.exception()
                    if exc:
                        logger.error(f"BetsyTeacher: Worker (Thread) task for {self.room_name} failed early: {exc}", exc_info=exc)
                        self.active = False
                        return False
                    else:
                        logger.info(f"BetsyTeacher: Worker (Thread) task for {self.room_name} completed very quickly (may be an issue).")
                        self.active = False
                        return False
                except asyncio.CancelledError:
                    logger.info(f"BetsyTeacher: Worker (Thread) task for {self.room_name} was cancelled during initial check.")
                    self.active = False
                    return False
            
            logger.info(f"BetsyTeacher: LiveKit Worker (Thread) for room {self.room_name} presumed started (task is running).")
            self.active = True
            return True

        except Exception as e:
            logger.error(f"BetsyTeacher: Error setting up or starting worker (Thread) for {self.room_name}: {e}", exc_info=True)
            self.active = False
            if self._worker_task and not self._worker_task.done():
                self._worker_task.cancel()
            return False
        
    async def stop(self) -> None:
        current_room_for_log = self.room_name or "N/A"
        logger.info(f"BetsyTeacher: Stop called for room {current_room_for_log}")
        
        task_to_stop = self._worker_task
        self._worker_task = None 
        self.active = False 

        if task_to_stop and not task_to_stop.done():
            logger.info(f"BetsyTeacher: Cancelling worker (Thread) task for room {current_room_for_log}")
            task_to_stop.cancel()
            try:
                await task_to_stop
            except asyncio.CancelledError:
                logger.info(f"BetsyTeacher: Worker (Thread) task for room {current_room_for_log} successfully cancelled.")
            except Exception as e:
                logger.error(f"BetsyTeacher: Error awaiting cancelled worker (Thread) task for {current_room_for_log}: {e}", exc_info=True)
        else:
            logger.info(f"BetsyTeacher: No active worker (Thread) task found for room {current_room_for_log} or task already done.")
        
        logger.info(f"BetsyTeacher: Stop process completed for room {current_room_for_log}.")

app = FastAPI(
    title="BETSY Agent API",
    description="API for the BETSY language teacher agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

teacher = BetsyTeacher()
class StartSessionRequest(BaseModel):
    instructions: str = Field(..., description="System prompt instructions for the language teacher")
    room_name: str = Field(..., description="LiveKit room name to join")

class SessionResponse(BaseModel):
    success: bool
    message: str
    session_id: Optional[str] = None
    room_name: Optional[str] = None

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
        
        logger.info(f"Successfully joined room: {request.room_name}")
        
        return SessionResponse(
            success=True,
            message=f"Session started successfully in room {request.room_name}",
            session_id="global", 
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
                livekit_url = os.getenv("LIVEKIT_URL")
                livekit_api_key = os.getenv("LIVEKIT_API_KEY")
                livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")

                if not all([livekit_url, livekit_api_key, livekit_api_secret]):
                    logger.error("LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET not configured. Cannot delete room.")
                else:
                    lk_api_client = livekit_api.LiveKitAPI(livekit_url, livekit_api_key, livekit_api_secret)
                    logger.info(f"Attempting to delete room: {current_room_name} from LiveKit server.")
                    delete_req = DeleteRoomRequest(room=current_room_name)
                    await lk_api_client.room.delete_room(delete_req)
                    logger.info(f"Successfully deleted room: {current_room_name} from LiveKit server.")
                    await lk_api_client.aclose()
            except Exception as delete_e:
                logger.error(f"Error deleting room {current_room_name} from LiveKit server: {delete_e}", exc_info=True)
        
        teacher.room_name = None

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

if __name__ == "__main__":
    uvicorn.run("api:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=True)