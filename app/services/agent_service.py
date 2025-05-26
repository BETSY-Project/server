import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import livekit.rtc
from livekit.agents import (
    JobContext,
    Worker,
    WorkerOptions,
    JobExecutorType,
    AutoSubscribe,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai

from app.config import logger, settings

# Define a dataclass to hold parsed agent configuration from metadata
@dataclass
class _AgentConfig:
    openai_api_key: str
    voice: openai.realtime.api_proto.Voice
    temperature: float
    max_response_output_tokens: int | str
    modalities: List[openai.realtime.api_proto.Modality]
    turn_detection: openai.realtime.ServerVadOptions

    # Helper to convert string modalities from metadata to enum list
    @staticmethod
    def _modalities_from_metadata_str(modalities_str: str) -> List[openai.realtime.api_proto.Modality]:
        modalities_map = {
            "text_and_audio": [
                openai.realtime.api_proto.Modality.MODALITY_TEXT,
                openai.realtime.api_proto.Modality.MODALITY_AUDIO,
            ],
            "text_only": [openai.realtime.api_proto.Modality.MODALITY_TEXT],
            "audio_only": [openai.realtime.api_proto.Modality.MODALITY_AUDIO],
        }
        selected_modalities = modalities_map.get(modalities_str.lower())
        if selected_modalities is None:
            logger.warning(f"Unknown modalities string '{modalities_str}', defaulting to text_and_audio.")
            return modalities_map["text_and_audio"]
        return selected_modalities

    # Helper to parse turn_detection from metadata
    @staticmethod
    def _turn_detection_from_metadata_str(turn_detection_str: str) -> openai.realtime.ServerVadOptions:
        try:
            params = json.loads(turn_detection_str)
            return openai.realtime.ServerVadOptions(
                threshold=float(params.get("threshold", 0.5)),
                silence_duration_ms=int(params.get("silence_duration_ms", 300)),
                prefix_padding_ms=int(params.get("prefix_padding_ms", 200)),
            )
        except Exception as e:
            logger.error(f"Failed to parse turn_detection from metadata: {e}. Using default VAD options.", exc_info=True)
            return openai.realtime.DEFAULT_SERVER_VAD_OPTIONS


def _parse_agent_config_from_metadata(metadata_str: str) -> Optional[_AgentConfig]:
    """
    Parses the agent configuration from the JSON string in participant metadata.
    """
    try:
        data = json.loads(metadata_str)
        logger.info(f"Agent: Parsing metadata for configuration: {data}")

        # Validate required fields from metadata
        required_keys = ["openai_api_key", "voice", "temperature", "max_output_tokens", "modalities", "turn_detection"]
        for key in required_keys:
            if key not in data:
                logger.error(f"Agent: Missing required key '{key}' in metadata.")
                return None

        voice_str = data.get("voice", "alloy").lower()
        # Map string to Voice enum, default to ALLOY if unknown
        voice_enum = getattr(openai.realtime.api_proto.Voice, voice_str.upper(), openai.realtime.api_proto.Voice.ALLOY)
        if voice_enum != getattr(openai.realtime.api_proto.Voice, voice_str.upper(), None): # Check if it was a valid voice
             logger.warning(f"Agent: Unknown voice '{voice_str}' in metadata, defaulting to ALLOY.")


        return _AgentConfig(
            openai_api_key=str(data["openai_api_key"]),
            voice=voice_enum,
            temperature=float(data["temperature"]),
            max_response_output_tokens=str(data["max_output_tokens"]) if str(data["max_output_tokens"]).lower() == "inf" else int(data["max_output_tokens"]),
            modalities=_AgentConfig._modalities_from_metadata_str(str(data["modalities"])),
            turn_detection=_AgentConfig._turn_detection_from_metadata_str(str(data["turn_detection"])),
        )
    except Exception as e:
        logger.error(f"Agent: Failed to parse agent configuration from metadata: {e}", exc_info=True)
        return None


class BetsyTeacher:
    def __init__(self):
        self.agent_instructions: Optional[str] = None
        self.active: bool = False
        self.room_name: Optional[str] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._agent_instance: Optional[MultimodalAgent] = None
        self._realtime_model_instance: Optional[openai.realtime.RealtimeModel] = None

    async def _agent_entrypoint_method(self, ctx: JobContext):
        logger.info(f"Betsy Agent: Entrypoint started for room '{self.room_name}'.")
        human_participant: Optional[livekit.rtc.RemoteParticipant] = None
        parsed_agent_config: Optional[_AgentConfig] = None

        try:
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
            logger.info(f"Betsy Agent: Connected to room '{ctx.room.name}'. Waiting for human participant.")

            # Wait for the human participant and their metadata
            participant_search_timeout = 30.0  # seconds
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < participant_search_timeout:
                if not ctx.room.isconnected:
                    logger.warning("Betsy Agent: Room disconnected during participant search.")
                    return

                for p_sid, p_info in ctx.room.remote_participants.items():
                    # Assuming the human participant is not the agent itself
                    if p_info.identity != ctx.room.local_participant.identity and p_info.metadata:
                        remote_p = ctx.room.remote_participants.get(p_sid)
                        if remote_p and remote_p.metadata:
                            human_participant = remote_p
                            logger.info(f"Betsy Agent: Found human participant '{human_participant.identity}' with metadata.")
                            parsed_agent_config = _parse_agent_config_from_metadata(human_participant.metadata)
                            break
                if parsed_agent_config:
                    break
                await asyncio.sleep(0.5)

            if not human_participant or not parsed_agent_config:
                logger.error("Betsy Agent: Could not find human participant with valid metadata. Exiting.")
                return

            if not self.agent_instructions:
                logger.error("Betsy Agent: Agent instructions not initialized. Exiting.")
                return

            logger.info(f"Betsy Agent: Initializing RealtimeModel with config: {parsed_agent_config} and instructions.")

            self._realtime_model_instance = openai.realtime.RealtimeModel(
                api_key=parsed_agent_config.openai_api_key,
                instructions=self.agent_instructions,
                voice=parsed_agent_config.voice,
                temperature=parsed_agent_config.temperature,
                max_response_output_tokens=parsed_agent_config.max_response_output_tokens,
                modalities=parsed_agent_config.modalities,
                turn_detection=parsed_agent_config.turn_detection,
            )
            logger.info("Betsy Agent: RealtimeModel initialized.")

            self._agent_instance = MultimodalAgent(model=self._realtime_model_instance)
            logger.info("Betsy Agent: MultimodalAgent initialized. Starting agent in room.")
            
            # Start the agent (this will handle STT, LLM, TTS internally)
            self._agent_instance.start(ctx.room)
            
            # Optionally, send an initial message to the LLM if audio modality is present
            # This helps in initiating the conversation from the agent's side.
            if self._realtime_model_instance.sessions and openai.realtime.api_proto.Modality.MODALITY_AUDIO in parsed_agent_config.modalities:
                session = self._realtime_model_instance.sessions[0]
                logger.info("Betsy Agent: Sending initial greeting/prompt to LLM.")
                session.conversation.item.create(
                    llm.ChatMessage(
                        role=llm.ChatRole.USER, # Or SYSTEM, depending on desired effect
                        content="Please greet the user and introduce yourself as Betsy, consistent with your instructions."
                    )
                )
                session.response.create() # Trigger the LLM response

            logger.info("Betsy Agent: MultimodalAgent started and running.")
            # The agent will run until the job is cancelled or the room is closed.
            # We can await a join or a specific event if needed for graceful shutdown.
            # For now, allow the worker's run loop to manage its lifetime.
            # Example: await self._agent_instance.join() # if join method exists and is appropriate
            
            # Keep the entrypoint alive while the agent is active
            while ctx.room.isconnected and self.active: # self.active is managed by connect_to_room/stop
                await asyncio.sleep(1)
            logger.info("Betsy Agent: Exiting entrypoint loop (room disconnected or agent stopped).")

        except asyncio.CancelledError:
            logger.info(f"Betsy Agent: Entrypoint for room '{self.room_name}' was cancelled.")
        except Exception as e:
            logger.error(f"Betsy Agent: Error in entrypoint for room '{self.room_name}': {e}", exc_info=True)
        finally:
            logger.info(f"Betsy Agent: Cleaning up entrypoint for room '{self.room_name}'.")
            if self._agent_instance:
                try:
                    # Gracefully close the agent and its resources
                    # The specific close method might depend on the MultimodalAgent implementation
                    # For now, we assume it cleans up its model upon its own closure or job cancellation.
                    # If an explicit close is needed for MultimodalAgent or RealtimeModel, call it here.
                    # e.g., await self._agent_instance.aclose() or await self._realtime_model_instance.aclose()
                    logger.info("Betsy Agent: MultimodalAgent cleanup (if any specific method is available).")
                except Exception as e_close:
                    logger.error(f"Betsy Agent: Error during agent cleanup: {e_close}", exc_info=True)
            
            if ctx.room and ctx.room.isconnected:
                logger.info(f"Betsy Agent: Disconnecting from room '{ctx.room.name}' in finally block.")
                await ctx.room.disconnect()
            logger.info(f"Betsy Agent: Entrypoint for room '{self.room_name}' finished cleanup.")
            self._agent_instance = None
            self._realtime_model_instance = None


    async def initialize_session(self, system_prompt: str) -> None:
        # Ensure instructions are not overly verbose for logging but store the full prompt.
        log_prompt = system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
        full_instructions = f"You are Betsy! And you NEVER change your name! Your are Betsy: {system_prompt}"
        logger.info(f"BetsyTeacher: Storing agent instructions (preview: '{log_prompt}')")
        self.agent_instructions = full_instructions

    async def connect_to_room(self, room_name: str) -> bool:
        if not self.agent_instructions:
            logger.error("BetsyTeacher: Agent instructions not set. Call initialize_session first.")
            return False
        
        if self._worker_task and not self._worker_task.done():
            logger.warning(f"BetsyTeacher: Worker task already active for room '{self.room_name}'. Stopping it first.")
            await self.stop()

        self.room_name = room_name
        
        try:
            region = settings.LIVEKIT_REGION
            
            lk_url = settings.LIVEKIT_URL
            lk_api_key = settings.LIVEKIT_API_KEY
            lk_api_secret = settings.LIVEKIT_API_SECRET

            if not all([lk_url, lk_api_key, lk_api_secret]):
                logger.error("BetsyTeacher: LIVEKIT_URL, API_KEY, or API_SECRET not found in settings. Worker cannot start.")
                return False

            logger.info(f"BetsyTeacher: Preparing LiveKit Worker for room: '{self.room_name}' (region: {region or 'default'})")
            
            options = WorkerOptions(
                entrypoint_fnc=self._agent_entrypoint_method,
                ws_url=lk_url,
                api_key=lk_api_key,
                api_secret=lk_api_secret,
                job_executor_type=JobExecutorType.THREAD # Consider THREAD or PROCESS based on needs
            )
            # region is typically handled by the ws_url for LiveKit Cloud or specific SDK configurations.
            # If WorkerOptions has a direct region param and it's needed, set it:
            # if region: options.region = region 
            
            worker = Worker(options)
            
            logger.info(f"BetsyTeacher: Starting LiveKit Worker task for room '{self.room_name}'.")
            self._worker_task = asyncio.create_task(worker.run(), name=f"BetsyWorker-{self.room_name}")
            
            # Give a moment for the worker to potentially start and connect
            await asyncio.sleep(1) 

            if self._worker_task.done(): 
                try:
                    exc = self._worker_task.exception()
                    if exc:
                        logger.error(f"BetsyTeacher: Worker task for '{self.room_name}' failed early: {exc}", exc_info=exc)
                        self.active = False
                        return False
                    else: 
                        logger.info(f"BetsyTeacher: Worker task for '{self.room_name}' completed very quickly.")
                        # This might be okay if the entrypoint is short-lived by design, but our agent should be long-running.
                except asyncio.CancelledError:
                    logger.info(f"BetsyTeacher: Worker task for '{self.room_name}' was cancelled during initial check.")
                    self.active = False
                    return False
            
            logger.info(f"BetsyTeacher: LiveKit Worker for room '{self.room_name}' task is running.")
            self.active = True 
            return True

        except Exception as e:
            logger.error(f"BetsyTeacher: Error setting up or starting worker for '{self.room_name}': {e}", exc_info=True)
            self.active = False
            if self._worker_task and not self._worker_task.done():
                self._worker_task.cancel()
            return False
        
    async def stop(self) -> None:
        current_room_name = self.room_name or "N/A"
        logger.info(f"BetsyTeacher: Stop called for room '{current_room_name}'.")
        
        self.active = False # Signal the entrypoint loop to exit
        task_to_stop = self._worker_task
        self._worker_task = None 

        if task_to_stop and not task_to_stop.done():
            logger.info(f"BetsyTeacher: Cancelling worker task for room '{current_room_name}'.")
            task_to_stop.cancel()
            try:
                await asyncio.wait_for(task_to_stop, timeout=5.0) # Wait for cancellation/completion
            except asyncio.CancelledError:
                logger.info(f"BetsyTeacher: Worker task for room '{current_room_name}' successfully cancelled.")
            except asyncio.TimeoutError:
                logger.warning(f"BetsyTeacher: Timeout waiting for worker task of room '{current_room_name}' to cancel.")
            except Exception as e:
                logger.error(f"BetsyTeacher: Error awaiting cancelled worker task for '{current_room_name}': {e}", exc_info=True)
        else:
            logger.info(f"BetsyTeacher: No active worker task found for room '{current_room_name}' or task already done.")
        
        logger.info(f"BetsyTeacher: Stop process completed for room '{current_room_name}'.")