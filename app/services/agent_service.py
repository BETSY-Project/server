import asyncio
import types
from typing import Optional

# Imports from the original api.py that BetsyTeacher uses
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    Worker,
    WorkerOptions,
    JobExecutorType,
    stt,
)
from livekit.plugins import openai, silero

# New imports for config and logging
from app.config import logger, settings

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
            logger.info(f"Agent Worker (Thread): Initializing for room '{self.room_name}' with instructions: '{self.agent_instructions[:50] if self.agent_instructions else 'N/A'}...'")
            
            logger.info(f"Agent Worker (Thread): Initializing LiveKit components for room '{self.room_name}'")
            
            agent = Agent(instructions=self.agent_instructions)
            
            vad_instance = None
            try:
                vad_opts_obj = types.SimpleNamespace(sample_rate=16000)
                vad_instance = silero.VAD(session=ctx.room, opts=vad_opts_obj)
                logger.info(f"Agent Worker (Thread): Silero VAD instantiated with ctx.room and opts: {vad_opts_obj}")
            except Exception as e_vad:
                logger.error(f"Agent Worker (Thread): Failed to instantiate silero.VAD: {e_vad}", exc_info=True)

            openai_stt = openai.STT()
            if vad_instance:
                adapted_stt = stt.StreamAdapter(stt=openai_stt, vad=vad_instance)
                logger.info("Agent Worker (Thread): OpenAI STT wrapped with StreamAdapter using Silero VAD.")
            else:
                adapted_stt = openai_stt
                logger.warning("Agent Worker (Thread): Silero VAD instantiation failed. Passing raw OpenAI STT to AgentSession. Expecting STT streaming error.")

            session = AgentSession(
                stt=adapted_stt,
                llm=openai.LLM(model="gpt-4o"),
                tts=openai.TTS(voice="alloy"),
                vad=vad_instance,
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
                    logger.info("Agent Worker (Thread): Attempting to say a welcome message.")
                    await session.generate_reply(instructions="Dis bonjour et présente-toi brièvement.")
                    logger.info("Agent Worker (Thread): Welcome message task created/finished.")
                except RuntimeError as e_runtime: 
                    logger.error(f"Agent Worker (Thread): RuntimeError trying to say welcome message (session started: {session._started}): {e_runtime}", exc_info=True)
                except Exception as e_say:
                    logger.error(f"Agent Worker (Thread): Generic error trying to say welcome message: {e_say}", exc_info=True)
            elif hasattr(session, '_started') and not session._started :
                 logger.warning(f"Agent Worker (Thread): Session not in a 'started' state (session._started = {session._started}), cannot say welcome message.")
            elif not ctx.room.isconnected:
                 logger.warning("Agent Worker (Thread): Room disconnected, cannot say welcome message.")

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
            # Use settings object instead of os.getenv
            region = settings.LIVEKIT_REGION
            
            lk_url = settings.LIVEKIT_URL
            lk_api_key = settings.LIVEKIT_API_KEY
            lk_api_secret = settings.LIVEKIT_API_SECRET

            if not all([lk_url, lk_api_key, lk_api_secret]):
                # This check might be redundant if Pydantic settings are correctly configured as required
                logger.error("BetsyTeacher: LIVEKIT_URL, API_KEY, or API_SECRET not found in settings. Worker will fail.")
                return False

            logger.info(f"BetsyTeacher: Preparing LiveKit Worker (Thread) for room: {self.room_name} (region: {region or 'default'}) using URL: {lk_url[:20] if lk_url else 'N/A'}...")
            
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