from __future__ import annotations

"""
Manages the Betsy voice agent, a LiveKit worker that facilitates real-time,
interactive lessons.

The agent connects to a LiveKit room and acts as a virtual teacher,
utilizing OpenAI's services for text generation and speech synthesis/recognition.
It handles session management, participant interaction, and configuration
based on participant metadata.
"""

import asyncio
import json
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel

import inspect
from contextlib import suppress
import livekit.rtc as rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobExecutorType,
    Worker,
    WorkerOptions,
)
from livekit.plugins import openai
from app.config import logger, settings

class Modality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"

class _AgentConfig(BaseModel):
    """
    Configuration for the Betsy agent, derived from participant metadata.
    Specifies OpenAI API key, voice, temperature, token limits, and modalities.
    """
    openai_api_key: str
    voice: str = "alloy"
    temperature: float = 0.8
    max_response_output_tokens: int | Literal["inf"] = "inf"
    modalities: list[Modality] = [Modality.TEXT, Modality.AUDIO]

    @classmethod
    def from_metadata(cls, meta: str) -> Optional["_AgentConfig"]:
        """
        Parse participant metadata JSON and validate it.
        Returns None if the payload is invalid.
        """
        try:
            data = json.loads(meta)
            raw_modalities = str(data.get("modalities", "text_and_audio")).lower()
            mapping = {
                "text_and_audio": [Modality.TEXT, Modality.AUDIO],
                "text_only": [Modality.TEXT],
                "audio_only": [Modality.AUDIO],
            }
            return cls(
                openai_api_key=str(data["openai_api_key"]),
                voice=str(data.get("voice", "alloy")),
                temperature=float(data.get("temperature", 0.8)),
                max_response_output_tokens=(
                    "inf"
                    if str(data.get("max_output_tokens", "inf")).lower() == "inf"
                    else int(data["max_output_tokens"])
                ),
                modalities=mapping.get(raw_modalities, mapping["text_and_audio"]),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.exception("Failed to parse participant metadata: %s", exc)
            return None

class BetsyTeacher:
    """Manages a LiveKit worker hosting the Betsy conversational agent."""

    def __init__(self) -> None:
        self.agent_instructions: Optional[str] = None
        self.active: bool = False
        self.room_name: Optional[str] = None

        self._worker_task: Optional[asyncio.Task[None]] = None
        self._session: Optional[AgentSession] = None

    # ----------------- FastAPI helpers ----------------- #
    async def initialize_session(self, system_prompt: str) -> None:
        """Store the system prompt that primes the LLM."""
        preview = (system_prompt[:100] + "…") if len(system_prompt) > 100 else system_prompt
        full_instructions = (
            f"You are Betsy! And you NEVER change your name! You are Betsy: {system_prompt}"
        )
        self.agent_instructions = full_instructions
        logger.info("Stored Betsy instructions (preview): %s", preview)

    async def connect_to_room(self, room_name: str) -> bool:
        """
        Connects the Betsy agent to the specified LiveKit room.
    
        If a worker is already running, it's stopped before a new one is started.
        Initializes and starts a LiveKit worker task.
    
        Args:
            room_name: The name of the LiveKit room to connect to.
    
        Returns:
            True if the connection was successful, False otherwise.
        """
        if not self.agent_instructions:
            logger.error("initialize_session must be called first")
            return False

        if self._worker_task and not self._worker_task.done():
            await self.stop()

        self.room_name = room_name

        try:
            options = WorkerOptions(
                entrypoint_fnc=self._agent_entrypoint_method,
                ws_url=settings.LIVEKIT_URL,
                api_key=settings.LIVEKIT_API_KEY,
                api_secret=settings.LIVEKIT_API_SECRET,
                job_executor_type=JobExecutorType.THREAD,
            )
            worker = Worker(options)
            self._worker_task = asyncio.create_task(worker.run(), name=f"BetsyWorker-{room_name}")
            await asyncio.sleep(0.5)  # let the worker start
            self.active = True
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Failed to start worker")
            self.active = False
            return False

    async def stop(self) -> None:
        """Terminate the worker and close the session, if any."""
        self.active = False

        if self._session:
            try:
                await self._session.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error closing AgentSession")
            self._session = None

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await asyncio.shield(self._worker_task)
            except asyncio.CancelledError:
                pass
        self._worker_task = None
        self.room_name = None

    # ----------------- LiveKit worker entrypoint ----------------- #
    async def _agent_entrypoint_method(self, ctx: JobContext) -> None:
        """
        Define the entrypoint for the LiveKit agent worker.
    
        This method is invoked when the worker commences its operation. It establishes
        a connection to the designated room, awaits the arrival of a participant
        possessing metadata, configures the agent using this metadata, initializes
        OpenAI services (both LLM and TTS), and subsequently initiates an AgentSession.
        The session is then maintained in an active state.
    
        Args:
            ctx: The job context provided by LiveKit Agents.
        """
        logger.info("[entrypoint] Worker for room '%s'", self.room_name)

        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # Wait for a remote participant that has metadata
        loop = asyncio.get_running_loop()

        # Future to signal when a suitable remote participant (one with metadata) connects.
        participant_future: asyncio.Future[rtc.RemoteParticipant] = loop.create_future()

        def _on_participant_connected(participant: rtc.RemoteParticipant) -> None:
            if (
                participant.identity != ctx.room.local_participant.identity
                and participant.metadata
                and not participant_future.done()
            ):
                participant_future.set_result(participant)

        # Register listener before checking any already‑connected participants
        ctx.room.on("participant_connected", _on_participant_connected)

        # Handle the case where the participant joined earlier
        for p in ctx.room.remote_participants.values():
            if p.identity != ctx.room.local_participant.identity and p.metadata:
                participant_future.set_result(p)
                break

        try:
            human = await asyncio.wait_for(participant_future, timeout=30)
        except asyncio.TimeoutError:
            logger.error("No participant with metadata joined within 30 s - aborting")
            return

        cfg = _AgentConfig.from_metadata(human.metadata)
        if not cfg or not self.agent_instructions:
            logger.error("Invalid participant metadata or missing instructions - aborting")
            return

        # ------------------------------------------------------------------
        # Build LLM, TTS and start exactly ONE AgentSession.
        # If the realtime socket dies, we exit; upper layers may spawn
        # a fresh worker instead of piling up handlers inside this one.
        # ------------------------------------------------------------------

        # Build kwargs defensively (older plugin versions may miss params)
        rt_kwargs = {
            "api_key": cfg.openai_api_key,
            "model": "gpt-4o-realtime-preview",
            "temperature": cfg.temperature,
        }
        sig = inspect.signature(openai.realtime.RealtimeModel)
        if "modalities" in sig.parameters:
            rt_kwargs["modalities"] = [m.value for m in cfg.modalities]
        if "max_response_output_tokens" in sig.parameters:
            rt_kwargs["max_response_output_tokens"] = cfg.max_response_output_tokens

        rt_model = openai.realtime.RealtimeModel(**rt_kwargs)
        tts_engine = openai.TTS(voice=cfg.voice, api_key=cfg.openai_api_key)

        self._session = AgentSession(llm=rt_model, tts=tts_engine)

        try:
            await self._session.start(
                room=ctx.room,
                agent=Agent(
                    instructions=self.agent_instructions,
                    llm=rt_model,
                    tts=tts_engine,
                ),
            )
            # ------------------------------------------------------------
            # Kick-off greeting (agent speaks first) WITHOUT overriding
            # the system prompt. We send an empty user_input, which the
            # agent interprets as “speak first”. Interruptions remain
            # allowed (default) to stay compatible with server-side
            # turn detection.
            # ------------------------------------------------------------
            greet_handle = await self._session.generate_reply(user_input="")
            await greet_handle.wait_for_playout()

            # Wait until the session stops (normal or error)
            await self._session.wait_closed()

        except Exception as exc:
            logger.warning("AgentSession aborted: %s", exc)

        finally:
            # Clean up cleanly
            with suppress(Exception):
                await self._session.close()
            self._session = None

            ctx.room.off("participant_connected", _on_participant_connected)

        logger.info("[entrypoint] Worker for room '%s' finished.", self.room_name)
