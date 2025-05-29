import secrets
import string
import re
import uuid
import json
from datetime import timedelta
from typing import Optional, Dict, Any

from livekit import api
from app.config import settings, logger as app_config_logger
from app.logger import ServerLogger # Import our custom logger class

# Ensure the logger instance is treated as ServerLogger for type hinting
assert isinstance(app_config_logger, ServerLogger), \
    f"Logger from app.config is not a ServerLogger instance, type: {type(app_config_logger)}"
logger: ServerLogger = app_config_logger

def _generate_random_string(length: int = 12) -> str:
    """Generates a random alphanumeric string."""
    # Using secrets for cryptographically strong random numbers
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def _clean_room_prefix(prefix: str) -> str:
    """Cleans a room prefix to ensure it's URL-safe."""
    # Remove any non-alphanumeric characters except hyphens
    # Replace spaces with hyphens, convert to lowercase
    prefix = prefix.strip().lower()
    prefix = re.sub(r'\s+', '-', prefix)
    prefix = re.sub(r'[^\w\-]', '', prefix)
    # Ensure it doesn't start or end with a hyphen
    prefix = re.sub(r'^-+|-+$', '', prefix)
    return prefix

def generate_room_name() -> str:
    """
    Generates a unique room name using the configured prefix and a random ID.
    """
    prefix = settings.LIVEKIT_ROOM_PREFIX
    random_id = _generate_random_string(12)
    clean_prefix = _clean_room_prefix(prefix)
    
    if not clean_prefix: # Handle case where prefix becomes empty after cleaning
        logger.warning("LIVEKIT_ROOM_PREFIX resulted in an empty string after cleaning. Using 'livekit-room' as a fallback prefix.")
        clean_prefix = "livekit-room"
        
    return f"{clean_prefix}-{random_id}"

def create_token(
    ttl_seconds: int = 3600
) -> Dict[str, Any]:
    """
    Generates a LiveKit access token, a unique room name, and a unique identity.

    Args:
        ttl_seconds: Token time-to-live in seconds.

    Returns:
        A dictionary containing the JWT, LiveKit URL, room_name, identity and more...

    Raises:
        ValueError: If LiveKit API key, secret, or URL is not configured.
    """
    api_key = settings.LIVEKIT_API_KEY
    api_secret = settings.LIVEKIT_API_SECRET
    livekit_url = settings.LIVEKIT_URL

    if not api_key or not api_secret or not livekit_url:
        logger.error("LiveKit API key, secret, or URL is not configured.")
        raise ValueError("LiveKit API key, secret, or URL is not configured.")

    identity = uuid.uuid4().hex
    logger.success(f"Generated identity: {identity}")

    room_name = generate_room_name()
    logger.success(f"Generated room name: {room_name} for identity: {identity}")

    video_grants = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_publish_data=True,
        can_subscribe=True
    )

    # Prepare metadata for the agent
    # Instructions are handled by the agent's initialize_session method for now.
    # OpenAI API key for the agent's LLM calls is sourced from server settings.
    metadata_payload = {
        "openai_api_key": settings.OPENAI_API_KEY, # Agent will use this for its OpenAI calls
        "voice": settings.AGENT_DEFAULT_VOICE,
        "temperature": settings.AGENT_DEFAULT_TEMPERATURE,
        "max_output_tokens": settings.AGENT_DEFAULT_MAX_RESPONSE_TOKENS,
        "modalities": settings.AGENT_DEFAULT_MODALITIES, # Agent will parse this string
        # Ensure other necessary fields expected by the agent's parse_session_config are here
        # For example, if the agent expects 'instructions' in metadata, it should be added.
        # For now, instructions are set directly on the agent instance.
    }
    logger.debug(f"Token metadata payload for agent: {metadata_payload}")

    # Create AccessToken using the builder pattern from livekit.api
    access_token_builder = api.AccessToken(api_key, api_secret) \
        .with_identity(identity) \
        .with_ttl(timedelta(seconds=ttl_seconds)) \
        .with_grants(video_grants) \
        .with_metadata(json.dumps(metadata_payload))

    jwt = access_token_builder.to_jwt()

    return {
        "token": jwt,
        "livekit_url": livekit_url,
        "room_name": room_name,
        "identity": identity
    }
