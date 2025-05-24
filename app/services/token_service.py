# server/app/services/token_service.py
import secrets
import string
import re
import uuid
from datetime import timedelta
from typing import Optional, Dict, Any

from livekit import api
from app.config import settings, logger

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
    region: Optional[str] = None,
    ttl_seconds: int = 3600
) -> Dict[str, Any]:
    """
    Generates a LiveKit access token, a unique room name, and a unique identity.

    Args:
        region: Optional region for LiveKit Cloud.
        ttl_seconds: Token time-to-live in seconds.

    Returns:
        A dictionary containing the JWT, LiveKit URL, room_name, identity, and region.

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
    logger.info(f"Generated identity: {identity}")

    room_name = generate_room_name()
    logger.info(f"Generated room name: {room_name} for identity: {identity}")

    video_grants = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_publish_data=True,
        can_subscribe=True
        # Note: 'region' is not a direct field in VideoGrants for livekit-api Python SDK.
        # It's handled at the client connection level or via LiveKit Cloud project settings.
        # We will return it for the client to use if needed.
    )

    # Create AccessToken using the builder pattern from livekit.api
    access_token = api.AccessToken(api_key, api_secret) \
        .with_identity(identity) \
        .with_ttl(timedelta(seconds=ttl_seconds)) \
        .with_grants(video_grants)

    jwt = access_token.to_jwt()

    return {
        "token": jwt,
        "livekit_url": livekit_url,
        "room_name": room_name,
        "identity": identity,
        "region": region
    }