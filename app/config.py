from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Define Pydantic Settings class
class Settings(BaseSettings):
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str
    LIVEKIT_URL: str
    OPENAI_API_KEY: str
    LIVEKIT_ROOM_PREFIX: str
    PORT: int
    HOST: str
    CLM_URL: str
    AGENT_DEFAULT_VOICE: str
    AGENT_DEFAULT_TEMPERATURE: float
    AGENT_DEFAULT_MAX_RESPONSE_TOKENS: int
    AGENT_DEFAULT_MODALITIES: str

    model_config = SettingsConfigDict(
        env_file=(".env.local", ".env"), # Load from .env.local first, then .env
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra fields from .env files
    )

# Load settings instance
# Pydantic will raise ValidationError if required fields are missing from env vars or .env files.
settings = Settings()

# Initialize and export the main application logger
# This must come *after* `settings` is initialized as `get_server_logger` may use `settings.CLM_URL`.
from app.logger import get_server_logger
logger = get_server_logger(name="app.config", service_name_for_clm="server")

# Log confirmation that configuration loading was attempted (and succeeded if no crash).
logger.info("Configuration loaded.")