import logging
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Configure logging
# Using a hierarchical logger name is good practice.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("app.config") # Logger for this config module

# 2. Define Pydantic Settings class
class Settings(BaseSettings):
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str
    LIVEKIT_URL: str
    OPENAI_API_KEY: str
    LIVEKIT_REGION: Optional[str] = None
    LIVEKIT_ROOM_PREFIX: str = "betsy-classroom" # Default room prefix
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    model_config = SettingsConfigDict(
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra fields from .env files
    )

# 3. Load settings with custom .env.local priority
# Pydantic BaseSettings can take an _env_file argument.
# We check for .env.local, then .env, then fall back to environment variables.
# Paths are relative to the current working directory when the script/app runs.
# If main.py is in server/ and run from server/, these paths are correct.

_env_file_to_load: Optional[Path] = None
# Assuming the script/application will be run from the 'server' directory
# where 'main.py' and the '.env' files reside.
_current_working_dir = Path(os.getcwd()) # This will be server/ if main.py is run from there
_env_local_path = _current_working_dir / ".env.local"
_env_path = _current_working_dir / ".env"

logger.info(f"Attempting to load .env files from CWD: {_current_working_dir}")

if _env_local_path.exists():
    _env_file_to_load = _env_local_path
    logger.info(f"Found .env.local, preparing to load: {_env_local_path.resolve()}")
elif _env_path.exists():
    _env_file_to_load = _env_path
    logger.info(f"Found .env, preparing to load: {_env_path.resolve()}")
else:
    logger.info("No .env.local or .env file found in CWD. Will load from environment variables or defaults.")

# Instantiate settings. Pydantic will raise ValidationError if required fields are missing.
if _env_file_to_load:
    settings = Settings(_env_file=_env_file_to_load)
else:
    settings = Settings()

# 4. Log loaded/missing required variables (similar to original behavior for info)
logger.info("Verifying loaded configuration settings:")
# Define which variables are truly required (no default, not Optional)
_required_field_names = {
    field_name for field_name, model_field in Settings.model_fields.items()
    if model_field.is_required()
}

for field_name, model_field in Settings.model_fields.items():
    value = getattr(settings, field_name)
    # Determine if the field is sensitive for masking
    is_sensitive = "KEY" in field_name.upper() or "SECRET" in field_name.upper()

    if value is not None: # Check for None explicitly
        val_str = str(value)
        display_value = (val_str[:3] + '***' if len(val_str) > 3 and is_sensitive else val_str)
        logger.info(f"  {field_name}: {display_value} (SET)")
    else:
        if field_name in _required_field_names:
            # This case should ideally be caught by Pydantic's ValidationError
            logger.error(f"  {field_name}: NOT SET (CRITICAL - This field is required and was not loaded!)")
        else:
            logger.info(f"  {field_name}: NOT SET (Using default or None for optional field)")

# Pydantic's BaseSettings raises pydantic.ValidationError if required fields
# (those without a default and not Optional) are missing during instantiation.
# The `settings = Settings(...)` calls above handle this.