import logging
import requests
import json
import os
import traceback

# Define custom SUCCESS log level
SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

# Mapping Python log levels to CLM string levels
LOG_LEVEL_MAP = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    SUCCESS_LEVEL: "success", # Custom success level
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}

# 1. Define our custom logger class
class ServerLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(SUCCESS_LEVEL):
            # The _log method is the standard way to issue messages from custom methods
            self._log(SUCCESS_LEVEL, message, args, **kwargs)

# 2. Tell the logging system to use our custom logger class
# This must be done BEFORE any loggers are instantiated via getLogger()
# It's crucial this line is executed early in the module's import lifecycle.
logging.setLoggerClass(ServerLogger)


class CustomLogManagerHandler(logging.Handler):
    """
    A custom logging handler that sends log records to the Custom Log Manager (CLM) service.
    Requires clm_endpoint to be provided.
    """
    def __init__(self, service_name="server", clm_endpoint=None):
        super().__init__()
        self.service_name = service_name
        if clm_endpoint is None:
            # This indicates a programming error if get_server_logger is supposed to provide it.
            # Log to stderr or raise an error, as this handler is non-functional without an endpoint.
            # For now, just print and disable.
            print(f"ERROR: CustomLogManagerHandler for service '{service_name}' initialized without a clm_endpoint. CLM logging will be disabled for this handler.")
            self.clm_endpoint = None
        else:
            self.clm_endpoint = clm_endpoint

    def emit(self, record: logging.LogRecord):
        if not self.clm_endpoint:
            # If endpoint is not set, do not attempt to send to CLM.
            # A console handler should exist on the logger for fallback.
            return

        log_level_str = LOG_LEVEL_MAP.get(record.levelno, "info") # Default to "info"
        
        # Get the raw message, not the fully formatted one for CLM
        actual_message = record.getMessage()
        
        details_payload = None
        if record.exc_info:
            if details_payload is None:
                details_payload = {}
            details_payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
        
        # For console fallback in case of CLM failure, we might still want the formatted message.
        # The self.format(record) applies the formatter associated with this handler.
        formatted_for_console_fallback = self.format(record)

        try:
            log_entry = {
                "service": self.service_name,
                "level": log_level_str,
                "message": actual_message,
            }
            if details_payload is not None: # Only add 'details' field if there's something to include
                log_entry["details"] = details_payload
            
            requests.post(self.clm_endpoint, json=log_entry, timeout=2)
        except requests.exceptions.RequestException as e:
            # Fallback to console with the originally intended formatted message
            print(f"Failed to send log to CLM for service '{self.service_name}': {e}")
            print(f"Original log ({self.service_name} - {log_level_str} - from {record.name}): {formatted_for_console_fallback}")
        except Exception as e:
            # Fallback to console
            print(f"Unexpected error in CustomLogManagerHandler.emit for service '{self.service_name}': {e}")
            print(f"Original log ({self.service_name} - {log_level_str} - from {record.name}): {formatted_for_console_fallback}")


def get_server_logger(name: str = 'betsy-server',
                      level: int = logging.INFO,
                      service_name_for_clm: str = "server"):
    
    # get_server_logger is now responsible for obtaining CLM_URL via settings.
    # This means this function should only be called when app.config.settings is available.
    resolved_clm_endpoint = None
    try:
        from app.config import settings as app_settings # Import settings here
        if hasattr(app_settings, 'CLM_URL') and app_settings.CLM_URL:
            resolved_clm_endpoint = f"{app_settings.CLM_URL.rstrip('/')}/log"
        else:
            print(f"WARNING: get_server_logger for '{name}': CLM_URL not found or empty in app.config.settings. CLM logging disabled.")
    except ImportError:
        print(f"WARNING: get_server_logger for '{name}': Could not import app.config.settings. CLM logging disabled.")
    except Exception as e:
        print(f"WARNING: get_server_logger for '{name}': Error accessing CLM_URL from settings: {e}. CLM logging disabled.")
    """
    Configures and returns a logger instance for server-side logging.
    Attempts to send logs to CLM. If CLM is unavailable during setup,
    it falls back to console-only logging. Individual CLM send failures
    are also logged to console by the CLM handler itself.
    """
    logger = logging.getLogger(name)

    if not logger.handlers: # Avoid re-adding handlers
        logger.setLevel(level) # Set the overall minimum level for the logger
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if resolved_clm_endpoint:
            # CLM Handler - added if endpoint was resolved
            clm_handler = CustomLogManagerHandler(
                service_name=service_name_for_clm,
                clm_endpoint=resolved_clm_endpoint
            )
            clm_handler.setLevel(level)
            clm_handler.setFormatter(formatter)
            logger.addHandler(clm_handler)
            # The CustomLogManagerHandler itself will print to console if a send fails.
        else:
            # Fallback to console-only if CLM endpoint couldn't be resolved at setup.
            # A warning about CLM being disabled would have been printed by the try/except block above.
            print(f"INFO: get_server_logger for '{name}': Adding console-only handler as CLM is not configured.")
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        logger.propagate = False # Prevent passing to parent loggers

    return logger

if __name__ == "__main__":
    print("--- Running server/app/logger.py direct test ---")
    # This test will likely show warnings about CLM being disabled,
    # as app.config.settings might not be available or fully configured
    # when logger.py is run directly without the main application context.
    
    # To make this test more useful, we might need to mock app.config.settings
    # or provide a dummy CLM_URL if settings cannot be imported.
    # For now, it will demonstrate the fallback or CLM attempt.

    # Attempt to get a logger instance (might warn if app.config is not found)
    test_logger_instance = get_server_logger(name="logger-direct-test", level=logging.DEBUG)
    test_logger_instance.info("This is an INFO log from logger.py direct execution.")
    test_logger_instance.debug("This is a DEBUG log from logger.py direct execution.")
    try:
        1 / 0
    except ZeroDivisionError:
        test_logger_instance.error("A test error from logger.py direct execution.", exc_info=True)
    
    # Test with an explicit (dummy) CLM URL if settings import fails
    # This part is more complex to set up without mocking app_settings properly.
    # For simplicity, the current test relies on the existing get_server_logger behavior.

    print("--- Finished server/app/logger.py direct test ---")