import logging
import requests
import json
import os
import traceback
from app.config import settings

# Construct CLM endpoint from settings
CLM_ENDPOINT = f"{settings.CLM_URL.rstrip('/')}/log" # CLM service endpoint

# Mapping Python log levels to CLM string levels
LOG_LEVEL_MAP = {
    logging.DEBUG: "info", # Or "debug" if CLM supports it
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "error", # Or "critical"
}

class CustomLogManagerHandler(logging.Handler):
    """
    A custom logging handler that sends log records to the Custom Log Manager (CLM) service.
    """
    def __init__(self, service_name="server"):
        super().__init__()
        self.service_name = service_name

    def emit(self, record: logging.LogRecord):
        """
        Formats and sends the log record to the CLM service.
        """
        log_level_str = "unknown" # Initialize with a default
        try:
            log_level_str = LOG_LEVEL_MAP.get(record.levelno, "info") # Re-assign if possible
            
            message = self.format(record) # Get the formatted message
            
            log_entry = {
                "service": self.service_name,
                "level": log_level_str,
                "message": message,
            }

            if record.exc_info:
                # Add formatted traceback as details if exception info is present
                log_entry["details"] = "".join(traceback.format_exception(*record.exc_info))

            requests.post(CLM_ENDPOINT, json=log_entry, timeout=2) # Fire and forget with a timeout
        except requests.exceptions.RequestException as e:
            # Fallback to console if CLM is unavailable or there's an error
            print(f"Failed to send log to CLM: {e}")
            print(f"Original log ({self.service_name} - {log_level_str}): {self.format(record)}")
        except Exception as e:
            print(f"Unexpected error in CustomLogManagerHandler: {e}")
            print(f"Original log ({self.service_name} - {log_level_str}): {self.format(record)}")


def get_server_logger(name: str = 'betsy-server', 
                      level: int = logging.INFO, 
                      service_name_for_clm: str = "server"):
    """
    Configures and returns a logger instance for server-side logging.
    Sends logs to CLM and optionally to console.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        logger.setLevel(level)

        # Formatter for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # CLM Handler
        clm_handler = CustomLogManagerHandler(service_name=service_name_for_clm)
        clm_handler.setLevel(level)
        clm_handler.setFormatter(formatter) # Formatter applied to message sent to CLM
        logger.addHandler(clm_handler)

        # Console Handler (optional, for local debugging)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.propagate = False # Avoid passing to root logger if it has handlers

    return logger

# Example usage (can be removed or kept for testing this module directly):
if __name__ == "__main__":
    # Configure a logger for testing
    test_logger = get_server_logger(name="my-test-app", level=logging.DEBUG)

    test_logger.debug("This is a debug message from the server.")
    test_logger.info("Server process started successfully.")
    test_logger.warning("A minor issue occurred, but operation continues.")
    test_logger.error("An error occurred while processing a request.")
    try:
        x = 1 / 0
    except ZeroDivisionError:
        test_logger.critical("Critical failure: Division by zero!", exc_info=True)
    
    print(f"Test logs sent. Check CLM at {CLM_ENDPOINT.replace('/log', '')} and console.")