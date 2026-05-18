import logging
import sys

_SPWM_LOGGER_NAME = "guanwu.video"

def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Configures the root guanwu.video logger with a standardized format."""
    logger = logging.getLogger(_SPWM_LOGGER_NAME)
    
    # If the logger already has handlers, assume it's configured. Avoid duplicated logs.
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Do not propagate to root python logger by default to avoid duplicate outputs
    # in environments that auto-configure root loggers (like uvicorn/fastapi roots)
    logger.propagate = False
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Returns a child logger under the guanwu.video hierarchy."""
    if not name.startswith(_SPWM_LOGGER_NAME):
        return logging.getLogger(f"{_SPWM_LOGGER_NAME}.{name}")
    return logging.getLogger(name)
