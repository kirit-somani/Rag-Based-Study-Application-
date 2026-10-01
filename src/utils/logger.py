import logging
import sys
from pathlib import Path

def setup_logger(name: str = "studyrag") -> logging.Logger:
    """Sets up a structured logger for the application."""
    logger = logging.getLogger(name)
    
    # Avoid duplicate logs if logger is already configured
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()
