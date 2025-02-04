import logging
import os
from logging.handlers import RotatingFileHandler
from logging import Logger


def create_logger(filename: str, max_bytes: int=10_000_000, backup_count: int=1) -> Logger:
    """It creates a logger for either the server or a client."""
    
    # Clear file
    if os.path.exists(filename):
        open(filename, "w").close()
    
    # Configure logger
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevents logging to console

    formatter_server = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler_server = RotatingFileHandler(filename=filename, maxBytes=max_bytes, backupCount=backup_count)
    handler_server.namer = lambda filename: filename.replace(".log.", ".") + ".log"  # Modify naming convention: logfile.log.1 → logfile.1.log
    handler_server.setFormatter(formatter_server)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler_server)
         
    return logger


logger = create_logger("client.log")








'''
# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server.log"),  # Log to file
        # logging.StreamHandler()  # Log to console
    ]
)
server_logger_2 = logging.getLogger(__name__)
'''