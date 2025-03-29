import logging
import os
from logging import Logger
from logging.handlers import RotatingFileHandler

import pandas as pd
from sklearn.utils import shuffle


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    for col in data.columns:
        data[col] = data[col].astype(str).str.replace(',', '.')
    
    data['case_csPCa'] = data['case_csPCa'].map({'YES': 1, 'NO': 0}).astype(int)
    
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Replace NaN values with column medians
    if data.isnull().sum().sum() > 0:
        data = data.apply(lambda col: col.fillna(col.median()) if col.isnull().any() else col)

    data = data.drop(columns=["study_id"])
    data_shuffled = shuffle(data)
    
    print(data)
    return data_shuffled



def create_logger(file_name: str, max_bytes: int=10_000_000, backup_count: int=1) -> Logger:
    """It creates a logger for either the server or a client."""
    
    folder_name = "logs"
    file_path = os.path.join(folder_name, file_name)
    os.makedirs(folder_name, exist_ok=True)
    
    # Clear file
    if os.path.exists(file_path):
        open(file_path, "w").close()
    
    # Configure logger
    logger = logging.getLogger(file_path)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevents logging to console

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = RotatingFileHandler(filename=file_path, maxBytes=max_bytes, backupCount=backup_count)
    handler.namer = lambda filename: filename.replace(".log.", ".") + ".log"  # Modify naming convention: logfile.log.1 → logfile.1.log
    handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
         
    return logger
