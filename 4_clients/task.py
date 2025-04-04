import logging
import os
from logging import Logger
from logging.handlers import RotatingFileHandler

import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(columns=["study_id"])
    
    # Drop fully empty columns (all NaNs)
    data = data.dropna(axis=1, how="all")
    
    # Map boolean column as number
    mode_value = data["case_csPCa"].dropna().mode().iloc[0]
    data['case_csPCa'] = data['case_csPCa'].map({'YES': 1, 'NO': 0}).fillna(mode_value).astype(int)

    # Format numbers
    for col in data.columns:
        data[col] = data[col].astype(str).str.replace(',', '.')
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Replace NaN values with column medians
    numeric_cols = data.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    data_shuffled = shuffle(data)
    print(data_shuffled)
    
    return data_shuffled


def old_plot_function(x, y, x_label, y_label, title):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
  
def plot_data(data):
    folder_name = "plots"
    os.makedirs(folder_name, exist_ok=True)
    boxplot_path = os.path.join(folder_name, "data_boxplot.png")
    corr_matrix_path = os.path.join(folder_name, "data_corr_matrix.png")
    
    # Boxplots
    fig, axes = plt.subplots(3, 1)
    fig.suptitle("Box Plots")
    sns.boxplot(data=data, x=data["patient_age"], ax=axes[0])
    sns.boxplot(data=data, x=data["psa"], ax=axes[1])
    sns.boxplot(data=data, x=data["prostate_volume"], ax=axes[2])
    plt.tight_layout()
    #plt.show()
    plt.savefig(boxplot_path)
    plt.close()
    
    # Correlation matrix
    plt.title("Correlation matrix")
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.tight_layout()
    #plt.show()
    plt.savefig(corr_matrix_path)
    plt.close()


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

    formatter = logging.Formatter('%(asctime)s - %(levelname)s:  %(message)s')
    handler = RotatingFileHandler(filename=file_path, maxBytes=max_bytes, backupCount=backup_count)
    handler.namer = lambda filename: filename.replace(".log.", ".") + ".log"  # Modify naming convention: logfile.log.1 → logfile.1.log
    handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
         
    return logger
