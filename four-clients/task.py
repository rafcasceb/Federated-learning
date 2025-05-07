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


def plot_loaded_data(data, client_number):
    folder_name = "plots"
    os.makedirs(folder_name, exist_ok=True)
    boxplot_path = os.path.join(folder_name, f"data_boxplot_{client_number}.png")
    corr_matrix_path = os.path.join(folder_name, f"data_corr_matrix_{client_number}.png")
    
    # Boxplots
    fig, axes = plt.subplots(3, 1)
    fig.suptitle("BOX PLOTS")
    sns.boxplot(data=data, x=data["patient_age"], ax=axes[0])
    sns.boxplot(data=data, x=data["psa"], ax=axes[1])
    sns.boxplot(data=data, x=data["prostate_volume"], ax=axes[2])
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()
    
    # Correlation matrix
    plt.title("CORRELATION MATRIX")
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(corr_matrix_path)
    plt.close()


def plot_accuracy_and_loss(train_acc, train_loss, test_acc, test_loss, client_number, num_epochs_by_fold, num_rounds, num_cross_val_folds_round):
    folder_name = "plots"
    os.makedirs(folder_name, exist_ok=True)
    accuracies_path = os.path.join(folder_name, f"training_testing_acc_{client_number}.png")
    loss_path = os.path.join(folder_name, f"training_testing_loss_{client_number}.png")
    
    num_epochs = num_rounds * num_cross_val_folds_round * num_epochs_by_fold
    num_epochs_by_round = num_cross_val_folds_round * num_epochs_by_fold
    
    range_num_rounds = range(1, num_rounds +1)
    range_num_epochs = range(1, num_epochs +1)
    range_test_epochs = range(num_epochs_by_fold, num_epochs +1, num_epochs_by_fold)  # test_epochs = [fold * num_epochs_by_fold for fold in range_num_cross_val_folds]  
    range_end_round_epochs = range(num_epochs_by_round, num_epochs +1, num_epochs_by_round)
    
    # Keep only final test validation per round
    for i in range_num_rounds:
        test_acc.pop(i*num_cross_val_folds_round)
        test_loss.pop(i*num_cross_val_folds_round) 
    
    plt.plot(range_num_epochs, train_acc, label="Training accuracy (by epochs)")
    plt.plot(range_test_epochs, test_acc, label="Testing accuracy (by rounds)", marker="o")
    plt.title("TRAINING VS TESTING ACCURACY")
    plt.xlabel("Epochs")
    plt.ylabel("Training vs Testing accuracy")
    plt.legend()
    for epoch in range_end_round_epochs:
        plt.axvline(x=epoch, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(accuracies_path)
    plt.close()
    
    plt.plot(range_num_epochs, train_loss, label="Training loss (by epochs)")
    plt.plot(range_test_epochs, test_loss, label="Testing loss (by rounds)", marker="o")
    plt.title("TRAINING VS TESTING LOSS")
    plt.xlabel("Epochs")
    plt.ylabel("Training vs Testing loss")
    plt.legend()
    for epoch in range_end_round_epochs:
        plt.axvline(x=epoch, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path)
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

