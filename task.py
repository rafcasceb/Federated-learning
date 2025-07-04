import logging
import os
from dataclasses import dataclass, field
from logging import Logger
from logging.handlers import RotatingFileHandler
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from flwr.common import Metrics
from sklearn.utils import shuffle



# -------------------------
# Preprocessing data
# -------------------------

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




# -------------------------
# Context auxiliary classes
# -------------------------

@dataclass
class HyperParameters:
    batch_size: int
    dropout: float
    input_size: int
    hidden_sizes: List[int]
    output_size: int
    
    num_rounds: int
    do_cross_val: bool
    num_cross_val_folds: int
    num_epochs: int
    test_size: float
    learning_rate: float
    binarization_threshold: float
    
    fraction_fit: float
    fraction_evaluate: float
    min_fit_clients: int
    min_evaluate_clients: int
    min_available_clients: int
    proximal_mu: float
    
    server_ip: str
    server_port: str
    @property
    def server_address(self) -> str:
        return f"{self.server_ip}:{self.server_port}"

@dataclass
class MetricsTracker:
    train_accuracies: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    test_accuracies: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)


@dataclass
class RandomState:
    is_test_mode: bool = False
    random_seed: int = 42
    shuffle_loaders: bool = field(init=False)
    
    def __post_init__(self):
        self.shuffle_loaders = not self.is_test_mode


@dataclass
class ClientContext:
    client_id: int
    logger: logging.Logger
    hyperparams: HyperParameters
    random_state: RandomState
    metrics_tracker: MetricsTracker = field(default_factory=MetricsTracker)


@dataclass
class ServerContext:
    logger: logging.Logger
    hyperparams: HyperParameters
    round_metrics: List[Tuple[int, Metrics]] = field(default_factory=list)


def load_client_context(client_id, logger, config_file_name: str, is_test_mode: bool) -> ClientContext:
    context = ClientContext(
        client_id = client_id,
        logger = logger,
        hyperparams = load_hyperparameters(config_file_name),
        random_state = RandomState(is_test_mode)
    )
    return context
    
    
def load_server_context(logger, config_file_name: str) -> ClientContext:
    context = ServerContext(
        logger = logger,
        hyperparams = load_hyperparameters(config_file_name)
    )
    return context
    

def load_hyperparameters(file_name: str) -> HyperParameters:
    with open(file_name, "r") as file:
        data = yaml.safe_load(file)
        return HyperParameters(**data)




# -------------------------
# Paths
# -------------------------

def create_os_paths(folder_name: str, *file_names: str) -> List:
    os.makedirs(folder_name, exist_ok=True)
    paths = [os.path.join(folder_name, fname) for fname in file_names]
    return paths




# -------------------------
# Logger
# -------------------------

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
    handler.namer = lambda filename: filename.replace(".log.", ".") + ".log"  # Rename: logfile.log.1 -> logfile.1.log
    handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
         
    return logger




# -------------------------
# Plotting
# -------------------------

def plot_loaded_data(data, client_id):
    folder_name = "plots"
    os.makedirs(folder_name, exist_ok=True)
    boxplot_path = os.path.join(folder_name, f"data_boxplot_{client_id}.png")
    corr_matrix_path = os.path.join(folder_name, f"data_corr_matrix_{client_id}.png")
    
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



def _create_paths_acc_and_loss_plots(client_id: int) -> Tuple[str, str]:
    FOLDER_NAME = "plots"
    os.makedirs(FOLDER_NAME, exist_ok=True)
    accuracies_path = os.path.join(FOLDER_NAME, f"train_test_acc_{client_id}.png")
    loss_path = os.path.join(FOLDER_NAME, f"train_test_loss_{client_id}.png")
    return accuracies_path, loss_path


def _plot_training_vs_testing_metric(train_x, train_y, test_x, test_y,
                                       metric_name:str, range_end_round_epochs, save_file_path: str):
    
    train_label = f"Training {metric_name.lower()} (by epochs)"
    test_label = f"Testing {metric_name.lower()} (by rounds)"
    title = f"TRAINING VS TESTING {metric_name.upper()}"
    x_label = "Epochs"
    y_label = f"Training vs Testing {metric_name.lower()}"
    
    plt.plot(train_x, train_y, label=train_label, color="#E06989") #E9839E
    plt.plot(test_x, test_y, label=test_label, marker="o", color="#309F69") #9F3058
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    for epoch in range_end_round_epochs:
        plt.axvline(x=epoch, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_file_path)
    plt.close()



def plot_accuracy_and_loss(train_acc: List[int], train_loss: List[int], test_acc: List[int], test_loss: List[int],
                           client_id: int, hyperparams: HyperParameters):
    
    accuracies_path, loss_path = _create_paths_acc_and_loss_plots(client_id)
    
    num_rounds = hyperparams.num_rounds
    num_epochs_by_round = hyperparams.num_epochs
    num_total_epochs = num_rounds * num_epochs_by_round

    range_num_epochs = range(1, num_total_epochs +1)
    range_end_round_epochs = range(num_epochs_by_round, num_total_epochs +1, num_epochs_by_round) 

    _plot_training_vs_testing_metric(
        range_num_epochs, train_acc, 
        range_end_round_epochs, test_acc,
        "accuracy", range_end_round_epochs, accuracies_path
    )
    _plot_training_vs_testing_metric(
        range_num_epochs, train_loss,
        range_end_round_epochs, test_loss,
        "loss", range_end_round_epochs, loss_path
    )


def plot_accuracy_and_loss_centralized(train_acc: List[int], train_loss: List[int], test_acc: List[int], test_loss: List[int],
                           client_id: int, hyperparams: HyperParameters):
    
    accuracies_path, loss_path = _create_paths_acc_and_loss_plots(client_id)
    
    num_epochs = hyperparams.num_epochs
    range_num_epochs = range(1, num_epochs +1)
    range_end_epoch = range(num_epochs, num_epochs +1) 

    _plot_training_vs_testing_metric(
        range_num_epochs, train_acc, 
        range_end_epoch, test_acc,
        "accuracy", range_end_epoch, accuracies_path
    )
    _plot_training_vs_testing_metric(
        range_num_epochs, train_loss,
        range_end_epoch, test_loss,
        "loss", range_end_epoch, loss_path
    )



def plot_accuracy_and_loss_cv(train_acc: List[int], train_loss: List[int], test_acc: List[int], test_loss: List[int],
                           client_id: int, hyperparams: HyperParameters):
    
    accuracies_path, loss_path = _create_paths_acc_and_loss_plots(client_id)
    
    num_rounds = hyperparams.num_rounds
    num_cross_val_folds_round = hyperparams.num_cross_val_folds
    num_epochs_by_fold = hyperparams.num_epochs

    num_epochs_by_round = num_cross_val_folds_round * num_epochs_by_fold
    num_epochs = num_rounds * num_epochs_by_round

    range_num_rounds = range(1, num_rounds +1)
    range_num_epochs = range(1, num_epochs +1)
    range_test_epochs = range(num_epochs_by_fold, num_epochs +1, num_epochs_by_fold)  # test_epochs = [fold * num_epochs_by_fold for fold in range_num_cross_val_folds]  
    range_end_round_epochs = range(num_epochs_by_round, num_epochs +1, num_epochs_by_round)

    # At the end of each round, an additional test is done; delete previous one and keep that one (to have only one per epoch)      
    for r in range_num_rounds:
        test_acc.pop(r*num_cross_val_folds_round -1)
        test_loss.pop(r*num_cross_val_folds_round -1)
        
    _plot_training_vs_testing_metric(
        range_num_epochs, train_acc, range_test_epochs, test_acc,
        "accuracy", range_end_round_epochs, accuracies_path
    )
    _plot_training_vs_testing_metric(
        range_num_epochs, train_loss, range_test_epochs, test_loss,
        "loss", range_end_round_epochs, loss_path
    )
