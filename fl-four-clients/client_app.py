import os
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flwr.client import NumPyClient, start_client
from sklearn.model_selection import train_test_split
from model import NeuralNetwork
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from task import (ClientContext, create_logger, load_client_context,
                  plot_accuracy_and_loss, plot_loaded_data, preprocess_data)
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy




CONFIGURATION_FILE = "config.yaml"
SERVER_IP = "192.168.18.12"
SERVER_PORT = "8081"



# -------------------------
# 1. Data Preparation
# -------------------------

def _read_data(excel_file_name: str, temp_csv_file_name:str, context: ClientContext):
    FOLDE_NAME = "data"
    excel_path = os.path.join(FOLDE_NAME, excel_file_name)
    temp_csv_path = os.path.join(FOLDE_NAME, temp_csv_file_name)
    os.makedirs(FOLDE_NAME, exist_ok=True)
    context.logger.info("Loading data from %s", excel_path)
    
    # Read Excel file and convert it into CSV for confort
    data_excel = pd.read_excel(excel_path)
    data_excel.to_csv(temp_csv_path, sep=";", index=False)
    data = pd.read_csv(temp_csv_path, sep=";")    
    context.logger.info("Data loaded. Shape: %s", data.shape)
       
    return data


def load_data(excel_name: str, temp_csv_name:str, context: ClientContext) -> Tuple[torch.Tensor, torch.Tensor]:
    data = _read_data(excel_name, temp_csv_name, context)
    data = preprocess_data(data)
    context.logger.info("Data preprocessing completed. Final shape: %s", data.shape)
       
    plot_loaded_data(data, context.client_id)

    # Separata data into inputs (x) and outputs (y)
    x = data.iloc[:, :-1].values  # Inputs characteristics (features);   all columns but last one
    y = data.iloc[:, -1].values   # Output characteristics (labels);     last column
    
    # Standardize input characteristics
    x = StandardScaler().fit_transform(x)
    
    # Convert data into PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return x_tensor, y_tensor




# -------------------------
# 2. Training and Evaluation
# ------------------------- 

def train(model: nn.Module, train_data: DataLoader, context: ClientContext) -> None:
    hp = context.hyperparams

    criterion = nn.BCEWithLogitsLoss()  # Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)
    accuracy_metric = BinaryAccuracy()
    loss_metric = MeanMetric()

    context.logger.info("Hyperparameters - LR: %f, Batch Size: %d, Epochs: %d", hp.learning_rate, hp.batch_size, hp.num_epochs)
    context.logger.info("Starting training for %d epochs...", hp.num_epochs)

    for epoch in range(hp.num_epochs):
        model.train()  # Set to training mode
        accuracy_metric.reset()
        loss_metric.reset()

        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            accuracy_metric.update(torch.sigmoid(outputs).squeeze(), labels)
            loss_metric.update(loss.item())

        epoch_accuracy = accuracy_metric.compute().item()
        epoch_loss = loss_metric.compute().item()
        context.logger.info("Epoch %d/%d -- Loss: %.4f, Accuracy: %.4f", epoch+1, hp.num_epochs, epoch_loss, epoch_accuracy)
    
        context.metrics_tracker.train_accuracies.append(epoch_accuracy)
        context.metrics_tracker.train_losses.append(epoch_loss)


def _calculate_average_test_metrics(all_labels: List[int], all_predictions: List[int], context: ClientContext) -> Dict[str,float]:
    accuracy = accuracy_score(all_labels, all_predictions)                        # (TP+FN) / (TP+TN+FP+FN)  =  accurate_predictions / all_predictions
    precision = precision_score(all_labels, all_predictions, zero_division=0)     # TP / (TP+FP)  =  TP / predicted_positives
    recall = recall_score(all_labels, all_predictions, zero_division=0)           # TP / (TP+FN)  =  TP / real_positives
    f1 = f1_score(all_labels, all_predictions, zero_division=0)                   # 2 * (precision * recall) / (precision + recall)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)           # accuracy for imbalanced DS (the lower than accuracy, the more imbalanced)
    mcc = matthews_corrcoef(all_labels, all_predictions)                          # randomness of predictions for imbalanced DS (-1=wrong, 0=random, 1=perfect)
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1,
        "Balanced accuracy": balanced_acc,
        "MCC": mcc
    }
    context.logger.info("Testing metrics -- Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1 score: %.3f, Balanced accuracy: %.3f, MCC: %.3f",
                        accuracy, precision, recall, f1, balanced_acc, mcc)
    return metrics 


def test(model: nn.Module, test_data: DataLoader, context: ClientContext) -> Tuple[float, Dict[str,float]]:
    model.eval()  # Set to evaluation mode
    loss_metric = MeanMetric()
    all_labels = []
    all_predictions = []
    hp = context.hyperparams
    
    context.logger.info("Using threshold %.2f for binarization.", hp.binarization_threshold)
    
    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in test_data:
            # Realize prediction
            outputs = torch.sigmoid(model(inputs)).squeeze()  # Sigmoid activation to transform logits into usable predictions
            #! print("Raw outputs:", outputs)
            predictions = (outputs > hp.binarization_threshold).float()  # Binarize predictions
            #! print("Binary outputs (predictions):", predictions)
            predictions = torch.nan_to_num(predictions, nan=0)

            # Collect labels and predictions
            loss_tensor = F.binary_cross_entropy(outputs, labels)
            loss_metric.update(loss_tensor.item())
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.detach().numpy())
    
    loss = loss_metric.compute().item()
    context.logger.info("Testing loss: %.4f", loss)
    metrics = _calculate_average_test_metrics(all_labels, all_predictions, context)
    
    context.metrics_tracker.test_accuracies.append(metrics.get("Accuracy"))
    context.metrics_tracker.test_losses.append(loss)
    
    return loss, metrics 




# -------------------------
# 3. Federated Learning Client
# -------------------------

class FlowerClient(NumPyClient):

    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, context: ClientContext) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.context = context
        
        self.context.logger.info("Client initialized.")
    
    #! TODO: Probar a quitar los config de estos tres métodos
    def get_parameters(self, config: Dict[str,Any]) -> List[np.ndarray]:
        self.context.logger.info("Fetching model parameters...")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.context.logger.info("Updating model parameters...")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters: List[np.ndarray], config: Dict[str,Any]) -> Tuple[List[np.ndarray], int, Dict]:
        logger = self.context.logger
        logger.info("")
        logger.info("=== [NEW TRAINING ROUND] ===")
        logger.info("Starting local training...")
        self.set_parameters(parameters)
        self.context.logger.info("Starting local model training...")
        train(self.model, self.train_loader, self.context)
        logger.info("Local training complete.")
        num_examples = len(self.train_loader.dataset)
        return self.get_parameters(config={}), num_examples, {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str,Any]) -> Tuple[float, int, Dict[str,float]]:
        self.context.logger.info("=== [VALIDATION REPORT] ===")
        self.context.logger.info("Starting local model validation...")
        self.set_parameters(parameters)
        loss, metrics = test(self.model, self.test_loader, self.context)
        num_examples = len(self.test_loader.dataset)
        return loss, num_examples, metrics



def client_fn(excel_file_name: str, temp_csv_file_name:str, context: ClientContext) -> FlowerClient:
    hp = context.hyperparams
    
    x, y = load_data(excel_file_name, temp_csv_file_name, context)
    if x.shape[1] != hp.input_size:
        context.logger.warning(f"Input size mismatch: data has {x.shape[1]}, but config has {hp.input_size}.")

    if context.hyperparams.test_size is None:
        test_size = 0.2
    else:
        test_size = context.hyperparams.test_size

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size = test_size,
        random_state = context.random_state.random_seed
    )
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=context.hyperparams.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=context.hyperparams.batch_size, shuffle=False)
    
    model = NeuralNetwork(
        input_size = hp.input_size,
        hidden_sizes = hp.hidden_sizes,
        output_size = hp.output_size,
        dropout = hp.dropout
    )
    
    return FlowerClient(model, train_loader, test_loader, context).to_client()




# -------------------------
# 4. Main Execution (legacy mode)
# -------------------------

def configure_environment(context: ClientContext):
    rs = context.random_state
    if rs.is_test_mode:
        np.random.seed(rs.random_seed)
        torch.manual_seed(rs.random_seed)
        context.logger.info(f"Running in Test mode (deterministic). Seeds set to {rs.random_seed}.")
    else:
        context.logger.info("Running in Production mode (non-deterministic).")


def start_flower_client(client_id: int, is_test_mode: bool=False):
    excel_file_name = f"PI-CAI_3__part{client_id}.xlsx" 
    temp_csv_file_name = f"temp_database_{client_id}.csv"
    
    logger_name = f"client_{client_id}.log"
    logger = create_logger(logger_name)
    logger.info("Starting FL client...")
    
    try:
        context = load_client_context(client_id, logger, CONFIGURATION_FILE, is_test_mode)
    except Exception as e:
        logger.error(f"Failed to load context: {str(e)}")
        logger.info("Closing FL client...")
        return
    
    configure_environment(context)

    server_address = f"{SERVER_IP}:{SERVER_PORT}"
    start_client(
        server_address=server_address,
        client=client_fn(excel_file_name, temp_csv_file_name, context),
    )
    
    mt = context.metrics_tracker
    plot_accuracy_and_loss(
        mt.train_accuracies, mt.train_losses, mt.test_accuracies, mt.test_losses,
        context.client_id, context.hyperparams
    )
    logger.info("Closing FL client...")
