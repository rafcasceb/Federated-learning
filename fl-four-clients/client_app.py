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
from model import NeuralNetwork
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from task import (TrainingContext, create_logger, load_context,
                  plot_accuracy_and_loss, plot_loaded_data, preprocess_data)
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAccuracy




CONFIGURATION_FILE = "config.yaml"




# -------------------------
# 1. Data Preparation
# -------------------------

def _read_data(excel_file_name: str, temp_csv_file_name:str, context: TrainingContext):
    folder_name = "data"
    excel_path = os.path.join(folder_name, excel_file_name)
    temp_csv_path = os.path.join(folder_name, temp_csv_file_name)
    os.makedirs(folder_name, exist_ok=True)
    context.logger.info("Loading data from %s", excel_path)
    
    # Read Excel file and convert it into CSV for confort
    data_excel = pd.read_excel(excel_path)
    data_excel.to_csv(temp_csv_path, sep=";", index=False)
    data = pd.read_csv(temp_csv_path, sep=";")    
    context.logger.info("Data loaded. Shape: %s", data.shape)
       
    return data


def load_data(excel_file_name: str, temp_csv_file_name:str, context: TrainingContext) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = _read_data(excel_file_name, temp_csv_file_name, context)
    data = preprocess_data(data)
    context.logger.info("Data preprocessing completed. Final shape: %s", data.shape)
       
    plot_loaded_data(data, context.client_id)

    # Separata data into inputs (x) and outputs (y)
    x = data.iloc[:, :-1].values  # Inputs characteristics (features);   all columns but last one
    y = data.iloc[:, -1].values   # Output characteristics (labels);     last column
    
    # Standardize input characteristics
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    # Convert data into PyTorch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    return x, y




# -------------------------
# 2. Training and Evaluation
# -------------------------

def train_cross_validation(model: nn.Module, x: torch.Tensor, y: torch.Tensor, context: TrainingContext):
    hp = context.hyperparams
    rs = context.random_state
    
    kfold = KFold(n_splits=hp.num_cross_val_folds_round, shuffle=True, random_state=rs.random_seed)
    folds_train_losses = []
    folds_train_accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(x)):
        context.logger.info(f"Training fold {fold_idx+1}/{hp.num_cross_val_folds_round}...")
        
        x_train_fold, y_train_fold = x[train_idx], y[train_idx]
        x_test_fold, y_test_fold = x[test_idx], y[test_idx]
        train_dataset = TensorDataset(x_train_fold, y_train_fold)
        test_dataset = TensorDataset(x_test_fold, y_test_fold)
        trainloader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=rs.shuffle_loaders)
        testloader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=rs.shuffle_loaders)

        train(model, trainloader, context)
        
        test_loss, test_metrics = test(model, testloader, context)
        folds_train_losses.append(test_loss)
        folds_train_accuracies.append(test_metrics.get("Accuracy"))
    
    avg_loss_all_folders = np.mean(folds_train_losses)
    avg_acc_all_folders = np.mean(folds_train_accuracies)
    context.logger.info(f"Cross-Validation -- Avg Loss: {avg_loss_all_folders:.4f}, Avg Accuracy: {avg_acc_all_folders:.4f}")
    

def train(model: nn.Module, train_data: DataLoader, context: TrainingContext) -> None:
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


def _calculate_average_test_metrics(all_labels: List[int], all_predictions: List[int], context: TrainingContext) -> Dict[str,float]:
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
    
    context.logger.info("Testing metrics -- Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 score: %.2f, Balanced accuracy: %.2f, MCC: %.2f",
                accuracy, precision, recall, f1, balanced_acc, mcc)
    
    return metrics 


def test(model: nn.Module, test_data: DataLoader, context: TrainingContext) -> Tuple[float, Dict[str,float]]:
    model.eval()  # Set to evaluation mode
    loss_metric = MeanMetric()
    all_labels = []
    all_predictions = []
    hp = context.hyperparams
    
    context.logger.info("Using threshold %.2f for binarization.", hp.binarization_threshold)
    
    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in test_data:
            #! TODO: Assert no NaN in inputs and labels?
            
            # Realize prediction
            outputs = torch.sigmoid(model(inputs)).squeeze()  # Apply sigmoid activation to transform logits in usable predictions
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
    context.logger.info("Loss: %.4f", loss)
    
    metrics = _calculate_average_test_metrics(all_labels, all_predictions, context)
    
    context.metrics_tracker.test_accuracies.append(metrics.get("Accuracy"))
    context.metrics_tracker.test_losses.append(loss)
    
    return loss, metrics 




# -------------------------
# 3. Federated Learning Client
# -------------------------

class FlowerClient(NumPyClient):
    
    def __init__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, context: TrainingContext) -> None:
        self.model = model
        self.x = x
        self.y = y
        self.context = context
        self.context.logger.info("Client initialized.")
    
    def get_parameters(self, config: Dict[str,Any]) -> List[np.ndarray]:
        self.context.logger.info("Fetching model parameters...")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.context.logger.info("Updating model parameters...")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters: List[np.ndarray], config: Dict[str,Any]) -> Tuple[List[np.ndarray], int, Dict]:
        self.context.logger.info("");
        self.context.logger.info("=== [NEW TRAINING ROUND] ===")
        self.context.logger.info("Starting local training...")
        self.set_parameters(parameters)
        train_cross_validation(self.model, self.x, self.y, self.context)
        self.context.logger.info("Local training complete.")
        return self.get_parameters(config={}), len(self.x), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str,Any]) -> Tuple[float, int, Dict[str,float]]:
        self.context.logger.info("=== [EVALUATION REPORT] ===")
        self.context.logger.info("Starting local model evaluation...")
        self.set_parameters(parameters)
        whole_dataset = TensorDataset(self.x, self.y)
        hp = self.context.hyperparams
        rs = self.context.random_state
        whole_dataloader = DataLoader(whole_dataset, batch_size=hp.batch_size, shuffle=rs.shuffle_loaders)
        loss, metrics = test(self.model, whole_dataloader, self.context)
        num_examples = len(self.x)
        return loss, num_examples, metrics



def client_fn(excel_file_name: str, temp_csv_file_name:str, context: TrainingContext) -> FlowerClient:
    hp = context.hyperparams
    
    x, y = load_data(excel_file_name, temp_csv_file_name, context)

    if x.shape[1] != hp.input_size:
        context.logger.warning(f"Input size mismatch: data has {x.shape[1]}, but config has {hp.input_size}.")

    model = NeuralNetwork(
        input_size = hp.input_size,
        hidden_sizes = hp.hidden_sizes,
        output_size = hp.output_size,
        dropout = hp.dropout
    )
    
    return FlowerClient(model, x, y, context).to_client()




# -------------------------
# 4. Main Execution (legacy mode)
# -------------------------

def _configure_environment(context: TrainingContext):
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
        context = load_context(client_id, logger, CONFIGURATION_FILE, is_test_mode)
    except Exception as e:
        logger.error(f"Failed to load context: {str(e)}")
        logger.info("Closing FL client...")
        return
    
    _configure_environment(context)
    
    server_ip = "192.168.18.12"
    server_port = "8081"
    server_address = f"{server_ip}:{server_port}"
    
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
