import os
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from flwr.client import NumPyClient, start_client
from flwr.common import Context
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from task import create_logger, plot_accuracy_and_loss, plot_loaded_data, preprocess_data
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, BinaryAccuracy


# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
HIDDEN_SIZES = [128, 128]
BINARIZATION_THRESHOLD = 0.4
NUM_EPOCHS = 10 #50
TEST_SIZE = 0.2
DROPOUT = 0.1

# Random seeds set for testing reproducibility
np.random.seed(55)
torch.manual_seed(55)
RANDOM_STATE = 42
SHUFFLE_LOADERS = False

# Others
logger = None
CLIENT_NUMBER = None
general_epoch_train_loss = []
general_epoch_train_acc = []
general_round_test_loss = []
general_round_test_acc = []




# -------------------------
# 1. Data Preparation
# -------------------------

def __read_data(excel_file_name: str, temp_csv_file_name:str):
    folder_name = "data"
    excel_path = os.path.join(folder_name, excel_file_name)
    temp_csv_path = os.path.join(folder_name, temp_csv_file_name)
    os.makedirs(folder_name, exist_ok=True)
    logger.info("Loading data from %s", excel_path)
    
    # Read Excel file and convert it into CSV for confort
    data_excel = pd.read_excel(excel_path)
    data_excel.to_csv(temp_csv_path, sep=";", index=False)
    data = pd.read_csv(temp_csv_path, sep=";")    
    logger.info("Data loaded. Shape: %s", data.shape)
       
    return data


def load_data(excel_file_name: str, temp_csv_file_name:str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = __read_data(excel_file_name, temp_csv_file_name)
    data = preprocess_data(data)
    logger.info("Data preprocessing completed. Final shape: %s", data.shape)
       
    plot_loaded_data(data, CLIENT_NUMBER)

    # Separata data into inputs (x) and outputs (y)
    x = data.iloc[:, :-1].values  # Inputs characteristics (features);   all columns but last one
    y = data.iloc[:, -1].values   # Output characteristics (labels);     last column
    
    # Divide data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Standardize input characteristics
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Convert data into PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return x_train, y_train, x_test, y_test




# -------------------------
# 2. Neural Network Model Definition
# -------------------------

class NeuralNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int) -> None:
        super(NeuralNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))  
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(DROPOUT))  

        for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            
            init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
            #init.xavier_uniform_(layers[-1].weight)
            layers.append(nn.BatchNorm1d(out_size))
            #layers.append(nn.LeakyReLU())
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        init.xavier_uniform_(layers[-1].weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



# -------------------------
# 3. Training and Evaluation
# -------------------------

def train(model: nn.Module, train_data: DataLoader, epochs: int =48) -> None:
    criterion = nn.BCEWithLogitsLoss()  # Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    accuracy_metric = BinaryAccuracy()
    loss_metric = MeanMetric()
    
    logger.info("Hyperparameters - LR: %f, Batch Size: %d, Epochs: %d", LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)
    logger.info("Starting training for %d epochs...", epochs)
    
    for epoch in range(epochs):
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
        logger.info("Epoch %d/%d -- Loss: %.4f, Accuracy: %.4f", epoch+1, epochs, epoch_loss, epoch_accuracy)
    
        general_epoch_train_acc.append(epoch_accuracy)
        general_epoch_train_loss.append(epoch_loss)


def __calculate_average_test_metrics(all_labels: List[int], all_predictions: List[int]) -> Dict[str,float]:
    accuracy = accuracy_score(all_labels, all_predictions)                        # (TP+FN) / (TP+TN+FP+FN)  =  accurate_predictions / all_predictions
    precision = precision_score(all_labels, all_predictions, zero_division=0)     # TP / (TP+FP)  =  TP / predicted_positives
    recall = recall_score(all_labels, all_predictions, zero_division=0)           # TP / (TP+FN)  =  TP / real_positives
    f1 = f1_score(all_labels, all_predictions, zero_division=0)                   # 2 * (precision * recall) / (precision + recall)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)           # accuracy for imbalanced DS (the lower than accuracy, the more imbalanced)
    mcc = matthews_corrcoef(all_labels, all_predictions)                          # randomness of predictions for imbalanced DS (-1=wrong, 0=random, 1=perfect)
    
    metrics = {"Accuracy": accuracy,
               "Precision": precision,
               "Recall": recall,
               "F1 score": f1,
               "Balanced accuracy": balanced_acc,
               "MCC": mcc}
    
    logger.info("Testing metrics -- Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 score: %.2f, Balanced accuracy: %.2f, MCC: %.2f",
                accuracy, precision, recall, f1, balanced_acc, mcc)
    
    return metrics 


def test(model: nn.Module, test_data: DataLoader) -> Tuple[float, Dict[str,float]]:
    model.eval()  # Set to evaluation mode
    loss_metric = MeanMetric()
    all_labels = []
    all_predictions = []
    
    logger.info("Using threshold %.2f for binarization", BINARIZATION_THRESHOLD)
    
    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in test_data:
            ##! TODO: Assert no NaN in inputs and labels?
            
            # Realize prediction
            outputs = torch.sigmoid(model(inputs)).squeeze()  # Apply sigmoid activation to transform logits in usable predictions
            print("Raw outputs:", outputs)
            predictions = (outputs > BINARIZATION_THRESHOLD).float()  # Binarize predictions
            print("Binary outputs (predictions):", predictions)
            predictions = torch.nan_to_num(predictions, nan=0)
            
            # Collect labels and predictions
            loss_tensor = F.binary_cross_entropy(outputs, labels)
            loss_metric.update(loss_tensor.item())
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.detach().numpy())
    
    loss = loss_metric.compute().item()
    logger.info("Loss: %.4f", loss)
    
    metrics = __calculate_average_test_metrics(all_labels, all_predictions)
    
    general_round_test_acc.append(metrics.get("Accuracy"))
    general_round_test_loss.append(loss)
    
    return loss, metrics 




# -------------------------
# 4. Federated Learning Client
# -------------------------

class FlowerClient(NumPyClient):
    
    def __init__(self, net: nn.Module, trainloader: DataLoader, testloader: DataLoader) -> None:
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        logger.info("Client initialized.")
    
    def get_parameters(self, config: Dict[str,Any]) -> list[np.ndarray]:
        logger.info("Fetching model parameters...")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        logger.info("Updating model parameters...")
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict)

    def fit(self, parameters: list[np.ndarray], config: Dict[str,Any]) -> Tuple[list[np.ndarray], int, Dict]:
        logger.info(""); logger.info("=== [NEW TRAINING ROUND] ===")
        logger.info("Starting local training...")
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=NUM_EPOCHS)
        logger.info("Local training complete")
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: list[np.ndarray], config: Dict[str,Any]) -> Tuple[float, int, Dict[str,float]]:
        logger.info("=== [EVALUATION REPORT] ===")
        logger.info("Starting local model evaluation...")
        self.set_parameters(parameters)
        loss, metrics = test(self.net, self.testloader)
        num_examples = len(self.testloader.dataset)
        return loss, num_examples, metrics



def client_fn(excel_file_name: str, temp_csv_file_name:str, context: Context) -> FlowerClient:
    """
    It creates an instance of FlowerClient with the configuration given. 
    No need to pass a context for the moment.
    """

    # Supposing x_train, y_train, x_test, y_test are tensors
    x_train, y_train, x_test, y_test = load_data(excel_file_name, temp_csv_file_name)

    # Create a TensorDataset
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    # Create a DataLodaer
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_LOADERS)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_LOADERS)

    # Create the neural network model
    input_size = x_train.shape[1]
    output_size = 1
    net = NeuralNetwork(input_size, HIDDEN_SIZES, output_size)
    
    return FlowerClient(net, trainloader, testloader).to_client()




# -------------------------
# 5. Main Execution (legacy mode)
# -------------------------

def start_flower_client(client_number: int, excel_file_name: str, temp_csv_file_name: str, logger_name: str, context: Context):
    global logger
    logger = create_logger(logger_name)
    
    global CLIENT_NUMBER
    CLIENT_NUMBER = client_number
    
    server_ip = "192.168.18.12"
    server_port = "8081"
    server_address = f"{server_ip}:{server_port}"  
    
    logger.info("Starting FL client...")
    start_client(
        server_address=server_address,
        client=client_fn(excel_file_name, temp_csv_file_name, context),
    )
    logger.info("Closing FL client...")
    
    plot_accuracy_and_loss(general_epoch_train_acc, general_epoch_train_loss,
                           general_round_test_acc, general_round_test_loss, CLIENT_NUMBER, NUM_EPOCHS)
