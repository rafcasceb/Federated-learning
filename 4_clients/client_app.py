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
from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import Context
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from task import create_logger, plot_function, preprocess_data, plot_seaborn_function, plot_data
from torch.utils.data import DataLoader, TensorDataset



BATCH_SIZE = 16
LEARNING_RATE = 0.0001
HIDDEN_SIZES = [128, 128]
BINARIZATION_THRESHOLD = 0.4
NUM_EPOCHS = 10 #50
TEST_SIZE = 0.2
DROPOUT = 0.1
logger = None


# Set the random seed for testing reproducibility
np.random.seed(55)
torch.manual_seed(55)
RANDOM_STATE = 42




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
    
    #plot_seaborn_function(data, "patient_age", "psa", "Whatever")
    #plot_seaborn_function(data, "patient_age", "prostate_volume", "Whatever")
    #plot_seaborn_function(data, "prostate_volume", "psa", "Whatever")
    # plot_seaborn_function(data, "patient_age", "psa", "Whatever")
    # plot_seaborn_function(data, "patient_age", "case_csPCa", "Whatever")
    # plot_function(data["patient_age"], data["case_csPCa"], "IDK", "Patient age", "Case csPCa", "Yeah?")
    
    plot_data(data)

    # Separata data into inputs (X) and outputs (y)
    X = data.iloc[:, :-1].values  # Inputs characteristics (features);   all columns but last one
    y = data.iloc[:, -1].values   # Output characteristics (labels);     last column
    
    # Divide data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Standardize input characteristics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert data into PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test




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
    
    logger.info("Hyperparameters - LR: %f, Batch Size: %d, Epochs: %d", LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS)
    logger.info("Starting training for %d epochs...", epochs)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)  # Multiply single loss times batch size
            
        epoch_loss = total_loss / len(train_data.dataset)  # Calculate average loss per sample
        logger.info("Epoch %d/%d - Loss: %.4f", epoch+1, epochs, epoch_loss)


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
    
    logger.info("Metrics -- Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 score: %.2f, Balanced accuracy: %.2f, MCC: %.2f",
                accuracy, precision, recall, f1, balanced_acc, mcc)
    
    return metrics 


def test(model: nn.Module, test_data: DataLoader) -> Tuple[float, Dict[str,float]]:
    model.eval()  # Evaluation model
    total_loss = 0.0
    total_samples = 0
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
            loss = F.binary_cross_entropy(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.detach().numpy())
    
    loss = total_loss / total_samples
    logger.info("Loss: %.4f", loss)
    
    metrics = __calculate_average_test_metrics(all_labels, all_predictions)
    
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
        logger.info("Local training complete.")
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

    # Supposing X_train, y_train, X_test, y_test are tensors
    X_train, y_train, X_test, y_test = load_data(excel_file_name, temp_csv_file_name)

    # Create a TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create a DataLodaer
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create the neural network model
    input_size = X_train.shape[1]
    output_size = 1
    net = NeuralNetwork(input_size, HIDDEN_SIZES, output_size)
    
    return FlowerClient(net, trainloader, testloader).to_client()




# -------------------------
# 5. Main Execution (legacy mode)
# -------------------------

def start_flower_client(excel_file_name: str, temp_csv_file_name:str, logger_name:str, context: Context):
    global logger
    logger = create_logger(logger_name)
    
    #server_ip = input("SERVER IP: ")
    #server_port = input("SERVER PORT: ")
    server_ip = "192.168.18.12"
    server_port = "8081"
    server_address = f"{server_ip}:{server_port}"  
    
    logger.info("Starting FL client...")
    start_client(
        server_address=server_address,
        client=client_fn(excel_file_name, temp_csv_file_name, context),
    )
    logger.info("Closing FL client...")