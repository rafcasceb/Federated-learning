from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import Context
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset



# -------------------------
# 1. Data Preparation
# -------------------------

def preprocess_data(data):
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
    
    

def load_data():
    excel_file_name = "PI-CAI_3_parte3.xlsx" 
    temp_csv_file_name = "temp_database3.csv"
    
    # Leer el archivo Excel y convertirlo a CSV por comodidad
    data_excel = pd.read_excel(excel_file_name)
    data_excel.to_csv(temp_csv_file_name, sep=";", index=False)
    data = pd.read_csv(temp_csv_file_name, sep=";")    
    
    data = preprocess_data(data)

    # Separar los datos en entradas (X) y salida (y)
    X = data.iloc[:, :-1].values  # Características de entrada (inputs/features);   todas las columnas menos la última
    y = data.iloc[:, -1].values   # Característica de salida (outputs/labels);      última columna
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar las características de entrada
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convertir los datos a tensores de PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test



# -------------------------
# 2. Neural Network Model Definition
# -------------------------

class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)    # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Fully connected layer 2
        self.fc3 = nn.Linear(hidden_size2, output_size)   # Fully connected layer 3
        self.relu = nn.ReLU()                             # Activation function

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out



# -------------------------
# 3. Training and Evaluation
# -------------------------

def train(model, train_data, epochs=10):
    learning_rate = 0.001
    criterion = nn.BCEWithLogitsLoss()  # Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)  # Multiplicar la pérdida por el tamaño del lote
            
        epoch_loss = total_loss / len(train_data.dataset)  # Calcular la pérdida promedio por muestra



def test(model, test_data):
    model.eval()  # Modo de evaluación
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in test_data:
            # Reemplazar NaN en labels e inputs
            labels = torch.nan_to_num(labels, nan=-1)   # -1 o el que queramos
            inputs = torch.nan_to_num(inputs, nan=0)
            
            # Realizar la predicción
            outputs = model(inputs).squeeze()
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Multiplicar la pérdida por el tamaño del lote
            
            # Convertir las predicciones a binario (0 o 1) y preparar métricas
            outputs = torch.sigmoid(model(inputs)).squeeze()  # Apply sigmoid activation
            predictions = (outputs > 0.5).float()
            predictions = torch.nan_to_num(predictions, nan=0)
            total_samples += labels.size(0)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.detach().numpy())
    
    # Calcular métricas promedio
    loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    metrics = {"accuracy": accuracy,
               "precision": precision,
               "recall": recall,
               "f1_score": f1}
        
    print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.2f}, '
          f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
    
    return loss, metrics 



# -------------------------
# 4. Federated Learning Client
# -------------------------

class FlowerClient(NumPyClient):
    
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = test(self.net, self.testloader)
        num_examples = len(self.testloader.dataset)
        return loss, num_examples, metrics


def client_fn(context: Context):
    """
    Create and return an instance of Flower `Client`.
    No need to pass a context since this code is only for one client.
    """
        
    # Suponiendo que tienes X_train, y_train, X_test, y_test como tensores
    X_train, y_train, X_test, y_test = load_data()

    # Crear un TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Crear un DataLoader
    batch_size = 16
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Crear el modelo de red neuroanl
    input_size = X_train.shape[1]
    hidden_size1 = 5
    hidden_size2 = 5
    output_size = 1
    net = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
    
    return FlowerClient(net, trainloader, testloader).to_client()



# -------------------------
# 5. Main Execution
# -------------------------

if __name__ == "__main__":
    print()
    server_ip = input("SERVER IP: ")
    server_port = input("SERVER PORT: ")
    server_address = f"{server_ip}:{server_port}"    
    print()
    
    start_client(
        server_address=server_address,
        client=client_fn(None),
    )
