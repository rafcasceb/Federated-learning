from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import Context
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset



def load_data():
    excel_file_name = "PI-CAI_3_parte1.xlsx" 
    temp_csv_file_name = "temp_database1.csv"
    
    # Leer el archivo Excel y convertirlo a CSV
    data_excel = pd.read_excel(excel_file_name)
    data_excel.to_csv(temp_csv_file_name, sep=";", index=False)
    
    # Leer el CSV
    data = pd.read_csv(temp_csv_file_name, sep=";")
    ## print(f"Filas cargadas inicialmente: {data.shape[0]}")
    
    
    # Reemplazar comas por puntos en todas las columnas
    for col in data.columns:
        data[col] = data[col].astype(str).str.replace(',', '.')
    
    data['case_csPCa'] = data['case_csPCa'].map({'YES': 1, 'NO': 0}).astype(int)
    
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # print("Nombres de las columnas:", data.columns)

    # Verificar si se han generado valores NaN en las columnas
    if data.isnull().sum().sum() > 0:
        # print("Advertencia: Se han encontrado valores NaN después de convertir las columnas a numéricas.")
        # print(f"Filas con NaN:\n{data[data.isnull().any(axis=1)]}")
        
        # Rellenar los NaN con la mediana de cada columna
        data = data.apply(lambda col: col.fillna(col.median()) if col.isnull().any() else col)
        # print(f"Filas después de rellenar NaN con la mediana: {data.shape[0]}")
        
    print(data)

    # Mezclar los datos
    data_shuffled = shuffle(data)

    # Separar los datos en entradas (X) y salida (y)
    X = data_shuffled.iloc[:, :-1].values  # Características de entrada
    y = data_shuffled.iloc[:, -1].values   # Característica de salida
    
    # print(f"Número de columnas en los datos: {data_shuffled.shape[1]}")
    # print(f"El número de columnas de X es: {X.shape[1]}")
    
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



# Red neuronal
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)   # Fully connected layer 1
        self.relu = nn.ReLU()                            # Activation function
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) # Fully connected layer 2
        self.fc3 = nn.Linear(hidden_size2, output_size)  # Fully connected layer 3

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



def train(model, train_data, epochs=10):
    learning_rate = 0.001
    criterion = nn.BCEWithLogitsLoss()  # Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()  # Entrenar el modelo
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
    total_accuracy = 0.0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in test_data:
            # Reemplazar NaN en las etiquetas por -1 (o cualquier valor que decidas)
            labels = torch.nan_to_num(labels, nan=-1)
            
            # Reemplazar NaN en los inputs por 0  (si es necesario)
            inputs = torch.nan_to_num(inputs, nan=0)
            
            # Realizar la predicción
            outputs = model(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels)
            total_loss += loss.item() * inputs.size(0)  # Multiplicar la pérdida por el tamaño del lote
            
            # Convertir las predicciones a binario (0 o 1)
            predicted = (outputs.squeeze() > 0.5).float()  
            batch_accuracy = accuracy_score(labels.numpy(), predicted.detach().numpy())
            total_accuracy += batch_accuracy * labels.size(0)  
            
            total_samples += labels.size(0)
    
    # Calcular la pérdida y precisión promedio
    overall_loss = total_loss / total_samples
    overall_accuracy = total_accuracy / total_samples
    
    print(f'Loss on test set: {overall_loss:.4f}, Accuracy on test set: {overall_accuracy*100:.2f}%')
    
    return overall_loss, overall_accuracy 



# Suponiendo que tienes X_train, y_train, X_test, y_test como tensores
X_train, y_train, X_test, y_test = load_data()

# Crear un TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Crear un DataLoader
trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)        # !!! el batch size es la x de mat1(x,y)
testloader = DataLoader(test_dataset, batch_size=16, shuffle=True)          # !!! aquí igual

# Definir los parámetros del modelo
input_size = X_train.shape[1]      # !!! aquí la x de mat2(x,y)   (num columnas - 1)
hidden_size1 = 5                    # !!! aquí la y de mat2(x,y)
hidden_size2 = 5
output_size = 1

# Crear el modelo
net = Net(input_size, hidden_size1, hidden_size2, output_size)



# Definir el cliente de flower
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}



def client_fn(context: Context):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()



# INICIAR CLIENTE 1
if __name__ == "__main__":

    # Solicitar la IP y el puerto desde la terminal
    server_ip = input("IP: ")
    server_port = input("PORT: ")

    # Construir la dirección del servidor
    server_address = f"{server_ip}:{server_port}"

    # Iniciar el cliente de Flower con la dirección proporcionada
    start_client(
        server_address=server_address,
        client=FlowerClient().to_client(),
    )
