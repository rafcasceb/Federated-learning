import torch
import torch.nn as nn
import torch.nn.init as init


class NeuralNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, dropout) -> None:
        super(NeuralNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))  
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))  

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
