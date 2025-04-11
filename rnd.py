import torch
import torch.nn as nn

class PredictorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, l1_lambda = .5):
        super(PredictorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)  # Output layer to predict state
        self.l1_lambda = l1_lambda

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def l1_regularization(self):
        # Calcul de la régularisation L1 sur les poids des couches
        l1_norm = 0
        for param in self.parameters():
            l1_norm += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_norm


class TargetNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(TargetNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)  # Output layer to represent state

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
