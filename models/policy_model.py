import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyModel(nn.Module):
    def __init__(self, input_channels=3, hidden_units=256, output_size=18):
        """
        Modèle CNN + MLP pour prédire un vecteur de probabilités de taille `output_size`.
        
        Parameters:
        - input_channels: Nombre de canaux d'entrée (3 pour les 3 images empilées).
        - height: Hauteur de l'image.
        - width: Largeur de l'image.
        - hidden_units: Nombre d'unités dans les couches entièrement connectées (MLP).
        - output_size: Taille de la sortie (ici 18).
        """
        super(PolicyModel, self).__init__()
        
        # CNN Layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate output size
        self._conv_output_dim = 17920 #self._get_conv_output_dim(height, width)
        
        # Flatten conv output
        self.flatten = nn.Flatten()
        
        # Dense Layer
        self.fc1 = nn.Linear(self._conv_output_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_size)
        
    def _get_conv_output_dim(self, height, width):
        """Calculer la dimension de la sortie après les couches de convolution"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, height, width)
            dummy_output = self.conv1(dummy_input)
            dummy_output = self.conv2(dummy_output)
            dummy_output = self.conv3(dummy_output)
            print(dummy_output.view(1, -1).size(1))
            return dummy_output.view(1, -1).size(1)
    
    def forward(self, x):
        """
        Propagation avant du modèle.
        
        Parameters:
        - x: Les données d'entrée, un stack de 3 images.
        
        Returns:
        - Un vecteur de taille 18 (après softmax).
        """

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        
        return x

    def l1_regularization(self):
        # Calculate the L1 Regularisation
        l1_norm = 0
        for param in self.parameters():
            l1_norm += torch.sum(torch.abs(param))
        return 0.5 * l1_norm

    def copy_weights_from(self, other_model):
        """
        Copie les poids d'un autre modèle `other_model` vers le modèle actuel.
        
        Parameters:
        - other_model (PolicyModel): Le modèle à partir duquel les poids seront copiés.
        """
        if self.state_dict().keys() != other_model.state_dict().keys():
            raise ValueError("Models' structure are different")
        
        # Copy weights
        self.load_state_dict(other_model.state_dict())
        print("Poids copiés avec succès.")
