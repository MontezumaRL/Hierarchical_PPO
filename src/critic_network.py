import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class CriticNetwork(nn.Module):
    
    def __init__(self, input_dims, alpha, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        # Convolutional layers for processing image input

        conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)  # (N, 32, 20, 20)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        relu2 = nn.ReLU()
        conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        relu3 = nn.ReLU()

        print("_______________________________")
        # Affichage de toutes les infos du cnn
        print("Affichage de toutes les infos du CNN CriticNetwork :")
        print("conv1", conv1)
        print("relu1", relu1)
        print("conv2", conv2)
        print("relu2", relu2)
        print("conv3", conv3)
        print("relu3", relu3)
        print("_______________________________")

        self.conv = nn.Sequential(
            conv1,  # Output: (N, 32, 20, 20)
            relu1,
            conv2,  # Output: (N, 64, 9, 9)
            relu2,
            conv3,  # Output: (N, 64, 7, 7)
            relu3
        )

        # Compute size after conv layers (64 * 7 * 7)
        conv_output_size = 64 * 7 * 7

        # Fully connected layers for value estimation
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Output: single value estimate
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        print("____________________________________")
        #print ("Inside Critique Network Forward") 
        print("state.shape _ before reshape", state.shape)
        #add 1 dimension to the tensor at the end (84,84)->(84,84,1)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        #print("state.shape after reshape", state.shape)
        state = state.to(self.device)
        conv_out = self.conv(state)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten before FC
        #print("conv_out.shape", conv_out.shape)
        value = self.fc(conv_out)
        #print("value.shape", value.shape)
        #print("Exiting Critique Network Forward")
        print("____________________________________")

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
