import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical



class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, chkpt_dir='tmp/ppo'):
        
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')


        print("_______________________________")
        print("input_dims[0]", input_dims[0])
        print("input_dims[1]", input_dims[1])
        print(input_dims)
        print("_______________________________")

        conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)  
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        relu2 = nn.ReLU()
        conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        relu3 = nn.ReLU()
        
        self.conv = nn.Sequential(
            conv1,  # Output: (N, 32, 20, 20)
            relu1,
            conv2,  # Output: (N, 64, 9, 9)
            relu2,
            conv3,  # Output: (N, 64, 7, 7)
            relu3
        )

        print("_______________________________")
        # Affichage de toutes les infos du cnn
        print("conv1", conv1)
        print("relu1", relu1)
        print("conv2", conv2)
        print("relu2", relu2)
        print("conv3", conv3)
        print("relu3", relu3)
        print("_______________________________")

        conv_output_size = 64 * 7 * 7

        # Fully connected layers for action selection

        linear1 = nn.Linear(conv_output_size, 512)
        relu4 = nn.ReLU()
        linear2 = nn.Linear(512, n_actions)
        softmax = nn.Softmax(dim=-1)

        self.fc = nn.Sequential(
            linear1,
            relu4,
            linear2,
            softmax  # Convert logits to probabilities
        )

        print("_______________________________")
        # Affichage de toutes les infos du fc
        print("linear1", linear1)
        print("relu4", relu4)
        print("linear2", linear2)
        print("softmax", softmax)
        print("_______________________________")

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        print("____________________________________")
        print ("Inside Actor Network Forward") 
        #add 1 dimension to the tensor at the end (84,84)->(84,84,1)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        print("state.shape", state.shape)
        batch_size = state.shape[0]
        state = state.to(self.device)
        conv_out = self.conv(state)
        #print("conv_out.shape before flatten", conv_out.shape)       
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten before FC layer
        print("conv_out.shape after flatten", conv_out.shape)
        l = []
        for i in range(batch_size):
            l.append(conv_out)
        conv_out = T.stack(l)
        print("conv_out.shape after reshape", conv_out.shape)
        action_probs = self.fc(conv_out)
        #print("conv_out.shape after reshape", conv_out.shape)
        action_probs = self.fc(conv_out)
        #print("action_probs", action_probs)
        print(" Exiting Actor Network Forward")
        print("________________________________")
        print("CATEGORICAL(action_probs)", Categorical(action_probs))
        
        return Categorical(action_probs)  # Output a probability distribution

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
