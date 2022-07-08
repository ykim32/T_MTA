import math
import torch #.cat as concatenate
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TQN_CNN_ORG(nn.Module):
    """Convolutional neural network for the Atari games."""
    NUM_FRAMES = 4
    
    def __init__(self, num_actions):
        """Initializes the neural network."""
        print("TQN_CNN_ORG")
        super(TQN_CNN_ORG, self).__init__()
        
        # First convolutional layer.
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        std = math.sqrt(2.0 / (4 * 84 * 84))
        nn.init.normal_(self.conv1.weight, mean=0.0, std=std)
        self.conv1.bias.data.fill_(0.0)
        
        # Second convolutional layer.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        std = math.sqrt(2.0 / (32 * 4 * 8 * 8))
        nn.init.normal_(self.conv2.weight, mean=0.0, std=std)
        self.conv2.bias.data.fill_(0.0)
        
        # Third convolutional layer.
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        std = math.sqrt(2.0 / (64 * 32 * 4 * 4))
        nn.init.normal_(self.conv3.weight, mean=0.0, std=std)
        self.conv3.bias.data.fill_(0.0)
                
        
        # First dense layer.
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        std = math.sqrt(2.0 / (64 * 64 * 3 * 3))
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        self.fc1.bias.data.fill_(0.0)

        #---------------------------------------        
        # Second dense layer: # 5/14 - 5/23 10 AM  
#         self.fc2 = nn.Linear(512 + 1, 256) 
#         self.out = nn.Linear(256, num_actions)
        # Output layer : First dense layer + time input layer
        self.out = nn.Linear(512 + 1, num_actions)
        

# Before 5/14, After 5/23 10 AM
    def forward(self, states, time_input):
        """Forward pass of the neural network with some inputs.
        Args:
            states: Tensor, batch of states.
        Returns:
            qs: Tensor, the q-values of the given state for all possible
                actions.
        """
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))  # Flatten input.
            
        #x = torch.cat((x.float() , time_input.float()), 1) # concatenated
        #x = F.relu(self.fc2(x))

        #---------------------------------------
        concatenated = torch.cat((x.float() , time_input.float()), 1)
        # Need 1-dense layer for time_input?
        return self.out(concatenated)

