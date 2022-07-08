import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    """Dense neural network for simple games."""

    def __init__(self, num_inputs, num_actions):
        """Initializes the neural network."""
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
#         self.fc4 = nn.Linear(512, 32)
        self.out = nn.Linear(64, num_actions)

    def forward(self, states):
        """Forward pass of the neural network with some inputs.
        Args:
            states: Tensor, batch of states.
        Returns:
            qs: Tensor, the q-values of the given state for all possible
                actions.
        """
        x = F.relu(self.fc1(states))       
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))                
        return self.out(x)
