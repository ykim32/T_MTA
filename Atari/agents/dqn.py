"""Implements the DQN algorithm."""
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class DQN(object):
    """Implement the DQN algorithm and some helper methods."""

    def __init__( self,
        policy_file, num_actions,
        main_nn,
        target_nn,
        lr=1e-5,
        discount=0.99,
        device="cpu",
    ):
        """Initializes the class.
        Args:
            env_name: str, id that identifies the environment in OpenAI gym.
            num_actions: int, number of discrete actions for the environment.
            main_nn: torch.nn.Module, a neural network from the ../networks/* directory.
            target_nn: torch.nn.Module, a neural network with the same architecture as main_nn.
            lr: float, a learning rate for the optimizer.
            discount: float, the discount factor for the Bellman equation.
            device: the result of running torch.device().
        """
        self.num_actions = num_actions
        self.discount = discount
        self.main_nn = main_nn
        self.target_nn = target_nn
        self.device = device

        self.optimizer = optim.Adam(self.main_nn.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss.

        # added -------------------------        
        if os.path.isfile(policy_file):
            self.main_nn = torch.load(policy_file)
            print("Loaded model from {}:".format(policy_file))
        else:
            print("Initializing main neural network from scratch.")            
        # added -------------------------    

    def save_checkpoint(self, policy_file):
        torch.save(self.main_nn, policy_file)
        #print("Saved main_nn at {}".format(policy_file))

    def take_exploration_action(self, state, env, epsilon=0.1):
        """Take random action with probability epsilon, else take best action.
        Args:
            state: Tensor, state to be passed as input to the NN.
            env: gym.Env, gym environemnt to be used to take a random action
                sample.
            epsilon: float, value in range [0, 1] to define the probability of
                taking a random action in the epsilon-greedy policy.
        Returns:
            action: int, the action number that was selected.
        """
        result = np.random.uniform()
        if result < epsilon:
            return env.action_space.sample()
        else:
            q = self.main_nn(state).cpu().data.numpy()
            return np.argmax(q)  # Greedy action for state

    def train_step(self, states, actions, rewards, next_states, dones, tdiscounts):
        """Perform a training iteration on a batch of data.
        Returns:
            loss: float, the loss value of the current training step.
        """
        # Calculate targets.
        max_next_qs = self.target_nn(next_states).max(-1).values
        
#         if self.TDiscountMode: #'TDiscount' in self.mode[0]:
            #print("{}: {}".format(self.mode, tdiscounts))
        target = rewards + (1.0 - dones) * tdiscounts * max_next_qs        
#         else: # DQN
            #print(sef.discount)
#             target = rewards + (1.0 - dones) * self.discount * max_next_qs        
        
        masked_qs = self.main_nn(states).gather(1, actions.unsqueeze(dim=-1))
        loss = self.loss_fn(masked_qs.squeeze(), target.detach())
        #nn.utils.clip_grad_norm_(loss, max_norm=10)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return (loss,)
