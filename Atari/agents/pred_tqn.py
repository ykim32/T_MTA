"""Implements the TQN (Time-aware Q-network) algorithm."""
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------
# ** Hyperparameters for Temporal discount function 
# 1) focusTimeWindow = 1/(ln(gamma)/ln(belief)) = ln(gamma)/ln(belief)
#   e.g. 1/ (np.log(0.99)/np.log(0.5)), np.log(0.5)/np.log(0.99)
# 2) static_gamma = belief ** (avg.timeInterval/targetTime)
#   e.g. regular or avg.time interval=1 : 0.5**(1/focusTimeWindow), 
#   e.g. irregular : 0.5**(avgTimeInterval/focusTimeWindow)

# NOTE: DQNirr --> should use "gamma = 0.5 ** (avg.TimeInterval/focusTimeWindow)
#         NOT gamma = 0.99   or compare two gammas.

# For TQN
# ADD time_interval as a state feature
# belief = 0.5  # fixed
# focusTimeWindow = np.log(0.5)/np.log(0.99) # without domain knoweldge



class PredTQN(object):
    """Implement the TQN algorithm and some helper methods."""


    def __init__(
        self,
        mode,
        load_frame, 
        
        env_name,
        num_actions,
        main_nn,
        target_nn,
        lr=1e-5,
        discount = 0.99,
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
        
        self.belief = 0.5,
        self.discount=discount, # temporal discount during training
        self.focusTimeWindow = np.log(self.belief)/np.log(self.discount),
        
        self.num_actions = num_actions
        self.main_nn = main_nn
        self.target_nn = target_nn
        self.device = device

        self.optimizer = optim.Adam(self.main_nn.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss.

        # yeojin -------------------------
        self.save_path = "./saved_models/{}/{}/{}-{}-".format(mode, env_name,\
                                           mode,env_name,type(main_nn).__name__)
        self.policy_file = "{}{}.pt".format(self.save_path, load_frame) 
        
        if not os.path.exists("saved_models/{}/{}/".format(mode, env_name)):
            os.makedirs("saved_models/{}/{}/".format(mode, env_name))
            
        if os.path.isfile(self.policy_file):
            self.main_nn = torch.load(self.policy_file)
            print("Loaded model from {}:".format(self.policy_file))
        else:
            print("Initializing main neural network from scratch.")
        # yeojin -------------------------    
   
    
    def save_checkpoint(self, cur_frame):
        saveFile = self.save_path + str(cur_frame) + ".pt"
        torch.save(self.main_nn, saveFile)
        print("Saved main_nn at {}".format(saveFile))

    def take_exploration_action(self, state, time_input, env, epsilon=0.1):
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
            #print("rand {} < epsilon {}".format(result, epsilon)) 
            return env.action_space.sample(), None
        else:
            q, t_state = self.main_nn(state, time_input) #.cpu().data.numpy()
            q = q.cpu().data.numpy()
            t_state = t_state.cpu().data.numpy()
            
            return np.argmax(q), t_state  # Greedy action for state

    def train_step(self, states, timeInvs, actions, rewards, next_states,\
                   dones, importances):
        """Perform a training iteration on a batch of data.
        Returns:
            loss: float, the loss value of the current training step.
        """
        # Calculate targets.
        max_next_qs = self.target_nn(next_states, timeInvs).max(-1).values
        target = rewards + (1.0 - dones) * self.discount * max_next_qs
        masked_qs = self.main_nn(states, timeInvs).gather(1,\
                                               actions.unsqueeze(dim=-1))
        loss = self.loss_fn(masked_qs.squeeze(), target.detach())
        #nn.utils.clip_grad_norm_(loss, max_norm=10)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return (loss,)
