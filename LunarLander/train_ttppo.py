#!/usr/bin/env python
""" coding: utf-8
========================================================================================
 Author: Yeo Jin Kim
 Date: 03/04/2022
 File: Main function for Lunar Lander-v2, 
            using Multi-Temporal Abstraction with Time-aware Truly PPO 
========================================================================================
 * Development environment: 
   - Linux: 3.10.0
   - Main packages: Python 3.6.9, pytorch 1.10.2, gym-box2d 0.19.0, pandas 1.3.4, numpy 1.21.5
========================================================================================
 * This code is based on the original Truly PPO code:
          https://github.com/wisnunugroho21/reinforcement_learning_truly_ppo
========================================================================================
 * Excution for each method:
 Time-aware TPPO: python T_TPPO.py -vclip=140 -network={DQN, TState, TDiscount, TQN} -seqLen=1 -trainMinTI=1 -trainMaxTI=13
 MTA-T-TPPO: python T_TPPO.py -vclip=140 -network={DQN, TState, TDiscount, TQN} -seqLen=1 -trainMinTI=1 -trainMaxTI=13

* Options
    -env: gym environment 
    -func: function approximation {Dense, LSTM}
    -network: {DQN, TState, TDiscount, TQN}
    -fold: start the training with 'fold' id for multiple random seeds
    -d: discount
    -trainMinTI: min training time interval
    -trainMaxTI: max training time interval 
    -gamma: discount factor

    -hidden: number of hidden units for deep function approximation
    -seqLen: sequence dimention (sequence length)
    -n_ep: max number of episodes for training
    -vclip: value for clipping

    -repStart: starting fold ID for repeting experiments
    -repEnd: last fold ID for repeting experiments
    -trainMode: 1 = train, 0 = test 
    -PO: 1 = partial observance, 0=full observance 
    -g: GPU ID
======================================================================================== 
"""
from collections import deque
import gym
from gym.envs.registration import register
    
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
#from torch.utils.tensorboard import SummaryWriter

import numpy as np
import sys
import numpy
import time
import datetime
import random
import os
import pandas as pd
import argparse
import shutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, HIDDEN),
                nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN),
                nn.ReLU(),
                nn.Linear(HIDDEN, action_dim),
                nn.Softmax(-1)
              ).float().to(device)
        
    def forward(self, states):
        return self.nn_layer(states)

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, HIDDEN),
                nn.ReLU(),
                nn.Linear(HIDDEN, HIDDEN),
                nn.ReLU(),
                nn.Linear(HIDDEN, 1)
              ).float().to(device)
        
    def forward(self, states):
        return self.nn_layer(states)

# class Actor_Model_LSTM(nn.Module): 
#     def __init__(self, state_dim, action_dim):
#         super(Actor_Model_LSTM, self).__init__() 
#         self.lstm = nn.LSTM(state_dim, HIDDEN, 1, batch_first=True)  # 1-layer LSTM  
#         self.linear = nn.Linear(HIDDEN, action_dim)
#         self.softmax =  nn.Softmax(-1)
        
#     def forward(self, states):
#         out, hidden = self.lstm(states)
#         out = out[:, -1, :] # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
#                             # so that it can fit into the fully connected layer
#         out = self.linear(out)
#         out = self.softmax(out)
#         return out.float().to(device) #self.nn_layer(states) 
    
# class Critic_Model_LSTM(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic_Model_LSTM, self).__init__()   

#         self.lstm = nn.LSTM(state_dim, HIDDEN, 1, batch_first=True)  # 1-layer LSTM  
#         self.linear = nn.Linear(HIDDEN, HIDDEN)
#         self.relu =   nn.ReLU()
#         self.output = nn.Linear(HIDDEN, 1)                        
        
#     def forward(self, states):
#         out, hidden = self.lstm(states)
#         out = out[:, -1, :]
#         out = self.linear(out)
#         out = self.relu(out)
#         out = self.output(out)        
#         return out.float().to(device)
    
    
class Memory(Dataset):
    def __init__(self):
        self.actions        = [] 
        self.states         = [] 
        self.rewards        = [] 
        self.dones          = [] 
        self.next_states    = [] 
        self.time_intervals  = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), np.array(self.next_states[idx], dtype = np.float32), np.array(self.time_intervals[idx], dtype=np.float32)

    def save_eps(self, state, action, reward, done, next_state, time_interval):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state) 
        self.time_intervals.append(time_interval)  ## added for Time-awareness

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:] 
        del self.time_intervals[:] ## added for Time-awareness

class Discrete():
    def sample(self, datas):
        distribution = Categorical(datas)
        return distribution.sample().float().to(device)
        
    def entropy(self, datas):
        distribution = Categorical(datas)    
        return distribution.entropy().float().to(device)
      
    def logprob(self, datas, value_data):
        distribution = Categorical(datas)
        return distribution.log_prob(value_data).unsqueeze(1).float().to(device)

    def kl_divergence(self, datas1, datas2):
        distribution1 = Categorical(datas1)
        distribution2 = Categorical(datas2)

        return kl_divergence(distribution1, distribution2).unsqueeze(1).float().to(device)  

class PolicyFunction():
    def __init__(self, gamma = 0.99, lam = 0.95):
        self.gamma  = gamma
        self.lam    = lam
        self.belief = 0.5
        self.focusTimeWindow = int(np.log(0.5)/np.log(GAMMA)*(TRAIN_MIN_TI + TRAIN_MAX_TI)/2) 
        print("gamma: {}, focusTimeWindow: {}".format(GAMMA, self.focusTimeWindow))       

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns     = []        
        
        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)
            
        return torch.stack(returns)
      
    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value           
        return q_values
      
    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae     = 0
        adv     = []     

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values          
        for step in reversed(range(len(rewards))):  
            gae = delta[step] + (1.0 - dones[step]) * self.gamma * self.lam * gae
            adv.insert(0, gae)
            
        return torch.stack(adv)
    
    def TQN_generalized_advantage_estimation(self, values, rewards, next_values, dones, time_intervals):
        gae     = 0
        adv     = []     

        temporal_gamma = np.round(self.belief**(time_intervals/self.focusTimeWindow), 6)      
        
        for step in reversed(range(len(rewards))):  
            delta   = rewards + (1.0 - dones) * temporal_gamma[step] * next_values - values          
            gae = delta[step] + (1.0 - dones[step]) * temporal_gamma[step] * self.lam * gae
            adv.insert(0, gae)
        #print("temporal_gamma: {}, delta:{}, gae:{}".format(temporal_gamma.shape, delta.shape, gae.shape))
        #print(temporal_gamma)        
        return torch.stack(adv)
    

class TrulyPPO():
    def __init__(self, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma, lam):
        self.policy_kl_range    = policy_kl_range
        self.policy_params      = policy_params
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
        self.entropy_coef       = entropy_coef

        self.distributions      = Discrete()
        self.policy_function    = PolicyFunction(gamma, lam)

    # Loss for PPO  
    def compute_loss(self, action_probs, old_action_probs, values, old_values, next_values, actions, 
                     rewards, dones, time_intervals):
        # Don't use old value in backpropagation
        Old_values          = old_values.detach()
        Old_action_probs    = old_action_probs.detach()     

        # Getting general advantages estimator and returns
        if TDISCOUNT:
            Advantages      = self.policy_function.TQN_generalized_advantage_estimation(values, rewards, 
                                                                   next_values, dones, time_intervals) 
        else:
            Advantages      = self.policy_function.generalized_advantage_estimation(values, rewards, next_values, dones)
            
        Returns         = (Advantages + values).detach()
        Advantages      = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach()

        # Finding the ratio (pi_theta / pi_theta__old): 
        logprobs        = self.distributions.logprob(action_probs, actions)
        Old_logprobs    = self.distributions.logprob(Old_action_probs, actions).detach()

        # Finding Surrogate Loss
        ratios          = (logprobs - Old_logprobs).exp() # ratios = old_logprobs / logprobs        
        Kl              = self.distributions.kl_divergence(old_action_probs, action_probs)

        # the probability ratio is clipped when the policy is out of the trust region        
        # the incentive for updating policy is removed when the policy is out of the trust region
        # roll-back: - self.policy_params * Kl
        pg_targets  = torch.where(  
            (Kl >= self.policy_kl_range) & (ratios > 1),
            ratios * Advantages - self.policy_params * Kl,  
            ratios * Advantages
        )
        pg_loss     = pg_targets.mean()

        # Getting Entropy from the action probability 
        dist_entropy    = self.distributions.entropy(action_probs).mean()

        # Getting Critic loss by using Clipped critic value
        if self.value_clip is None:
            critic_loss   = ((Returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped  = old_values + torch.clamp(values - Old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
            vf_losses1    = (Returns - values).pow(2) * 0.5 # Mean Squared Error
            vf_losses2    = (Returns - vpredclipped).pow(2) * 0.5 # Mean Squared Error        
            critic_loss   = torch.max(vf_losses1, vf_losses2).mean() 

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss 
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss

class Agent():  
    def __init__(self, state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                 batchsize, PPO_epochs, gamma, lam, learning_rate, save_path):        
        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batchsize          = batchsize       
        self.PPO_epochs         = PPO_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim               

        if FUNC == 'Dense':
            self.actor              = Actor_Model(state_dim, action_dim)
            self.actor_old          = Actor_Model(state_dim, action_dim)
            self.critic             = Critic_Model(state_dim, action_dim)
            self.critic_old         = Critic_Model(state_dim, action_dim)            
        else:
            self.actor              = Actor_Model_LSTM(state_dim, action_dim)
            self.actor_old          = Actor_Model_LSTM(state_dim, action_dim)
            self.critic             = Critic_Model_LSTM(state_dim, action_dim)
            self.critic_old         = Critic_Model_LSTM(state_dim, action_dim)            
            
        self.actor_optimizer    = Adam(self.actor.parameters(), lr = learning_rate)
        self.critic_optimizer   = Adam(self.critic.parameters(), lr = learning_rate)
        self.memory             = Memory()
        self.policy_function    = PolicyFunction(gamma, lam)  

        self.distributions      = Discrete()
        self.policy_loss        = TrulyPPO(policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma, lam)
        self.save_path          = save_path_fold
        
        if is_training_mode:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()

    def save_eps(self, state, action, reward, done, next_state, time_interval):  # TI
        self.memory.save_eps(state, action, reward, done, next_state, time_interval)

    def act(self, state):
        state           = torch.FloatTensor(state).unsqueeze(0).to(device).detach()
        action_probs    = self.actor(state)
            
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action  = self.distributions.sample(action_probs) 
        else:
            action  = torch.argmax(action_probs, 1)  
                       
              
        return action.int().cpu().item()


    def training_ppo(self, states, actions, rewards, dones, next_states, time_intervals):                 
        action_probs, values            = self.actor(states), self.critic(states)
        old_action_probs, old_values    = self.actor_old(states), self.critic_old(states)
        next_values                     = self.critic(next_states)
        
        loss    = self.policy_loss.compute_loss(action_probs, old_action_probs, values, old_values, next_values,
                                                actions, rewards, dones, time_intervals)        
  

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        self.actor_optimizer.step() 
        self.critic_optimizer.step() 

    # Update the model
    def update_ppo_TQN(self):        
        dataloader  = DataLoader(self.memory, self.batchsize, shuffle = False) # shuffle: False = a batch is sequential
        
        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states, time_intervals in dataloader:
                self.training_ppo(states.float().to(device), actions.float().to(device), 
                                  rewards.float().to(device),
                                  dones.float().to(device), next_states.float().to(device),
                                  time_intervals.float().to(device))  # time_intervals: added for Time-awareness
                
        # Clear the memory
        self.memory.clear_memory()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def save_weights(self):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()
            }, self.save_path+'weights/actor.tar')
        
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict()
            }, self.save_path+'weights/critic.tar')
        
    def load_weights(self):
        actor_checkpoint = torch.load(self.save_path+'weights/actor.tar')
        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])

        critic_checkpoint = torch.load(self.save_path+'weights/critic.tar')
        self.critic.load_state_dict(critic_checkpoint['model_state_dict'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])

class Runner_TQN():
    def __init__(self, env, agent, render, training_mode, n_update, trainMaxTI, trainMinTI, seqLen):
        self.env = env
        self.agent = agent
        self.render = render
        self.training_mode = training_mode

        self.n_update = n_update
        self.t_updates = 0
        
        # TQN parameters --------
        self.TState = TSTATE
        self.trainMinTI = trainMinTI 
        self.trainMaxTI = trainMaxTI
        
        self.func_approx = FUNC
        self.seqLen = seqLen # maxSeqLen for temporal abstraction
        
        self.normTI = self.trainMaxTI * self.seqLen
        
    def generateStateInput(self, t_state, time_interval, workMEM):
        if self.TState: # Add the normalized time interval as a state input
            t_state = np.array(t_state.tolist()+[time_interval/self.normTI])
        workMEM.append(t_state)        
        
        if len(workMEM) > self.seqLen:
            workMEM = workMEM[-self.seqLen:]
        
        # ----------------------------------------------
        # Convert t_state to RNN data format
        X = np.array(workMEM)
        if len(workMEM) < self.seqLen:
            X = np.pad(X, [(self.seqLen-len(workMEM), 0), (0,0)], mode='constant')

        if 'Dense' in self.func_approx: # concatenate the 'seqLen' number of recent states 
            X = X.flatten()

        t_state_input = np.expand_dims(X, axis=0)

        return t_state_input, workMEM         
        
    def generateStateInput_by_Momentum(self, t_state, time_interval, workMEM):
        if self.TState: # Add the normalized time interval as a state input
            t_state = np.array(t_state.tolist()+[time_interval/self.normTI])
        workMEM.append(t_state)        
        
        if len(workMEM) > self.seqLen:
            workMEM = workMEM[-self.seqLen:]
        
        # ----------------------------------------------
        # Convert t_state to RNN data format
        X = np.array(workMEM)
        if len(workMEM) < self.seqLen:
            X = np.pad(X, [(self.seqLen-len(workMEM), 0), (0,0)], mode='constant')

        # Dense network: concatenate the 'seqLen' number of recent states 
        if 'Dense' in self.func_approx: 
            X = X.flatten()

        t_state_input = np.expand_dims(X, axis=0)

        return t_state_input, workMEM 
        
    # Normalize and update the given time_interval with maxTimeInterval
    def update_ti_queue(self, time_interval, t_state_queue):
        if self.TState:          
            t_state_queue[-1][-1] = time_interval/self.normTI 
        return t_state_queue
    
        
    def run_episode(self):
        ############################################
        state = self.env.reset() 
        if PO: # partial observance
            state = np.concatenate([state[0:5], state[6:8]], axis=0)  
        
        
        done = False
        total_reward = 0
        eps_time = 0
        action_count = 0
        network_update_count = 0
        
        workMEM = [] # working memory W
        
        #----------------------------------------------------------------
        # Generate the input data for LSTM or Dense networks
        # put the current observation to the short-term memory: state_queue
        if self.TState:            
            state = np.array(state.tolist()) # to add skip time interval             
            
            
        TI_limit = random.randint(self.trainMinTI, self.trainMaxTI)        
        
        lt_cur_state, workMEM = self.generateStateInput(
                               state, TI_limit, workMEM)        
        
        lt_cur_state = lt_cur_state[0]  ## check!
        #print("state: {},  lt_cur_state: {}".format(state.shape, lt_cur_state.shape))
        ############################################
        for _ in range(10000): 
            #action = self.agent.act(state)
            action = self.agent.act(lt_cur_state)
            action_count += 1

            # Random temporal abstraction ------------- (skimmimg strategy)
             # Make a outer loop with seqLen for Random Temporal Abstraction 
            
            cum_ti, cum_mid_reward = 0, 0               
            for j in range(1, TI_limit+1):
                next_obs, mid_reward, done, _ = self.env.step(action)

                if PO:  # Partial Observance
                    next_obs = np.concatenate([next_obs[0:5], next_obs[6:8]], axis=0)  
                
                cum_ti += 1                    
                cum_mid_reward += mid_reward   

                if done:
                    break               
            
            # score += self.step_score * cum_ti  # score during the given time interval

            workMEM = self.update_ti_queue(cum_ti, workMEM) 

            # Generate random time interval
            TI_limit = random.randint(self.trainMinTI, self.trainMaxTI)

            # Next state ----------------------------------------------------------------
            lt_next_state, workMEM = self.generateStateInput(next_obs, TI_limit, workMEM) 
            lt_next_state  = lt_next_state[0]  # added
            
            #ep_reward += cum_mid_reward  
            reward = cum_mid_reward
            
            
            # update time in episode, total update, reward
            eps_time += cum_ti # 1 
            self.t_updates += 1
            total_reward +=  cum_mid_reward # reward
                        
            # -----------------------------------------
                      
            if self.training_mode: 
                self.agent.save_eps(lt_cur_state.tolist(), action, reward, float(done), lt_next_state.tolist(),
                                   cum_ti) 
                
            #state = next_state
            lt_cur_state = lt_next_state
                    
            if self.render:
                self.env.render()     
            
            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                self.agent.update_ppo_TQN()
                self.t_updates = 0
                network_update_count += 1
            
            if done: 
                break                
        
        if self.training_mode and self.n_update is None:
            self.agent.update_ppo_TQN()
            network_update_count += 1            
                    
        return total_reward, eps_time, network_update_count, action_count

    
def setup_save_path(training_mode, save_path_fold):
    print(save_path_fold)   
    if training_mode and not os.path.exists(save_path_fold):
        os.makedirs(save_path_fold)
        if not os.path.exists(save_path_fold+'weights'):
            os.makedirs(save_path_fold+'weights')

    elif training_mode:
        print("     WARNING: save path exists!!! - it will be overwritten!")

    res = pd.DataFrame(columns = ['episode','avgReward', 'epTime', 'learnTime', 'tot_net_update',
                                  'tot_action_count', 'tot_env_step'])        
    return res

    
def main():
    ############## Hyperparameters ##############
    training_mode       = TRAIN_MODE # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    if training_mode:
        load_weights        = False # If you want to load the agent, set this to True
        save_weights        = True # If you want to save the agent, set this to True
    else:
        load_weights        = True # If you want to load the agent, set this to True
        save_weights        = False # If you want to save the agent, set this to True
        
    
    reward_threshold    = 495 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    using_google_drive  = False

    render              = False # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
    n_update            = 128 # How many episode before you update the Policy. Recommended set to 128 for Discrete
    n_plot_batch        = 100000000 # How many episode you want to plot the result
    n_episode           = N_EPISODE # 100000 # How many episode you want to run
    n_saved             = N_EPISODE # How many episode to run before saving the weights

    policy_kl_range     = 0.0008 # Recommended set to 0.0008 for Discrete
    policy_params       = 20 # Recommended set to 20 for Discrete
    value_clip          = VCLIP # 1.0 # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef        = 0.05 # How much randomness of action you will get
    vf_loss_coef        = 1.0 # Just set to 1
    batchsize           = 32 # How many batch per update. size of batch = n_update / minibatch. Recommended set to 4 for Discrete
    PPO_epochs          = 4 # How many epoch per update. Recommended set to 10 for Discrete
    
    gamma               = GAMMA # Just set to 0.99
    lam                 = 0.95 # Just set to 0.95
    learning_rate       = 0.0001 # 2.5e-4 # Just set to 0.95
    ############################################# 
#     writer              = SummaryWriter()

    env_name            = ENV_NAME # Set the env you want
    env                 = gym.make(env_name)

    state_dim           = env.observation_space.shape[0]
    if PO:    
        state_dim       -= 1 # Partial Observable Env.: exclude velocities       
    
    action_dim          = env.action_space.n 
    

    # T-TPPO : temporally abstracted dimension
    seqLen              = SEQ_LEN
    trainMaxTI          = TRAIN_MAX_TI
    trainMinTI          = TRAIN_MIN_TI

    if FUNC=='Dense' and TSTATE:
        state_dim_TA    = (state_dim + 1) * seqLen
    elif FUNC=='Dense' and TSTATE==False:
        state_dim_TA    = state_dim * seqLen
    elif FUNC == 'LSTM' and TSTATE: 
        state_dim_TA    = (state_dim + 1) 
    else:
        state_dim_TA    = state_dim
    print("state dim: {}, TA state dim: {}".format(state_dim, state_dim_TA))

    agent               = Agent(state_dim_TA, action_dim, training_mode, policy_kl_range, policy_params, value_clip,\
                                entropy_coef, vf_loss_coef, batchsize, PPO_epochs, gamma, lam, learning_rate, save_path) 

    runner              = Runner_TQN(env, agent, render, training_mode, n_update, trainMaxTI, trainMinTI, seqLen)
    #############################################     
    if using_google_drive:
        from google.colab import drive
        drive.mount('/test')

    if load_weights:
        agent.load_weights()
        print('Weight Loaded')

    start = time.time()
    
    avg_score = deque(maxlen=100)
    avg_time = deque(maxlen=100)
    FirstHit_Yet = True
    hit_ep = N_EPISODE + 1
    hit_netup = 0
    hit_actcnt= 0
    hit_envstep = 0
    hit_learnTime = 0    
    env_step = 0
    tot_net_update = 0
    tot_act_count = 0
    
    res = setup_save_path(training_mode, save_path_fold)
        
    try:
        
        for i_episode in range(1, n_episode + 1):
            total_reward, eps_time, network_update_count, action_count = runner.run_episode()
            
            env_step += eps_time
            tot_net_update += network_update_count
            tot_act_count += action_count
            
            avg_score.append(total_reward)
            avg_time.append(eps_time)
            
            if training_mode: 
                
                # log episode
                if i_episode % 10 == 0:
                    learnTime = np.round((time.time()-start)/60, 1)                
                    res.loc[len(res)] = [i_episode, np.round(np.mean(avg_score), 1), np.round(np.mean(avg_time),1), \
                                         learnTime, tot_net_update, tot_act_count, env_step]
                    res.to_csv(save_path_fold+'res.csv', index=False)
                # print leaning log
                if i_episode % 200 == 0:
                    print('{}\tR {:<8.1f} len {:<5} avg (R {:<8.1f} len {:<5.1f}) {:<4.1f} min\tnet_up {:<8} act {:<10} cumlen: {:<10} TA_rate {:<5.3f}'.format(i_episode, total_reward, eps_time, np.mean(avg_score), np.mean(avg_time), learnTime, tot_net_update, tot_act_count, env_step, tot_act_count/env_step))

                # record first hit    
                if FirstHit_Yet and np.mean(avg_score) >= 200:
                    learnTime = np.round((time.time()-start)/60, 1)
                    print("   *** First >= 200: {:.1f} e {} net_up {:<8} act {:<10}".format(np.mean(avg_score), i_episode,
                                                                                   tot_net_update, tot_act_count), end=' ')   
                    print("cumlen {:<10} TA_rate {:<5.3f} learnTime: {} min".format(env_step, tot_act_count/env_step,
                                                                                    learnTime))   
                    hit_ep, hit_netup, hit_actcnt = i_episode, tot_net_update, tot_act_count
                    hit_envstep, hit_learnTime = env_step, learnTime

                if save_weights:
                    if i_episode % n_saved == 0:
                        agent.save_weights() 
                        print("saved weights")
                    
            else:  # Test
                if i_episode % 20 == 0:
                    print('{}\tR {:<8.1f} len {:<5} avg (R {:<8.1f} len {:<5.1f})'.format(i_episode, total_reward, eps_time,\
                                                                                  np.mean(avg_score), np.mean(avg_time)))
           # writer.add_scalar('rewards', total_reward, i_episode)
                    
        if training_mode:
            if FirstHit_Yet: # if the agent never hit the threshold score, set the last values of the last episode
                hit_ep, hit_netup, hit_actcnt = i_episode+1, tot_net_update, tot_act_count
                hit_envstep, hit_learnTime = env_step, learnTime
                print("   *** First >= 200: {:.1f} e {} net_up {:<8} act {:<10}".format(np.mean(avg_score), i_episode,
                                                                               tot_net_update, tot_act_count), end=' ')   
                print("cumlen {:<10} TA_rate {:<5.3f} learnTime: {} min".format(env_step, tot_act_count/env_step, learnTime))   
        else: # test       
            resTest_path = save_test_result(avg_score, avg_time)
            print("test path: ", resTest_path)

    except KeyboardInterrupt:        
        print('\nTraining has been Shutdown \n')

    finally:
        finish = time.time()
        timedelta = finish - start
        if training_mode:
            print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))            
            print("save path: {}".format(save_path_fold+'res.csv'))
         
        return hit_ep, hit_netup, hit_actcnt, hit_envstep, hit_learnTime

def parse(parser):
    # Parse arguments
    parser.add_argument("-env", type=str,default='LunarLander-v2')
    parser.add_argument("-func", type=str, choices={"Dense", "LSTM"},default="Dense")    
    parser.add_argument("-network", type=str, action='store', 
                          choices={"DQN","TState","TDiscount","TQN"}, default='TQN')
    parser.add_argument("-fold", type=int, default = 0)   # start the training with 'fold' id for multiple random seeds
    parser.add_argument("-trainMinTI",  type=int, default=1)     # min training time interval
    parser.add_argument("-trainMaxTI",  type=int, default=4)     # max training time interval 
    parser.add_argument("-gamma",  type=float, default=0.99)      # discount factor

    parser.add_argument("-hidden", type=int, default=1024) 
    parser.add_argument("-seqLen", type=int, default=3) 
    parser.add_argument("-n_ep", type=int, default=5000) 
    parser.add_argument("-vclip", type=int, default=140) 
    
    parser.add_argument("-repStart", type=int, default=0)
    parser.add_argument("-repEnd", type=int, default=10) 
    parser.add_argument("-trainMode", type=int, default=1)
    parser.add_argument("-PO", type=int, default=0)  # Partial Observable Mode
    
#     parser.add_argument("-g", type=str, default='0')              # GPU ID
    
    args = parser.parse_args()
    print(args)  
    return args 



def print_setting(i, args):
        print(" ================= {} ===================".format(i))
        save_path = 'result/{}/{}/'.format(ENV_NAME[:5],FUNC)
        if PO:
            save_path += 'PO/'
        save_path += '{}_g{}/seq{}_ti{}_{}_h{}_clip{}/'.format(NETWORK,str(GAMMA)[2:],SEQ_LEN, TRAIN_MIN_TI, \
                                                               TRAIN_MAX_TI, HIDDEN, VCLIP) 
        save_path_fold = save_path + '{}/'.format(i)    
        
        print("{}, {}-TPPO, {},seqLen: {},  TI:{}-{}, {} - hidden: {}".format(ENV_NAME, NETWORK, \
                                           FUNC, SEQ_LEN, TRAIN_MIN_TI, TRAIN_MAX_TI,  VCLIP, HIDDEN)) 
        return save_path, save_path_fold

def print_result(firstHit, info, hit_ep, hit_netup, hit_actcnt, hit_envstep, hit_learnTime):        
    firstHit.loc[len(firstHit)] = info + [hit_ep, hit_netup, hit_actcnt, hit_envstep, hit_learnTime]
    firstHit.to_csv(save_path+'firstHit.csv', index=False)

    tmp = firstHit.copy(deep=True)
    tmp.loc[len(tmp)] = ['']*6 + ['Average'] + firstHit[['ep', 'net_update', 'action_count', 'env_step', 'learnTime']].mean().round(1).values.tolist()
    print(tmp)
    return firstHit

def save_test_result(avg_score, avg_time):

    if os.path.exists(save_path+'test.csv'):
        resTest = pd.read_csv(save_path+'test.csv', header=0)
    else:
        resTest = pd.DataFrame(columns=['env','func','seqLen','minTI','maxTI','hidden','vclip','avgReward','stdReward', 'avgTime'])
        
    resTest.loc[len(resTest)] = info+[np.round(np.mean(avg_score), 1), np.round(np.std(avg_score), 1), np.round(np.mean(avg_time),1)]
    print("\t*** Avg. Reward: {:.0f} ({:.0f}), avgTime: {:.0f}".format(resTest.avgReward.mean(), resTest.avgReward.std(), resTest.avgTime.mean()))
    resTest.to_csv(save_path+'test.csv', index=False)
    return resTest_path


def set_result_file(args):    
    if args.repStart == 0:
        firstHit = pd.DataFrame(columns = ['env', 'func', 'seqLen', 'minTI', 'maxTI', 'hidden', 'vclip', 'ep', 'net_update',\
                                           'action_count', 'env_step', 'learnTime'])
    else:
        save_path, _ = print_setting(0, args)
        firstHit = pd.read_csv(save_path+'firstHit.csv', header=0) 
                               
    info = [ENV_NAME, FUNC, SEQ_LEN, TRAIN_MIN_TI, TRAIN_MAX_TI, HIDDEN, VCLIP]
    return firstHit, info



if __name__ == '__main__':  

    
    parser = argparse.ArgumentParser()
    args = parse(parser)    

    ENV_NAME = args.env
    FUNC = args.func 
    HIDDEN              = args.hidden   
    TRAIN_MIN_TI        = args.trainMinTI
    TRAIN_MAX_TI        = args.trainMaxTI     
    SEQ_LEN             = args.seqLen 
    N_EPISODE           = args.n_ep
    VCLIP               = args.vclip
    TRAIN_MODE          = args.trainMode
    NETWORK             = args.network
    PO                  = args.PO
    GAMMA               = args.gamma

    TSTATE              = False
    TDISCOUNT           = False
    if args.network == 'TQN' or args.network == 'TState':  
        TSTATE          = True        
    if args.network == 'TQN' or args.network == 'TDiscount':
        TDISCOUNT       = True
    
    totStartTime = time.time()
    firstHit, info = set_result_file(args)    
    
    for i in range(args.repStart, args.repEnd): 
        save_path, save_path_fold = print_setting(i, args)  
        
        # Train
        TRAIN_MODE = 1
        N_EPISODE = 5000        
        hit_ep, hit_netup, hit_actcnt, hit_envstep, hit_learnTime = main()
        firstHit = print_result(firstHit, info, hit_ep, hit_netup, hit_actcnt, hit_envstep, hit_learnTime)        
        
        # Test
        TRAIN_MODE = 0
        N_EPISODE = 100        
        _ = main()
        # remove wegiths
        shutil.rmtree(save_path_fold+"weights")
    totLearnTime = (time.time() - totStartTime)/60 
    print("Total Time: {:.1f} min ({:.1f} hours)".format(totLearnTime, totLearnTime/60))
