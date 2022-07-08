#!/usr/bin/env python
# coding: utf-8
"""========================================================================================
# Author: Yeo Jin Kim
# Date: 01/20/2021
# File: Main function for Mountain Car, 
#            using Multi-Temporal Abstraction with Time-aware deep Q-networks (T-MTA)
# ========================================================================================
# * Development environment: 
#   - Linux: 3.10.0
#   - Main packages: Python 3.7.13, Tensorflow 2.4.1, Keras 2.3.1, pandas 1.3.4, numpy 1.21.5
# ========================================================================================
# * Offline training data: 'data/' (pre-collected offline data, which can be replaced)
# ========================================================================================
# * Excution for each method:
# DQN: $ python train_mc.py -func={Dense, LSTM} -n=DQN -trainMinTI=1 -trainMaxTI=1 -testMinTI=1 -testMaxTI=8
# TQN: $ python train_mc.py -func={Dense, LSTM} -n=TQN -trainMinTI=1 -trainMaxTI=1 -testMinTI=1 -testMaxTI=8
# TA-DQN: $ python train_mc.py -func={Dense, LSTM} -n=DQN -trainMinTI=2 -trainMaxTI=3 -testMinTI=8 -testMaxTI=16
# TA-TQN: $ python train_mc.py -func={Dense, LSTM} -n=TQN -trainMinTI=2 -trainMaxTI=2 -testMinTI=8 -testMaxTI=16
#
# Options
#   -f: start the training with 'fold' id for multiple random seeds
#   -d: discount
#   -trainMaxTI: max training time interval 
#   -trainMinTI: min training time interval
#   -testMaxTI: max test time interval
#   -testMinTI: max test time interval
#   -dataTrainMaxTI: max time interval from the training offline data
#   -maxTrainEpisode: max training update
#   -maxTestPeriod:  max test period
#   -testPeriod: test every 'testPeriod' up to min(maxTrainEpisode, maxTestPeriod)

#   -hiddenSize: number of hidden units for deep function approximation
#   -seqDim: sequence dimention (sequence length)
#   -tau: For the softe update of target networks
    
#   -dueling: Dueling network - 1: on / 0: off
#   -double:  Double network - 1: on / 0: off 
#   -keyword: Additional keywords for a distinguished model name
#   -g: GPU ID
# ========================================================================================

"""
import warnings 
warnings.filterwarnings("ignore")
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # Ignore detailed log massages for GPU
import tensorflow as tf
# to prevent tensorflow version warnings
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  
import random
import gym
import numpy as np
import pickle
import time
from collections import deque
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, LSTM, Add
from keras.optimizers import Adam, RMSprop
import datetime
import argparse
import pandas as pd


def OurDenseModel(input_shape, action_space, LEARNING_RATE, hiddenSize):
    X_input = Input(input_shape)
    print("Hidden unit: {}".format(hiddenSize))

    X = Dense(hiddenSize, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    hiddenSize = int(hiddenSize/2)    
    X = Dense(hiddenSize, activation="relu", kernel_initializer='he_uniform')(X)     
    hiddenSize = int(hiddenSize/2)    
    X = Dense(hiddenSize, activation="relu", kernel_initializer='he_uniform')(X) 
    # Output Layer with # of actions: 3 nodes (left, neutral, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='Dense model')

    optimizer = Adam(LEARNING_RATE)
    model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])
    

    return model, optimizer 
    
def OurLSTMModel(input_shape, action_space, LEARNING_RATE, hiddenSize):
    print("Hidden unit: {}".format(hiddenSize))
    
    model = Sequential()
    model.add(LSTM(hiddenSize, activation='tanh', input_shape=input_shape))
    model.add(Dense(action_space))
    
    optimizer = Adam(LEARNING_RATE)
    model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])

    return model, optimizer    

# Dueling: Split into Value & Action advantage streams
def OurDuelingModel(input_shape, action_space, LEARNING_RATE, hiddenSize):
    print("Hidden unit: {}".format(hiddenSize))
    
    X_input = Input(input_shape)
    X = LSTM(hiddenSize, activation='tanh', input_shape=input_shape)(X_input)
    V = Dense(1, kernel_initializer='he_uniform')(X)
    A = Dense(action_space, kernel_initializer='he_uniform')(X)
    #Q = V + (A - A.mean(dim=1, keepdim=True))
    X = Add()([V, A])
    
    model = Model(inputs = X_input, outputs = X, name='Dueling model')
    optimizer = Adam(LEARNING_RATE)
    model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])

    return model, optimizer  

    
class DQNAgent:
    def __init__(self,TState, TDiscount, args, k):
        self.env = gym.make(args.env)
        self.envName = args.env
        self.modelName = args.network
        self.func_approx = args.func_approx
        self.seqDim = args.seqDim        
        self.gamma = args.discount        # init the temporal discount with a given constant discount
        self.static_gamma = args.discount # constant discount
                        
        self.DUELING = args.dueling
        self.DOUBLE = args.double

        self.Soft_Update = True  # target network soft update flag
        self.TAU = args.tau      # target network soft update hyperparamter        
        self.Period_Update = 100 # target network periodical update

        self.testPeriod = args.testPeriod
        self.maxTestPeriod = args.maxTestPeriod
                
        self.memory_size = 15000
        self.memory = deque(maxlen=int(self.maxTestPeriod)) 
        self.train_start = 1000 # replay start
        
        self.logInv = 100
        self.solvedEp = 0
        self.maxTrainScore = 0 # avg. highest scores
        self.highestScore = -200 # individual highest score
        self.numUpdate = 0
        self.failUpdate = 0
        
        # --------------------------------------------------------------------
        # Temporal abstraction     
        self.dataTrainMaxTI = args.dataTrainMaxTI
        self.trainMinTI = args.trainMinTI
        self.trainMaxTI = args.trainMaxTI
        self.testMinTI = args.testMinTI
        self.testMaxTI = args.testMaxTI
        
        if TDiscount:
            self.belief = 0.5
            self.focusTimeWindow = np.round(((self.trainMinTI+self.trainMaxTI)/2*((self.dataTrainMaxTI+1)/2)) * (np.log(self.belief)/np.log(self.static_gamma)),0)  # or set to 200 steps, and adjust the belief 
        else:
            self.focusTimeWindow = 'None'
 
        self.fold = k
        self.keyword = args.keyword
        
        self.hiddenSize = args.hiddenSize 
        self.outPath = 'out/{}/ti_train{}_{}_test{}_{}/{}{}_sq{}_tau{}g{}_{}/'.format(
                              self.modelName, args.trainMinTI,args.trainMaxTI, args.testMinTI,args.testMaxTI,
                              args.func_approx, args.hiddenSize, self.seqDim,
                              int(self.TAU*100), int(self.static_gamma*100), args.keyword)  
        if not os.path.exists(self.outPath):
            os.makedirs(self.outPath)
            print("create path: {}".format(self.outPath))


        self.state_size = self.env.observation_space.shape[0]        
        self.state_input_size = self.state_size
        if TState:
            self.state_input_size += 1        
        if 'Dense' in self.func_approx:  # input size = concatenated states 
            self.state_input_size *= self.seqDim
                    
        self.action_size = self.env.action_space.n
        self.EPISODES = args.maxTrainEpisode
        self.TEST_EPISODES = 100
        

        self.epsilon = 1.0           # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 32
        self.LEARNING_RATE_ANNEAL = False
        self.LEARNING_RATE = self.TAU
        self.lr_min = 0.005
        self.lr_period = 2000
        self.polFile = self.outPath+self.envName[:-3]+"_"+self.modelName+str(int(self.static_gamma*100))+"_ti"+str(self.trainMaxTI)+"_"+str(self.fold)+".h5"       

        
        if args.env=='CartPole-v0':
            self.maxScore = 200
            self.thrScore = 195 # defines "solving" as getting average reward of 195 over 100 consecutive trials.
            self.minScore = 50
            self.max_avg_log_score = 0
            self.step_score = 1
        elif args.env == 'CartPole-v1':
            self.maxScore = 500
            self.thrScore = 480
            self.minScore = 100
            self.max_avg_log_score = 0
            self.step_score = 1
        elif args.env == 'MountainCar-v0':
            self.maxScore = -100
            self.thrScore = -110
            self.minScore = -150
            self.max_avg_log_score = -200
            self.step_score = -1
        else: # Acrobot-v1
            self.thrScore = -150
            self.max_avg_log_score = -200
            self.step_score = -1

        # create main model
        if 'Dense' in self.func_approx:
            self.model, self.optimizer = OurDenseModel(
                           input_shape=(self.state_input_size,), 
                           action_space = self.action_size, 
                           LEARNING_RATE= self.LEARNING_RATE, 
                           hiddenSize = self.hiddenSize)
            self.target_model, _ = OurDenseModel(
                           input_shape=(self.state_input_size,), 
                           action_space = self.action_size, 
                           LEARNING_RATE= self.LEARNING_RATE, 
                           hiddenSize = self.hiddenSize)                           
                      
        elif self.DUELING:
            self.model, self.optimizer = OurDuelingModel(
                           input_shape=(self.seqDim, self.state_input_size,),
                           action_space = self.action_size, 
                           LEARNING_RATE= self.LEARNING_RATE, 
                           hiddenSize = self.hiddenSize)
            
            self.target_model, _ = OurDuelingModel(
                           input_shape=(self.seqDim, self.state_input_size,),
                           action_space = self.action_size, 
                           LEARNING_RATE= self.LEARNING_RATE, 
                           hiddenSize = self.hiddenSize)                           

        elif 'LSTM' in self.func_approx:
            self.model, self.optimizer = OurLSTMModel(
                           input_shape=(self.seqDim, self.state_input_size,),
                           action_space = self.action_size, 
                           LEARNING_RATE= self.LEARNING_RATE, 
                           hiddenSize = self.hiddenSize)
            
            self.target_model, _ = OurLSTMModel(
                           input_shape=(self.seqDim, self.state_input_size,),
                           action_space = self.action_size, 
                           LEARNING_RATE= self.LEARNING_RATE, 
                           hiddenSize = self.hiddenSize)
            
              
    def remember(self, state, action, time_interval, reward, next_state, done):
        experience = state, action, time_interval, reward, next_state, done
        
        self.memory.append((experience))
            
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def remember_success(self, state, action, time_interval, reward, next_state, done):
        experience = state, action, time_interval, reward, next_state, done
        self.Smemory.append((experience))
            
    # ------------------------------------------------------    
    # Get the best action 
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    # ------------------------------------------------------        
    # Batch training 
    def replay(self, memory, mode):
            
        self.numUpdate += 1  # number of parameter update
            
        minibatch = random.sample(memory, min(len(memory), self.batch_size))

        if self.func_approx == 'LSTM':
            state = np.zeros((self.batch_size, self.seqDim, self.state_input_size))
            next_state = np.zeros((self.batch_size, self.seqDim, self.state_input_size))
        else: # Dense
            state = np.zeros((self.batch_size, self.state_input_size))
            next_state = np.zeros((self.batch_size, self.state_input_size))

        action, ti, reward, done = [], [], [], []
        
        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            ti.append(minibatch[i][2])
            reward.append(minibatch[i][3])
            next_state[i] = minibatch[i][4]
            done.append(minibatch[i][5])

        # do batch prediction to save speed
        target = self.target_model.predict(state)   # init with the current Q-values from the target network
        
        nextQ_main = self.model.predict(next_state) # next Q-values from next states in the main network
        nextQ_target = self.target_model.predict(next_state) # get actions from next states in the target network
         
        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if TDiscount: # Temporal discount function 
                self.gamma = self.belief**(ti[i]/self.focusTimeWindow)
                
            if self.DOUBLE: # Double DQN
                # current Q network selects the action : a'_max = argmax_a' Q(s', a')
                a = np.argmax(nextQ_main[i])
                # target Q Network evaluates the action : Q_max = max_a' Q_target(s', a'_max)
                target[i][action[i]] = reward[i] + (1-done[i]) * self.gamma * (nextQ_target[i][a])

            else:  # Standard DQN: chooses/ evaluates the action from the target network : Q_max = max_a' Q_target(s', a')
                # For the selected action of each tuple [i] of the batch, update the target Q value
                target[i][action[i]] = reward[i] + (1-done[i]) * self.gamma * (np.amax(nextQ_target[i])) 
                    
             
        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)        
        
        # target network update 
        self.update_target_network()    

        
        
    # ------------------------------------------------------    
    # Target network update
    def update_target_network(self):
        if not self.Soft_Update:
            if (self.numUpdate + 1)% self.Period_Update == 0:
                self.target_model.set_weights(self.model.get_weights())
            return
        
        if self.Soft_Update:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

        
    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)
        
    def saveResult(self, scoreHistory):
        edf = pd.DataFrame(columns = ['episode', 'score'])
        for i in range(len(scoreHistory)):
            edf.loc[len(edf)] = [i, scoreHistory[i]]
        edf.to_csv(self.outPath + 'trainScore_'+self.modelName+str(int(self.static_gamma*100))+"_ti"+str(self.trainMaxTI)\
                   +'_'+str(self.fold)+'.csv', index=False)
        
    def saveTest(self, meanScore, stdScore):
        testFile = self.outPath + '{}_test_results.csv'.format(self.envName[:-3])
        if os.path.exists(testFile):
            resdf = pd.read_csv(testFile, header = 0)
        else:
            resdf = pd.DataFrame(columns = ["env,", "model", "funcApprox",  "seqDim", "hiddenSize", 
                                            "discount", "trainMaxTI", "thrScore", "fold","solvedEpisode",
                                            "trainMaxScore ", "avgTestScore", "stdTestScore"])
        resdf.loc[len(resdf)] = [self.envName[:-3], self.modelName, self.func_approx,
                                 self.seqDim, self.hiddenSize, self.gamma,
                                 self.trainMaxTI, self.thrScore, self.fold, self.solvedEp, 
                                 self.maxTrainScore, meanScore, stdScore]
        resdf.to_csv(testFile, index=False)
    
    # Normalize and update the given time_interval with maxTimeInterval
    def update_ti_queue(self, time_interval, t_state_queue):
        if TState:          
            t_state_queue[-1][-1] = time_interval/(self.trainMaxTI*self.dataTrainMaxTI) 
        return t_state_queue
        
    
    def generateStateInput(self, t_state, time_interval, workMEM):

        if TState: # Add the normalized time interval as a state input
            t_state = np.array(t_state.tolist()+[time_interval/(self.trainMaxTI*self.dataTrainMaxTI)])
            
        workMEM.append(t_state)        
        
        if len(workMEM) > self.seqDim:
            workMEM = workMEM[-self.seqDim:]
        
        # ----------------------------------------------
        # Convert t_state to RNN data format
        X = np.array(workMEM)

        if len(workMEM) < self.seqDim:
            X = np.pad(X, [(self.seqDim-len(workMEM), 0), (0,0)], mode='constant')

        if 'Dense' in self.func_approx: # concatenate the 'seqDim' number of recent states 
            X = X.flatten()

        t_state_input = np.expand_dims(X, axis=0)

        return t_state_input, workMEM       
                      

      
    def showSetting(self):    
        print("===================================================")  
        print("Method: {}-{} (Doub-{}, Duel-{})".format(self.modelName, self.func_approx, self.DOUBLE, self.DUELING))     
        print("FocusTimeWindow: {}, Time interval: [{}, {}]".format(self.focusTimeWindow, self.trainMinTI, self.trainMaxTI))  
        print("Momory size: {} / train_start: {}".format(self.memory_size, self.train_start))          
        print("===================================================")  
      
    # Offline training for Temporal Abstraction with TQN      
    def run_offline_irregular_TQN(self):
        DEBUG = False
        self.showSetting()
        sucMEM = []
        
        scoreHistory = []
        firstPolicy = True
        tot_ep_reward = []
        train_start_flag = False
        Solved = False
        train_start_ep = 0
        e = 0    
        idx = 0
        highestScore = 0
        
        
        if 'reg' in self.keyword:
            nextTimeInterval = self.trainMaxTI
        else:
            nextTimeInterval = int((self.trainMinTI + self.trainMaxTI)/2)
            
        dataPath = 'data/mc_offline_DQN_polTIreg8_genMaxTI8_ep100_step2937_score-130.505.pk'
        print("Offline data: {}".format(dataPath))
        
        with open (dataPath, 'rb') as fp:
            data = np.array(pickle.load(fp))

        dataLastIdx = len(data)-1
    
        # experience: [episode_id, step_id, obs, action, time_interval, acc_reward, done] 
        ep_idx, step_idx, obs_idx, act_idx, ti_idx, reward_idx, done_idx = 0, 1, 2, 3, 4, 5, 6
        
        
        while (e < self.EPISODES): 
            e += 1
            
            init_obs = data[idx][obs_idx] # For offline learning (c.f. for online learning: use 'self.env.reset()')
            if TState:            
                init_obs = np.array(init_obs.tolist()) # to add skip time interval 

            done = False
            
            workMEM = [] # working memory W
            
            action_list = []
            timeIntervalList = []
            score, goalScore, ep_reward = 0, 0, 0
            uncertDecNum, confDecNum, totDecNum = 0, 0, 0
                        
            time_interval_limit = random.randint(self.trainMinTI, self.trainMaxTI)  
            
            #----------------------------------------------------------------
            # Generate the input data for LSTM or Dense networks
            # put the current observation to the short-term memory: state_queue
            lt_cur_state, workMEM = self.generateStateInput(
                                   init_obs, time_interval_limit, workMEM)
            
            while not done:
                if idx >= len(data)-1:
                    idx = 0

                action = data[idx][act_idx]   #self.act(lt_cur_state) 
                action_list.append(action)
                
                cum_mid_reward = 0
                cum_ti = 0   
                                        
                for j in range(1, time_interval_limit+1):
                    #next_obs, mid_reward, done, _ = self.env.step(action)
                    next_obs = data[idx+j][obs_idx]
                    mid_reward = data[idx+j][reward_idx] # the reward of current state is in the next tuple                    
                    ti = data[idx+j-1][ti_idx]   # TI is accumulated before the next state (upTI3-Added: 1/9/2021/1:28PM)
                    done = data[idx+j][done_idx]  
                    
                    cum_ti += ti                    
                    cum_mid_reward += mid_reward 
                    
                    
                    # temporal abstraction only for the states with the same action taken                     
                    if action != data[idx+j][act_idx] or done: 
                        break
                        
                    if idx + j >= len(data)-1:
                        break
                
                score += self.step_score * cum_ti  # score during the given time interval
                
                # ** Update the actual time interval (upTI-Added: 1/9/2021/12:30PM)
                workMEM = self.update_ti_queue(cum_ti, workMEM) 
                
                if done : #and score >= goalScore:  # success episode
                    if self.highestScore < score:
                        self.highestScore = score
                
                timeIntervalList.append(cum_ti)
                
                # Generate random time interval
                TI_limit = random.randint(self.trainMinTI, self.trainMaxTI)
                
                if idx +j + TI_limit > dataLastIdx: # Do not exceed the last event of dataset
                    TI_limit -= (idx +j + TI_limit - dataLastIdx)

                # Next state ----------------------------------------------------------------
                #expectedNextTI = time_interval_limit*((self.trainMinTI + self.trainMaxTI+1)/2) # For online, use avg. TI 
                expNextTI = 0
                for tmp in range(TI_limit): 
                    expNextTI += data[idx+j+tmp][ti_idx]
                
                lt_next_state, workMEM = self.generateStateInput(next_obs, expNextTI, workMEM) 
                
                ep_reward += cum_mid_reward
                
                if done and np.abs(score) < self.env._max_episode_steps: 
                    reward = 200
                else:
                    reward = cum_mid_reward
                    
                # Remember to replay memory D   
                self.remember(lt_cur_state, action, cum_ti, reward, lt_next_state, done) # long-term    
               
                # Move to the next target state
                lt_cur_state = lt_next_state
                idx += j
                if idx >= dataLastIdx: 
                    idx = 0

                # ----------------------------------------------------------------
                # TRAINING: Optimize the networks with a batch from the replay memory M
                if len(self.memory)  >= self.train_start:
                    self.replay(self.memory, 'f')  
                            
                totDecNum +=1

                # Periodical Test ----------------------------------------------------
                if ( self.numUpdate % self.testPeriod == 0 ) and (self.numUpdate > 0):
                    self.save(self.polFile)
                    avgTestScore, stdTestScore = self.test(e)                    
                        
                if self.numUpdate >= self.maxTestPeriod:
                    break
                #----------------------------------------------------------------    
                    
            if len(self.memory) >= self.train_start : # At the end of episode after staring the training        
                if not train_start_flag:              # Reset the episode number once the replay start
                    print("*** Start training")
                    train_start_ep = e
                    train_start_flag = True
                    e=0 
                              
                scoreHistory.append(score)
                tot_ep_reward.append(ep_reward)
                avg_log_score =  np.mean(scoreHistory[-self.logInv:])
                                
                # Update the learning rate 
                if self.LEARNING_RATE_ANNEAL and (e % self.lr_period==0) and (self.LEARNING_RATE > self.lr_min):
                    self.LEARNING_RATE *= 0.99 
                    self.optimizer.lr.assign(self.LEARNING_RATE)
                    
                if self.max_avg_log_score < avg_log_score:
                    self.max_avg_log_score = avg_log_score
                
                    
            if self.numUpdate == self.maxTestPeriod:
                break

        self.solvedEp = e
        self.maxTrainScore = self.max_avg_log_score
        self.saveResult(scoreHistory)
         
        print("* Save the final model: ", self.polFile)
        
        return e, self.numUpdate, self.max_avg_log_score

    # ----------------------------------
    # Online test 
    def test(self, trainEpisode):        

        self.load(self.polFile)    # load the periodically saved policy
        rewardList = []
        timeIntervalList=[]
        totDecNumList = []
        
        for e in range(self.TEST_EPISODES):
            obs = self.env.reset()
            workMEM = []            
            action_list = []
            done = False

            ep_reward, step, totDecNum = 0, 0, 0
            while not done:
                #----------------------------------------------------------------
                # Select the random time interval for abstraction                
                time_interval_limit= random.randint(self.testMinTI,self.testMaxTI)
                
                # Generate the state input for LSTM or Dense networks
                lt_state, workMEM = self.generateStateInput(
                              obs, time_interval_limit, workMEM)                                              
                #----------------------------------------------------------------               
                action = np.argmax(self.model.predict(lt_state))                
                action_list.append(action)
                
                cum_mid_reward = 0
                for j in range(1, time_interval_limit+1):
                    obs, mid_reward, done, _ = self.env.step(action)  # Online test
                    cum_mid_reward += mid_reward

                    if done:
                        break  
                
                time_interval = j     # j : Accumulated time
                timeIntervalList.append(time_interval)
                ep_reward += cum_mid_reward    # episode reward
                totDecNum +=1                    
 
                if done:
                    break
                
            rewardList.append(ep_reward)
            totDecNumList.append(totDecNum)
        
        avgTestScore = np.mean(rewardList)
        stdTestScore = np.std(rewardList)
        totDec = np.mean(totDecNumList)
    
        info.loc[len(info)] = [trainEpisode, self.numUpdate, self.maxTrainScore, avgTestScore, \
                               stdTestScore, totDec]
        info.to_csv(agent.outPath+"train_sum.csv", index=False)             
        print("[{}: {}] {}(h:{})-ti:{:.1f}\ta: {:.1f} ({:.1f})/ score: {:.1f}, std: {:.1f} (ep: {}) GM:{}".
            format(self.modelName, self.keyword, self.numUpdate, self.highestScore, np.mean(timeIntervalList), 
                   np.mean(action_list), totDec, avgTestScore, stdTestScore, trainEpisode, len(self.memory)))

        self.saveTest(avgTestScore, stdTestScore)
        return avgTestScore, stdTestScore       



def parse(parser):
    # Parse arguments
    parser.add_argument("-env", "--env", type=str,default='MountainCar-v0')
    parser.add_argument("-func", "--func_approx", type=str, choices={"Dense", "LSTM"},default="LSTM")    
    parser.add_argument("-n", "--network", type=str, action='store', 
                          choices={"DQN","TState","TDiscount","TQN"}, required=True)
    parser.add_argument("-f", "--fold", type=int, default = 0)   # start the training with 'fold' id for multiple random seeds
    parser.add_argument("-d", "--discount", type=float, action='store', help="discount", default=0.99)
    parser.add_argument("-trainMaxTI",  type=int, default=1)     # max training time interval 
    parser.add_argument("-trainMinTI",  type=int, default=1)     # min training time interval
    parser.add_argument("-testMaxTI",  type=int, default=8)      # max test time interval
    parser.add_argument("-testMinTI",  type=int, default=1)      # max test time interval
    parser.add_argument("-dataTrainMaxTI",  type=int, default=4) # max time interval from the training offline data
    
    parser.add_argument("-m", "--maxTrainEpisode", type=int, default=15000) # For the max training update, it will choose
    parser.add_argument("-maxTestPeriod", type=int, default=15000)          # min(maxTrainEpisode, maxTestPeriod)
    parser.add_argument("-testPeriod", type=int, default=500)               # test every 'testPeriod' up to 'maxTestPeriod'

    parser.add_argument("-hs", "--hiddenSize", type=int, default=64) 
    parser.add_argument("-sd", "--seqDim", type=int, default=2) 
    parser.add_argument("-tau", type=float, default=0.01)         # For the softe update of target networks
    
    parser.add_argument("-dueling", type=int, default=0)          # Dueling network - 1: on / 0: off
    parser.add_argument("-double", type=int, default=0)           # Double network - 1: on / 0: off 
    parser.add_argument("-keyword", type=str, default='')         # Additional keywords for a distinguished model name
    
    parser.add_argument("-g", type=str, default='0')              # GPU ID
    
    args = parser.parse_args()
    print(args) 
    return args    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train and test on CartPole")
    args = parse(parser)

    config = tf.ConfigProto() #tf.compat.v1.ConfigProto()#
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    session = tf.Session(config=config)  #tf.compat.v1.Session(config=env.config)

    meanScoreList = []
    stdScoreList = []
    TDiscount = False
    TState = False
    
    if 'TQN' in args.network or 'TDiscount' in args.network:
        TDiscount = True    # Time-aware discount
    if 'TQN' in args.network or 'TState' in args.network:
        TState = True       # Time-aware state approximation
        
    
    info = pd.DataFrame(columns = ['solvedEp', 'numUpdate', 'maxTrainScore',
               'avgTestScore','stdTestScore', 'totDecison'])
    runTime = []           

    print("Start experiment: {}".format(str(datetime.datetime.now().strftime('%m/%d %H:%M'))))
    solved_ep = []
    totFolds =10   # default: train 10 times with 10 random seeds

    for k in range(args.fold, totFolds): #args.fold+1
        startTime = time.time()
        random.seed(k)          # change the random seed
        
        print("\n***** {} - Fold {} *****".format(args.network, k))              
        agent = DQNAgent(TState, TDiscount, args, k)                    # Init the agent
        e, numUpdate, maxTrainScore = agent.run_offline_irregular_TQN() # Training          

        curRunTime = (time.time()-startTime)/60
        runTime.append(curRunTime)
    print("Learning Time: {:.2f} min".format(curRunTime))

    resFile = agent.outPath + "test_avg10.csv"   # final average scores over 10 random seeds
    if not os.path.exists(resFile):
        resdf = pd.DataFrame(columns = ["env", "model", "funcApprox",  "seqDim", 
                "hiddenSize", "dueling", "double", "per", "discount",
                "trainMaxTI", "fold", "avgSolvedEp","stdSolvedEp", 
                "avgNumUpdate", "avgNumUpdate", "avgTestScore", "stdTestScore"])
    else:
        resdf = pd.read_csv(resFile, header=0)

    resdf.loc[len(resdf)] = [args.env[:-3], args.network, args.func_approx, 
                             args.seqDim, args.hiddenSize, args.dueling, args.double,
                             args.per, args.discount, 
                             args.trainMaxTI, totFolds, info.solvedEp.mean(),
                             info.solvedEp.std(), info.numUpdate.mean(), info.numUpdate.std(),
                             info.avgTestScore.mean(), info.avgTestScore.std()]

    resdf.to_csv(resFile, index=False)
    print("===================================================================") 
    print(" * {}: {}_{} ".format(args.env, args.network, args.func_approx))
    print(" * Solved episodes: {:.1f} - {:.1f}".format(info.solvedEp.mean(), info.solvedEp.std()))
    print(" * Final avg. scores: {:.1f}, {:.1f}".format(info.avgTestScore.mean(), 
                                                        info.avgTestScore.std()))
    print("Learning Time: avg: {:.2f} / total: {:.2f} min".format(np.mean(runTime),np.sum(runTime)))     
    print("End experiment: {}".format(str(datetime.datetime.now().strftime('%m/%d %H:%M'))))
