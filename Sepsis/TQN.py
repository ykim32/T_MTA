# ========================================================================================
# Author: Anonymized
# Date: Sep 11, 2020
# File: Library for policy training: DQN, RQN, TQN
# ========================================================================================

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import math
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle
import copy
import shutil
import argparse
from functools import reduce
import datetime
import time
import argparse


#----------------
#  Recurrent Q-network / Deep Q-network
class RQnetwork():
    def __init__(self, env, myScope):
        self.phase = tf.placeholder(tf.bool)
        self.num_actions = len(env.actions)
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
        #self.hidden_state = tf.placeholder(tf.float32, shape=[None, env.hidden_size],name="hidden_state")
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        if env.func_approx=='LSTM': # 1-layer of LSTM + 2-layer of FC
            self.state = tf.placeholder(tf.float32, shape=[None, env.maxSeqLen, len(env.stateFeat)], name="input_state")
            
            lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(env.hidden_size),rnn.BasicLSTMCell(env.hidden_size)])
            self.state_in = lstm_cell.zero_state(self.batch_size,tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(\
                    inputs=self.state, cell=lstm_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope+'_rnn')
            self.rnn_output = tf.unstack(self.rnn, env.maxSeqLen, 1)[-1]
            #self.streamA, self.streamV = tf.split(self.rnn_output, 2, axis=1)

            self.fc1, self.fc1_bn, self.fc1_ac = setHiddenLayer(self.rnn_output, env.hidden_size, self.phase, last_layer=0)
            if env.DUELING:
                self.fc2, self.fc2_bn, self.fc2_ac = setHiddenLayer(self.fc1_ac, env.hidden_size, self.phase, last_layer=1)
            else:
                self.fc2, self.fc2_bn, self.fc2_ac = setHiddenLayer(self.fc1_ac, self.num_actions, self.phase, last_layer=1)
            self.fc_out = self.fc2_ac

        elif env.func_approx == 'FC': # 4 fully-connected layers 
            self.state = tf.placeholder(tf.float32, shape=[None, len(env.stateFeat)*env.maxSeqLen], name="input_state")
            
            self.fc1, self.fc1_bn, self.fc1_ac = setHiddenLayer(self.state, env.hidden_size, self.phase, last_layer=0)
            self.fc2, self.fc2_bn, self.fc2_ac = setHiddenLayer(self.fc1_ac, env.hidden_size, self.phase, last_layer=0)
            self.fc3, self.fc3_bn, self.fc3_ac = setHiddenLayer(self.fc2_ac, env.hidden_size, self.phase, last_layer=0)
            if env.DUELING:
                self.fc4, self.fc4_bn, self.fc4_ac = setHiddenLayer(self.fc3_ac, env.hidden_size, self.phase, last_layer=1)
            else:
                self.fc4, self.fc4_bn, self.fc4_ac = setHiddenLayer(self.fc3_ac, self.num_actions, self.phase, last_layer=1)     
            self.fc_out = self.fc4_ac
            
        # advantage and value streams
        if env.DUELING:
            self.streamA, self.streamV = tf.split(self.fc_out, 2, axis=1)
                           
            self.AW = tf.Variable(tf.random_normal([env.hidden_size//2,self.num_actions])) 
            self.VW = tf.Variable(tf.random_normal([env.hidden_size//2,1]))    
            self.Advantage = tf.matmul(self.streamA, self.AW)     
            self.Value = tf.matmul(self.streamV, self.VW)
        
            #Then combine them together to get our final Q-values.
            self.q_output = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        else:
            self.q_output = self.fc_out
            
        self.predict = tf.argmax(self.q_output,1, name='predict') # vector of length batch size
        
        #Below we obtain the loss by taking the sum of squares difference between the target and predicted Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,self.num_actions,dtype=tf.float32)
        
        # Importance sampling weights for PER, used in network update  xdg       
        self.imp_weights = tf.placeholder(shape=[None], dtype=tf.float32)
        
        # select the Q values for the actions that would be selected         
        self.Q = tf.reduce_sum(tf.multiply(self.q_output, self.actions_onehot), reduction_indices=1) # batch size x 1 vector
        
        # regularisation penalises the network when it produces rewards that are above the
        # reward threshold, to ensure reasonable Q-value predictions      
        self.reg_vector = tf.maximum(tf.abs(self.Q)-env.REWARD_THRESHOLD,0)
        self.reg_term = tf.reduce_sum(self.reg_vector)
        self.abs_error = tf.abs(self.targetQ - self.Q)
        self.td_error = tf.square(self.targetQ - self.Q)
        
        #square_loss = self.td_error * 0.5      
        #self.huber_loss = tf.where(self.abs_error <= env.huber_loss_threshold, 
        #                           square_loss,
        #                           env.huber_loss_threshold * (self.abs_error - 0.5 * env.huber_loss_threshold))
        #self.old_loss = tf.reduce_mean(self.huber_loss)
        
        #below is the loss when we are not using PER
        self.old_loss = tf.reduce_mean(self.td_error)


        
        # as in the paper, to get PER loss we weight the squared error by the importance weights
        self.per_error = tf.multiply(self.td_error, self.imp_weights)

        # total loss is a sum of PER loss and the regularisation term
        if env.per_flag:
            self.loss = tf.reduce_mean(self.per_error) + env.reg_lambda*self.reg_term
        else:
            self.loss = self.old_loss + env.reg_lambda*self.reg_term
            

        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
        # Ensures that we execute the update_ops before performing the model update, so batchnorm works
            self.update_model = self.trainer.minimize(self.loss)
            

def setHiddenLayer(state, hidden_size, phase, last_layer):
    if last_layer:
        fc = tf.contrib.layers.fully_connected(state, hidden_size, activation_fn=None)
    else:
        fc = tf.contrib.layers.fully_connected(state, hidden_size) 
    fc_bn = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=phase)
    fc_ac = tf.maximum(fc_bn, fc_bn*0.01)
    return fc, fc_bn, fc_ac    


#----------------
# RL

# For validation set (estimate ECR)
def process_eval_batch(env, df, X=[], X_next=[]):

    #a = df.copy(deep=True)
    idx = df.index.values.tolist()
    actions = np.array(df.Action.tolist())
    next_actions = np.array(df.next_action.tolist()) 
    rewards = np.array(df[env.rewardFeat].tolist())
    done_flags = np.array(df.done.tolist())
    
    if 'lstm' in env.keyword: # 3D-shape
        states = np.array(makeX_event_given_batch(df, env.stateFeat, env.pid, env.maxSeqLen)) 
        next_states = np.array(makeX_event_given_batch(df, env.nextStateFeat, env.pid, env.maxSeqLen))
    else: # dense networks : 2D-shape
        states = np.array(makeX_event_given_batch_dense(df, env.stateFeat, env.pid, env.maxSeqLen)) 
        next_states = np.array(makeX_event_given_batch_dense(df, env.nextStateFeat, env.pid, env.maxSeqLen))
        
        
    if 'TQN' in env.method or 'TState' in env.method: # Time interval should normalized with (60 min * maxTI)
        if 'lstm' in env.keyword: # 3D-shape
            states[:, :,-1] /= env.normTI 
            next_states[:, :,-1] /= env.normTI 
        else: # dense networks : 2D-shape  
#             states[:, -1] /= env.normTI 
#             next_states[:, -1] /= env.normTI 
            featLen = len(env.stateFeat)   # updated 060121
            for i in range(1, env.maxSeqLen+1): # Time interval should normalized with (60 min * maxTI)
                states[:, (featLen-1)*i] /= env.normTI
                next_states[:, (featLen-1)*i] /= env.normTI
            
                  
    if 'TQN' in env.method or 'TDiscount' in env.method: #'Expo' in env.character or 'Hyper' in env.character:
        tGammas = np.array(df.loc[:, env.discountFeat].tolist())
        print("** Set Exponential Discount: {}".format(np.shape(tGammas)))
    else:
        tGammas = []
        
    return (states, actions, rewards, next_states, next_actions, done_flags, tGammas)   



def update_target_graph(tf_vars, tau):
    total_vars = len(tf_vars)
    op_holder = []
    for idx,var in enumerate(tf_vars[0:int(total_vars/2)]):
        op_holder.append(tf_vars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tf_vars[idx+int(total_vars/2)].value())))
    return op_holder

def update_target(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def update_targetupdate_t (op_holder,sess):
    for op in op_holder:
        sess.run(op)
        

def remember(self, experience):
    #experience = state, action, time_interval, reward, next_state, done
        
    self.memory.append((experience))
            
    if len(self.memory) > self.train_start:
        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


# make tuples (o, a, Dt, r, o', done)
def make_tuples(env, df):

    obs = np.array(df[env.numFeat])
    actions = np.array(df.Action.tolist())
    next_actions = np.array(df.next_action.tolist()) 
    rewards = np.array(df[env.rewardFeat].tolist())
    done_flags = np.array(df.done.tolist())
    time_intv = np.array(df.TD.tolist()) 
    return obs, actions, time_intv, rewards, next_actions, done_flags 



def mt_anal(tf, env, df, testdf, valAll):
        
    env.startTime = time.time()
    learnedFeats = ['target_action'] 
    env.save_path = env.save_dir+"ckpt"      #The path to save our model to.
    
    # The main training loop is here
    tf.reset_default_graph()
    
    env.mainQN = RQnetwork(env, 'main')
    env.targetQN = RQnetwork(env, 'target')
    
    env.saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    env.policySess = tf.Session(config=env.config) 
    
    # load policy 
    env, log_df, env.maxECR, env.bestECRepoch, startIter = initialize_model(env, env.policySess,\
                                env.save_dir, env.save_path, init) # load a model if it exists
         
    # Init the validation set ----------------------------------   
    testdf['target_action'] = 0    
    edf, shockRate90, adf = test_cv(env, testdf, pe=-1, equalMode=True)
    
    if env.trainMaxTI >1 :
        apath = "data/anal/TA-{}/".format(env.method)
    else:
        apath = "data/anal/{}/".format(env.method)
    if not os.path.exists(apath):
        os.makedirs(apath)
    adf.to_csv(apath+"anal{}.csv".format(env.fold), index=False)
    #print(apath)
    env.policySess.close 

                     
def mt_rl_learning(tf, env, df, testdf, valAll):
    
    env.diffActionNum, env.diffActionRate = 0, 0
    env.maxECR = 0
    env.bestECRepoch, env.bestEpoch = 1, 1
    env.net_loss = 0.0
    
    env.startTime = time.time()
    learnedFeats = ['target_action'] 
    env.save_path = env.save_dir+"ckpt"      #The path to save our model to.
    
    # The main training loop is here
    tf.reset_default_graph()
    
    env.mainQN = RQnetwork(env, 'main')
    env.targetQN = RQnetwork(env, 'target')
    
    env.saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    trainables = tf.trainable_variables()
    env.target_ops = update_target_graph(trainables, env.tau)

    #with tf.Session(config=env.config) as sess:
    # Init the data
    env.policySess = tf.Session(config=env.config) 

    env, log_df, env.maxECR, env.bestECRepoch, startIter = initialize_model(env, env.policySess,\
                                env.save_dir, env.save_path, init) # load a model if it exists
     
    # Make training tuples -------------------------------------
    obsTrain, actionTrain, tiTrain, rewardTrain, next_actionTrain, doneTrain = make_tuples(env, df)
    expTrain = obsTrain, actionTrain, tiTrain, rewardTrain, next_actionTrain, doneTrain
    
    # Init the validation set ----------------------------------   
    testdf['target_action'] = 0
    predActions = testdf[['target_action']].copy(deep=True)
    
    # Init the replay memory with 'train_start' number of experiences -------
    tridx = 0 # index of training event
    workMEM = []
    cur_state, workMEM = env.makeTempAbstract(obsTrain[tridx], tiTrain[tridx], workMEM)

    for i in range(env.train_start):
        tridx, cur_state, workMEM = rememberRandomTIexp(env, tridx, cur_state, workMEM, expTrain, len(df))
    
    # Start training ------------------------------------------
    workMEM = []
    cur_state, workMEM = env.makeTempAbstract(obsTrain[tridx], tiTrain[tridx], workMEM)
  
    
    for i in range(startIter, env.numSteps):
        # Skip the selected time interval for the next state -----------------
        if tridx >= len(df)-1:
            tridx = 0
    
        if doneTrain[tridx] == 1: # at a terminal state
            tridx += 1 # move to the first event of the next visit 
            workMEM = [] # init the working memory and get a current state
            cur_state, workMEM = env.makeTempAbstract(obsTrain[tridx], tiTrain[tridx], workMEM)
            
        # Remember a random TI experience ----------------------------------
        if tridx + 1< len(df)-1: # the next time step should be in the data length
            tridx, cur_state, workMEM = rememberRandomTIexp(env, tridx, cur_state, workMEM, expTrain, len(df))

        # --------------------------------------------------------------------
        # Batch training
        env = batch_training(env, i)
        
        # ----------------------------------------------------        
        # Periodical Evaluation 
        if ((i+1) % env.period_eval==0): # evaluate the 1st iteration to check the initial condition 
            #saveModel2(env, env.save_dir+'models/pol_{}/'.format(i+1), i)
            env, predActions, log_df = evaluation(env, df, testdf, valAll, log_df, startIter, i, predActions)

        
        if startIter < env.numSteps:    
            log_df.to_csv(env.save_dir+"results/log.csv", index=False)

    env.policySess.close 
    return df     

def initialize_model(env, sess, save_dir, save_path, init):
    log_df = pd.DataFrame(columns = ['timestep', 'avgLoss', 'MAE', 'avgMaxQ', 'avgECR', 'learningRate', 'gamma', \
                             'ActDifNum', 'ActDifRatio', 'shockRate90'])  
    maxECR = 0
    bestShockRate = 1
    bestECRepoch = 1
    bestEpoch = 1
    startIter = 0
       
    if env.load_model == True:
#         print('Trying to load model...')
        try:
            #tf.keras.backend.set_learning_phase(0)
            restorer = tf.train.import_meta_graph(save_path + '.meta', clear_devices=True)
            restorer.restore(sess, tf.train.latest_checkpoint(save_dir))        
            print ("Model restored")
            
            log_df = pd.read_csv(save_dir+"results/log.csv", header=0)
            env.learnRate = float(log_df.tail(1).learningRate)
            maxECR = float(log_df.avgECR.max())
            bestECRepoch = int(log_df.loc[log_df.avgECR.idxmax(), 'timestep'])
#             print("Evaluation period: {} epochs".format(env.period_eval))
#             print("Previous maxECR: {:.2f} (e={})".format( maxECR,bestECRepoch))
            startIter = int(log_df.tail(1).timestep)+1     
        except IOError:
            print ("No previous model found, running default init")
            sess.run(init)
#         try:
#             per_weights = pickle.load(open( save_dir + "per_weights.p", "rb" ))
#             imp_weights = pickle.load(open( save_dir + "imp_weights.p", "rb" ))

#             # the PER weights, governing probability of sampling, and importance sampling
#             # weights for use in the gradient descent updates
#             df['prob'] = per_weights
#             df['imp_weight'] = imp_weights
#             print ("PER and Importance weights restored")
#         except IOError:
            
#             print("No PER weights found - default being used for PER and importance sampling")
    else:
        #print("Running default init")
        sess.run(init)
    print("Start Interation: {}".format(startIter))
    return env, log_df, maxECR, bestECRepoch, startIter
            
# Remember a random TI experience ----------------------------------
def rememberRandomTIexp(env, tridx, cur_state, workMEM, expTrain, df_len):
    
    obsTrain, actionTrain, tiTrain, rewardTrain, next_actionTrain, doneTrain = expTrain
    
    TI_limit = random.randint(env.trainMinTI, env.trainMaxTI) 
    
    if tridx + TI_limit > df_len-1: # Do not exceed the last event of dataset
        TI_limit -= (tridx + TI_limit - df_len)

    
    accTI, accReward = 0, 0  # accumulated TI and reward
    
    for ti in range(1, TI_limit+1): # Abstract with the selected random time interval
        if tridx+ti > df_len-1: # break when the next index is the last event of the training set
            break

        accTI += tiTrain[tridx+ti-1]        # TI is accumulated before the next state
        accReward += rewardTrain[tridx+ti]  # the reward of current state is in the next tuple                    

        if doneTrain[tridx+ti] == 1:
            break            
        
    if tridx+ti > df_len-1:
        ti -= 1
            
    # update the time interval between the current and the next state
    # Normalize and update the given time_interval with maxTimeInterval
    workMEM = env.update_ti_queue(accTI, workMEM)
    cur_state[:, -1, -1] = accTI/env.normTI # 062221 update cur_state with accTI

    ## ******* 
    #next_state, workMEM = env.makeTempAbstract(obsTrain[tridx], accTI, workMEM)#  062221
    #expNextTI = should know the next random TI for the temporal abstraction
    next_state, workMEM = env.makeTempAbstract(obsTrain[tridx+ti], tiTrain[tridx+ti], workMEM)
 
    exp = (cur_state, actionTrain[tridx], accTI, accReward, next_state, next_actionTrain[tridx], doneTrain[tridx])
    env.memory.append(exp) 

    cur_state = next_state
    tridx += ti            
    
    return tridx, cur_state, workMEM


def getTargetQ(env, nextQ, done_flags, selected_actions, rewards, tGammas):
    #print("done_flags", done_flags)
    end_multiplier = 1 - done_flags # handles the case when a trajectory is finished
            
    # Double DQN: target Q value using Q values from the target, and actions from the main
    if env.DOUBLE or 'sarsa' in env.keyword: # target next Q of Double DQN
        q_value = nextQ[range(len(nextQ)), selected_actions]
    else: # Standard DQN: target next Q on the target network with the action from the target network
        q_value = np.amax(nextQ, axis=1)
        
    # empirical hack to make the Q values never exceed the threshold - helps learning
    if env.Q_clipping:
        q_value[q_value > Q_THRESHOLD] = Q_THRESHOLD
        q_value[q_value < -Q_THRESHOLD] = -Q_THRESHOLD

    # definition of target Q
    if 'TQN' in env.method or 'TDiscount' in env.method:
        targetQ = rewards + (tGammas*q_value * end_multiplier)
    else:
        targetQ = rewards + (env.gamma*q_value * end_multiplier)    

    return targetQ



def batch_training(env, i):

    # Get a training batch
    states, actions, TIs, rewards, next_states, next_actions, done_flags, tGammas = env.get_mt_batch() 
        
    # anneal the learning rate
    if (i+1 % env.learnRatePeriod == 0) and (env.learnRate > 0.0001):
        env.learnRate *= env.learnRateFactor
        if env.learnRate < 0.0001:
            env.learnRate = 0.0001
                
    # Q values for the next timestep from target network, as part of the Double DQN update            
    nextQ = env.policySess.run(env.targetQN.q_output, 
                               feed_dict={env.targetQN.state:next_states,
                                       env.targetQN.phase:True, 
                                       env.targetQN.learning_rate:env.learnRate,
                                       env.targetQN.batch_size:env.batch_size})
        
    if 'sarsa' in env.keyword: # take the given actions from the training data
        targetQ = getTargetQ(env, nextQ, done_flags, next_actions, rewards, tGammas) 
    else:
        # For Double DQN
        actions_from_q1 = env.policySess.run(env.mainQN.predict, 
                                             feed_dict={env.mainQN.state:next_states, 
                                             env.mainQN.phase:True, 
                                             env.mainQN.learning_rate:env.learnRate, 
                                             env.mainQN.batch_size:env.batch_size})
        targetQ = getTargetQ(env, nextQ, done_flags, actions_from_q1, rewards, tGammas) 
            
    # Calculate the importance sampling weights for PER
#     if env.per_flag:
#         imp_sampling_weights = np.array(sampled_df['imp_weight'] / float(max(df['imp_weight'])))  # NO sampled_df
#         imp_sampling_weights[np.isnan(imp_sampling_weights)] = 1
#         imp_sampling_weights[imp_sampling_weights <= 0.001] = 0.001
#     else:
    imp_sampling_weights = np.array([1]*env.batch_size)
        # print("imp_sampling_weights: {} - {}".format(np.shape(imp_sampling_weights), imp_sampling_weights))
           
    # Train with the batch-----------------------------------
    _, loss, error = env.policySess.run([env.mainQN.update_model, 
                                         env.mainQN.loss, 
                                         env.mainQN.abs_error], \
                     feed_dict={env.mainQN.state: states, 
                                env.mainQN.targetQ: targetQ, 
                                env.mainQN.actions: actions, 
                                env.mainQN.phase: True, 
                                env.mainQN.imp_weights: imp_sampling_weights, 
                                env.mainQN.batch_size:env.batch_size,
                                env.mainQN.learning_rate:env.learnRate,})

    # Update target towards main network : using soft update with tau = 0.001
    update_target(env.target_ops, env.policySess)
    env.net_loss += sum(error)
             
    # Set the selection weight/prob to the abs prediction error 
    # and update the importance sampling weight
#     if env.per_flag:
#         new_weights = pow((error + env.per_epsilon), env.per_alpha)
#         df.loc[df.index.isin(sampled_df.index), 'prob'] = new_weights
#         df.loc[df.index.isin(sampled_df.index), 'imp_weight'] = pow(((1.0/len(df)) * (1.0/new_weights)), env.beta_start)

    return env

def evaluation(env, df, testdf, valAll, log_df, startIter, i, predActions):

    env.saver.save(env.policySess, env.save_path)
    env.av_loss = env.net_loss/(env.period_eval * env.batch_size)          
    env.net_loss = 0.0
                            
    # Evaluate the validation set ----------------------------------------------   
    testdf, ecr = do_eval_lstm(env.policySess, env, env.mainQN, env.targetQN, testdf, valAll)

    env.mean_abs_error = testdf.error.mean()
    env.mean_ecr = np.mean(ecr)
    env.avg_maxQ = testdf.groupby(env.pid).target_q.mean().mean() # mean maxQ by trajectory 
 
    if env.mean_ecr > env.maxECR and i > 1 :
        env.maxECR = env.mean_ecr
        env.bestECRepoch = i+1

    # Check the num of different actions from the previous evaluation -----------
    if i+1 > env.period_eval and i > startIter + env.period_eval:
        curActions = testdf[['target_action']].copy(deep=True)
        env.diffActionNum = len(testdf[curActions['target_action'] != predActions['target_action']])
        env.diffActionRate = env.diffActionNum/len(curActions)

    predActions = testdf[['target_action']].copy(deep=True)

  
    # Test ---------------------------------------------------------------------------
    #edf, shockRate90, _ = test_cv(env, testdf, pe=i, equalMode=True)
    edf, shockRate90 = test(env, pe=i, equalMode=True)
        
    # Save model ---------------------------------------------------------------- 
    log_df = saveModel(env, env.save_dir+'models/pol_{}/'.format(i+1), i, log_df, df, shockRate90)
    
    env.startTime = time.time()
          
    if env.per_flag and ((i+1) % env.beta_period == 0) and (env.beta_start < 1.0):
        env.beta_start += env.beta_increment

    return env, predActions, log_df


def test_cv(env, testdf, pe, equalMode, keyword=''):
    
    # Recommend actions for the test set
    recAction = env.policySess.run(env.mainQN.predict, feed_dict={env.mainQN.state : env.testX,\
                    env.mainQN.phase : 0, env.mainQN.batch_size : len(env.testX)})
    testdf.loc[:, 'recAction'+str(env.fold)] = recAction
    #print("{}: {:.2f}".format(method, np.mean(recAction)))
   
    # Get shock rate for each similarity rate ------------------------------
    methodName = 'M{}-{}{}-t{}'.format(env.method[0], env.func_approx[0], env.maxSeqLen, env.testMaxTI)
    avgSim, res = offlineEval_single(testdf, methodName, env.fold, 10, accumMode=True, equalMode=equalMode)
    res.reset_index(drop=True, inplace=True)
    shockRate90 = np.round(res.tail(1).shockRate.values[0], 3)    
    
    if pe >0:
        if not os.path.exists(env.save_dir + 'results/res/'):
            os.makedirs(env.save_dir + 'results/res/')
        res.to_csv(env.save_dir+'results/res/res_e{}.csv'.format(pe+1), index=False)
        env.resAll.loc[len(env.resAll)] = [pe+1, shockRate90]
        env.resAll.to_csv(env.save_dir+'results/shockRate90.csv', index=False)
        
    adf = testdf[['VisitIdentifier', 'MinutesFromArrival', 'ShockOnsetTime', 'Action', 'recAction'+str(env.fold)]]
    return res, shockRate90, adf



def do_eval_lstm(sess, env, mainQN, targetQN, testdf, testAll): 
    
    np.set_printoptions(precision=2)
    #(states, actions, rewards, next_states, next_actions, done_flags, tGammas)
    states, actions, rewards, next_states, _, done_flags, tGammas = testAll
    
    # firstly get the chosen actions at the next timestep
    actions_from_q1 = sess.run(mainQN.predict,
                               feed_dict={mainQN.state:next_states, 
                                          mainQN.phase:0,               
                                          mainQN.batch_size:len(states)})
    
    # Q values for the next timestep from target network, as part of the Double DQN update
    nextQ = sess.run(targetQN.q_output,
                     feed_dict={targetQN.state:next_states, 
                             targetQN.phase:0, 
                             targetQN.batch_size:len(states)})
    
    # handles the case when a trajectory is finished
    end_multiplier = 1 - done_flags
    
    # target Q value using Q values from the target, and actions from the main
    if env.DOUBLE: # target next Q of Double DQN
        q_value = nextQ[range(len(nextQ)), actions_from_q1]
    else: # target next Q of the original DQN
        q_value = np.amax(nextQ, axis=1)
    
    # definition of target Q
    if 'TQN' in env.method or 'TDiscount' in env.method: 
        targetQ = rewards + (tGammas * q_value * end_multiplier)            
    else:
        targetQ = rewards + (env.gamma * q_value * end_multiplier)

    # get the output q's, actions, and loss
    q_output, actions_taken, abs_error = sess.run([mainQN.q_output,mainQN.predict, mainQN.abs_error],
                                                  feed_dict={mainQN.state:states,
                                                             mainQN.targetQ:targetQ, 
                                                             mainQN.actions:actions,
                                                             mainQN.phase:False,
                                                             mainQN.batch_size:len(states)})
    
    testdf.loc[:, 'target_action'] = actions_taken
    testdf.loc[:, 'target_q'] = q_output[range(len(q_output)), actions_taken] # agent recommeded action Q
    testdf.loc[:, 'phys_q'] = q_output[range(len(q_output)), actions]  # original action Q
    testdf.loc[:, 'error'] = abs_error
    testdf.loc[:, env.Qfeat] = np.array(q_output)  # save all_q to dataframe      
        
    ecr_ret = testdf.groupby(env.pid).head(1).target_q
    if env.DEBUG:
        print("abs_error: {:.1f} mean: {} - {}".format(len(abs_error), np.mean(abs_error), abs_error))
            
    return testdf, ecr_ret


# for all the selected data
def makeX_event_given_batch(df, feat, pid, MRL): 
    X = []
    eids = df[pid].unique().tolist()
    idx = df.index.tolist()

    for i in range(len(eids)):
        edf = df[df[pid] == eids[i]]
        tmp = np.array(edf[feat])
        for j in range(len(edf)):
            X.append(np.array(tmp[:j+1]))            
    X = pad_sequences(X, maxlen = MRL, dtype='float')
    return X

def makeX_event_given_batch_dense(df, feat, pid, MRL): 
    X = []
    eids = df[pid].unique().tolist()
    idx = df.index.tolist()

    for i in range(len(eids)):
        edf = df[df[pid] == eids[i]]
        tmp = np.array(edf[feat])
        for j in range(len(edf)):
            X.append(np.array(tmp[:j+1]).flatten())            
    X = pad_sequences(X, maxlen = MRL*len(feat), dtype='float')
    return X


def test(env, pe, equalMode):
    #testdf = pd.read_csv(env.testfile, header=0)
    testdf = pd.read_csv("../Sepsis/data/h48/2_mayo_TD2m_48h_RwdNorm_test_1221.csv", header=0)
    testdf['Phys_Action'] = testdf.Action.tolist()
    
    if 'TQN' in env.method or 'TState' in env.method:
        dataType = 'TQN'
    else:
        dataType = 'DQN'

    if ('reg' in env.keyword) or (env.testMinTI == env.testMaxTI):
        modelName = '{}_{}{}_reg{}'.format(dataType, env.func_approx, env.maxSeqLen, env.testMaxTI)        
    else:
        modelName = 'M{}_{}{}_ti{}_{}'.format(dataType, env.func_approx, env.maxSeqLen, env.testMinTI, env.testMaxTI)
    if env.UPPER_TI:
        modelName += '_UPPER{}'.format(env.UPPER_TI)
        
    dataPath = 'data/ti{}/{}_val.p'.format(env.testMaxTI, modelName)
    with open (dataPath, 'rb') as fp:
        testX = np.array(pickle.load(fp))
    
    # Recommend actions for the test set
    recAction = env.policySess.run(env.mainQN.predict, feed_dict={env.mainQN.state : testX,\
                    env.mainQN.phase : 0, env.mainQN.batch_size : len(testX)})
    testdf.loc[:, 'recAction'+str(env.cvFold)] = recAction
    #print("{}: {:.2f}".format(method, np.mean(recAction)))
   
    # Get shock rate for each similarity rate ------------------------------
    methodName = 'M{}-{}{}-t{}'.format(env.method[0], env.func_approx[0], env.maxSeqLen, env.testMaxTI)
    avgSim, res = offlineEval_single(testdf, methodName, env.cvFold, 10, accumMode=True, equalMode=equalMode)
    res.reset_index(drop=True, inplace=True)
    shockRate90 = np.round(res.tail(1).shockRate.values[0], 3)    

    if not os.path.exists(env.save_dir + 'results/res/'):
        os.makedirs(env.save_dir + 'results/res/')
    res.to_csv(env.save_dir+'results/res/res_e{}.csv'.format(pe+1), index=False)
    
    resAllPath = env.save_dir+'results/shockRate90_{}.csv'.format(modelName)
    if os.path.exists(resAllPath):
        resAll = pd.read_csv(resAllPath, header=0)
    else:
        resAll = pd.DataFrame(columns = ['epoch', 'shockRate'])
    resAll.loc[len(resAll)] = [pe+1, shockRate90]
    resAll.to_csv(resAllPath, index=False)
        
#     env.resAll.loc[len(env.resAll)] = [pe+1, shockRate90]
#     env.resAll.to_csv(env.save_dir+'results/shockRate90_{}.csv'.format(modelName), index=False)

    return res, shockRate90

# ===============================================================
# Test 

def offlineEval_single(testdf, modelName, fold, numBin, accumMode, equalMode):

    testdf['Target_Action'] = testdf['recAction'+str(fold)].values
    testdf = setActionName(testdf)
        
    #simdr, res = testShockRate(modelName, testdf, numBin, accumMode, equalMode, save=False)
    simdr = pd.DataFrame(columns = ['Models', 'Shock', 'Non-shock', 'Overall'])

    testdf.rename(columns = {'Agent_Action': 'Target_Action'}, inplace=True)   
    
    _, res = compareSimilarity(testdf)
    
    if equalMode: # shockrate according to the upper N % number of visits most similar to a policy 
        res, posSimRate, negSimRate, avgSimRate = getPolicySimRate_equal(res, numBin=10)  
    else: # shock rate according to the similarity [0, 1] to a policy 
        res, posSimRate, negSimRate, avgSimRate = getPolicySimRate(res, binNum, accumMode)       
    # ---------------------------
    simdr.loc[len(simdr), :] = [modelName, posSimRate, negSimRate, avgSimRate]
    
    simdr.set_index("Models", inplace = True)
    simdr['diff'] = simdr['Non-shock'] - simdr['Shock']
      
    return simdr, res


def setActionName(df):
    df.loc[df.Phys_Action==0, 'Phys_Action'] = 'N'
    df.loc[df.Phys_Action==1, 'Phys_Action'] = 'O'
    df.loc[df.Phys_Action==2, 'Phys_Action'] = 'A'
    df.loc[df.Phys_Action==3, 'Phys_Action'] = 'V'
    
    df.loc[df.Target_Action==0, 'Target_Action'] = 'N'
    df.loc[df.Target_Action==1, 'Target_Action'] = 'O'
    df.loc[df.Target_Action==2, 'Target_Action'] = 'A'
    df.loc[df.Target_Action==3, 'Target_Action'] = 'V'
    return df

# Compare the treatments between the training data set and the agent's recommended actions
def compareSimilarity(df):
    pid = 'VisitIdentifier' 
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    col = 'Sim_Action'
    df.reset_index(drop=True, inplace=True)  # added 052521
    df[col] = 0
    df.loc[df[df.Phys_Action == df.Target_Action].index, col] = 1
    df.loc[df[((df.Phys_Action=='V')&(df.Target_Action=='AV'))|
              ((df.Phys_Action=='V')&(df.Target_Action=='OV'))].index,col] = 0.5
    df.loc[df[((df.Phys_Action=='A')&(df.Target_Action=='AV'))|
              ((df.Phys_Action=='A')&(df.Target_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Phys_Action=='O')&(df.Target_Action=='OV'))|
              ((df.Phys_Action=='O')&(df.Target_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Phys_Action=='A')&(df.Target_Action=='OAV'))|
              ((df.Phys_Action=='O')&(df.Target_Action=='OAV'))
             |((df.Phys_Action=='V')&(df.Target_Action=='OAV'))].index,col] = 1/3
    df.loc[df[((df.Target_Action=='V')&(df.Phys_Action=='AV'))|
              ((df.Target_Action=='V')&(df.Phys_Action=='OV'))].index,col] = 0.5
    df.loc[df[((df.Target_Action=='A')&(df.Phys_Action=='AV'))|
              ((df.Target_Action=='A')&(df.Phys_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Target_Action=='O')&(df.Phys_Action=='OV'))|
              ((df.Target_Action=='O')&(df.Phys_Action=='OA'))].index,col] = 0.5
    df.loc[df[((df.Target_Action=='A')&(df.Phys_Action=='OAV'))|
              ((df.Target_Action=='O')&(df.Phys_Action=='OAV'))
             |((df.Target_Action=='V')&(df.Phys_Action=='OAV'))].index,col] = 1/3
    df.loc[df[((df.Target_Action=='OA')&(df.Phys_Action=='OAV'))|
              ((df.Target_Action=='OV')&(df.Phys_Action=='OAV'))
             |((df.Target_Action=='AV')&(df.Phys_Action=='OAV'))].index,col] = 2/3
    df.loc[df[((df.Phys_Action=='OA')&(df.Target_Action=='OAV'))|
              ((df.Phys_Action=='OV')&(df.Target_Action=='OAV'))
             |((df.Phys_Action=='AV')&(df.Target_Action=='OAV'))].index,col] = 2/3
    
    # Result Analysis 
    rdf = (df.groupby(pid)[col].sum() / df.groupby(pid).size()).reset_index(name='Sim_ratio').sort_values(['Sim_ratio'], ascending=False)
    rdf['Shock'] = 0
    posvids = df[df.Shock == 1].VisitIdentifier.unique()
    rdf.loc[rdf[pid].isin(posvids), 'Shock']  = 1
    
    return df, rdf

def getPolicySimRate_equal(rdf, numBin): 
    rdf = rdf.reset_index(drop=True)
    
    col = 'Sim_ratio'
    res = pd.DataFrame(columns=['SimRate', 'pos', 'neg'])
    avgSimRate = rdf[col].mean()
    posSimRate = rdf[rdf.Shock==1][col].mean()
    negSimRate = rdf[rdf.Shock==0][col].mean()
    
    numV = int(876/numBin)
    
    for i in range(numBin):
        posNum = rdf.loc[:(i+1)*numV].Shock.sum()
        negNum = numV*(i+1) - posNum
        res.loc[len(res),:] = [0.9-i/numBin, posNum, negNum]
        
    res['tot'] = res.pos.values + res.neg.values
    res['shockRate'] = np.round((res.pos.values / res.tot.values).tolist(), 3)
    res['simPopRate'] = np.round((res.tot.values / 876).tolist(), 3) #res.loc[0, 'tot']
    
    res = res.sort_values(['SimRate'])
    
    return res, posSimRate, negSimRate, avgSimRate

def getPolicySimRate(rdf, div, accumMode): # div: how many 
    col = 'Sim_ratio'
    res = pd.DataFrame(columns=['SimRate', 'pos', 'neg'])
    avgSimRate = rdf[col].mean()
    shockSimRate = rdf[rdf.Shock==1][col].mean()
    nonShockSimRate = rdf[rdf.Shock==0][col].mean()

    for i in range(0, div): # handle similarity < 1
        if accumMode:
            condition = (rdf[col] < (i+1)/div) & (rdf[col] >= i/div)
        else:
            condition = (rdf[col] >= i/div)
            
        s0 = len(rdf[condition & (rdf.Shock == 0)]) 
        s1 = len(rdf[condition & (rdf.Shock == 1)])
        
        res.loc[len(res),:] = [(i)/div, s1, s0]
    
    # add the cases with the similarity == 1 to the previous bin
    res.loc[len(res)-1, 'pos'] += len(rdf[(rdf[col] == 1) &(rdf.Shock==1)])
    res.loc[len(res)-1, 'neg'] += len(rdf[(rdf[col] == 1) &(rdf.Shock==0)])

    res['tot'] = res.pos.values + res.neg.values
    tmp = res.copy(deep=True)
    tmp.loc[tmp.tot==0, 'tot'] = 1 # to revent divided by 0
    res['shockRate'] = res.pos.values / tmp.tot.values
    res['simPopRate'] = res.tot.values / 876 #res.loc[0, 'tot']
    
    if accumMode:
        res.loc[:, 'pos'] = res[::-1].pos.cumsum()[::-1]
        res.loc[:, 'neg'] = res[::-1].neg.cumsum()[::-1]
        res.loc[:, 'tot'] = res[::-1].tot.cumsum()[::-1]
        tmp = res.copy(deep=True)
        tmp.loc[tmp.tot==0, 'tot'] = 1 # to revent divided by 0
                    
        res.loc[:, 'shockRate'] = np.round((res.pos.values / tmp.tot.values).tolist(), 3)
        res.loc[:, 'simPopRate'] =  np.round((res.tot.values / res.loc[0, 'tot']).tolist(), 3)
        if DEBUG:
            print("pos:", len(res.pos.values), res.pos.values)
            print("tot:", len(tmp.tot.values), tmp.tot.values)
            print("shockRate: {}", np.round(res.shockRate.values.tolist(), 3))

    return res, shockSimRate, nonShockSimRate, avgSimRate


def saveModel(env, copyPath, i, log_df, df, shockRate90):

    if ((i+1) % env.saveResultPeriod==0):
        if not os.path.exists(copyPath):
            os.makedirs(copyPath)
        
        shutil.copyfile(env.save_dir+'checkpoint', copyPath+'checkpoint')
        shutil.copyfile(env.save_dir+'ckpt.data-00000-of-00001', copyPath+'ckpt.data-00000-of-00001')
        shutil.copyfile(env.save_dir+'ckpt.index', copyPath+'ckpt.index')
        shutil.copyfile(env.save_dir+'ckpt.meta', copyPath+'ckpt.meta')
        if env.per_flag:
           # Saving PER and importance weights
            with open(env.save_dir + 'per_weights.p', 'wb') as f:
                pickle.dump(df['prob'], f)
            with open(env.save_dir + 'imp_weights.p', 'wb') as f:
                pickle.dump(df['imp_weight'], f)
            # Copy PER and importance weights to a policy folder
            shutil.copyfile(env.save_dir+'imp_weights.p', copyPath+'imp_weights.p')
            shutil.copyfile(env.save_dir+'per_weights.p', copyPath+'per_weights.p')
       
    print("{}_{}/{}/{}/h{}/g{:.3f}[{:.0f}] avgL:{:.2f}, MAE: {:.2f} Q:{:.2f}, E:{:.2f} (best: {:.2f} - {}),".
           format(env.method, env.keyword, env.date, env.fold, env.hidden_size, env.gamma,(i+1),env.av_loss,\
                                env.mean_abs_error, env.avg_maxQ, env.mean_ecr, env.maxECR, env.bestECRepoch), end=' ')
    print("act:{}({:.3f})\t[Shock:{:.3f}] run time: {:.1f} m".format(env.diffActionNum, env.diffActionRate, shockRate90,
                                                             (time.time()-env.startTime)/60))
    log_df.loc[len(log_df),:] = [i+1, env.av_loss, env.mean_abs_error, env.avg_maxQ, env.mean_ecr, env.learnRate,
                                         env.gamma, env.diffActionNum, env.diffActionRate, shockRate90]
    log_df.to_csv(env.save_dir+"results/log.csv", index=False)
    
    return log_df

# Save the hyper-parameters
def saveHyperParameters(env):    
    hdf = pd.DataFrame(columns = ['type', 'value'])
    hdf.loc[len(hdf)] = ['file', env.filename]
    hdf.loc[len(hdf)] = ['method', env.method]
    hdf.loc[len(hdf)] = ['func_approx', env.func_approx]
    hdf.loc[len(hdf)] = ['Dueling', env.DUELING]
    hdf.loc[len(hdf)] = ['Double', env.DOUBLE]
    hdf.loc[len(hdf)] = ['trainMinTI', env.trainMinTI]
    hdf.loc[len(hdf)] = ['trainMaxTI', env.trainMaxTI]
    hdf.loc[len(hdf)] = ['testMinTI', env.testMinTI]    
    hdf.loc[len(hdf)] = ['testMaxTI', env.testMaxTI]
    
    hdf.loc[len(hdf)] = ['training_iteration', env.numSteps]
    hdf.loc[len(hdf)] = ['gamma', env.gamma]
    hdf.loc[len(hdf)] = ['hidden_size', env.hidden_size]
    hdf.loc[len(hdf)] = ['maxSeqLen', env.maxSeqLen]
    hdf.loc[len(hdf)] = ['batch_size', env.batch_size]
    hdf.loc[len(hdf)] = ['targetUpdate_tau', env.tau]

    hdf.loc[len(hdf)] = ['learning_rate_anneal', env.LEARNING_RATE]
    hdf.loc[len(hdf)] = ['learning_rate', env.learnRate]
    hdf.loc[len(hdf)] = ['learning_rate_factor', env.learnRateFactor]
    hdf.loc[len(hdf)] = ['learning_rate_period', env.learnRatePeriod]
    
    hdf.loc[len(hdf)] = ['Q_clipping', env.Q_clipping]
    hdf.loc[len(hdf)] = ['Q_THRESHOLD', env.Q_THRESHOLD]
    hdf.loc[len(hdf)] = ['REWARD_THRESHOLD', env.REWARD_THRESHOLD]
    hdf.loc[len(hdf)] = ['keyword', env.keyword]
    hdf.loc[len(hdf)] = ['character', env.character]
    hdf.loc[len(hdf)] = ['belief', env.belief]
    
    hdf.loc[len(hdf)] = ['numFeat', env.numFeat]
    hdf.loc[len(hdf)] = ['stateFeat', env.stateFeat]
    hdf.loc[len(hdf)] = ['actions', env.actions]
    
    hdf.loc[len(hdf)] = ['per_flag', env.per_flag]
    hdf.loc[len(hdf)] = ['per_alpha', env.per_alpha]
    hdf.loc[len(hdf)] = ['per_epsilon', env.per_epsilon]
    hdf.loc[len(hdf)] = ['beta_start', env.beta_start]
    hdf.loc[len(hdf)] = ['beta_increment', env.beta_increment]
    hdf.loc[len(hdf)] = ['beta_period', env.beta_period]
    hdf.loc[len(hdf)] = ['reg_lambda', env.reg_lambda]
    
    return hdf

    
# init for PER important weights and params
def initPER(df, env):
    df.loc[:, 'prob'] = abs(df[env.rewardFeat])
    temp = 1.0/df['prob']
    temp[temp == float('Inf')] = 1.0
    df.loc[:, 'imp_weight'] = pow((1.0/len(df) * temp), env.beta_start)
    return df

