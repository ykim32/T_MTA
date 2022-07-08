import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import pickle
import copy
import random
import argparse
import time
from keras.preprocessing.sequence import pad_sequences


def initData_env(file, env):
    df = pd.read_csv(file, header=0) 
    df = df.sort_values([env.pid, env.timeFeat])
    df.reset_index(drop=True, inplace=True)
                  
    # set 'done_flag'
    df['done']=0
    df.loc[df.groupby(env.pid).tail(1).index, 'done'] = 1
    # next actions
    df['next_action'] = 0 
    df.loc[:, 'next_action'] = df.groupby(env.pid).Action.shift(-1).fillna(0)
    df['next_action'] = pd.to_numeric(df['next_action'], downcast='integer')
    # df.loc[:, 'next_actions'] = np.array(sdf.groupby('VisitIdentifier').Action.shift(-1).fillna(0), dtype=np.int)

    # next states
    env.nextStateFeat = ['next_'+s for s in env.stateFeat]
    df[env.nextStateFeat] = df.groupby(env.pid).shift(-1).fillna(0)[env.stateFeat]
    
    # action Qs
    Qfeat = ['Q'+str(i) for i in range(len(env.actions))]
    for f in Qfeat:
        df[f] = np.nan
        
    # Temporal Difference for the decay discount factor gamma * exp (-t/tau)
    df.loc[:, 'TD'] = (df.groupby(env.pid)[env.timeFeat].shift(-1) - df[env.timeFeat]).fillna(0).tolist()
    return df

        
def initialize_model(env, sess, save_dir, df, save_path, init):
    log_df = pd.DataFrame(columns = ['timestep', 'avgLoss', 'MAE', 'avgMaxQ', 'avgECR', 'learningRate', 'gamma', \
                             'ActDifNum', 'ActDifRatio', 'shockRate90'])  
    maxECR = 0
    bestShockRate = 1
    bestECRepoch = 1
    bestEpoch = 1
    startIter = 0
       
    if env.load_model == True:
        print('Trying to load model...')
        try:
            
            #tf.keras.backend.set_learning_phase(0)
            restorer = tf.train.import_meta_graph(save_path + '.meta', clear_devices=True)
            restorer.restore(sess, tf.train.latest_checkpoint(save_dir))        
            print ("Model restored")
            
            log_df = pd.read_csv(save_dir+"results/log.csv", header=0)
            env.learnRate = float(log_df.tail(1).learningRate)
            maxECR = float(log_df.avgECR.max())
            bestECRepoch = int(log_df.loc[log_df.avgECR.idxmax(), 'timestep'])
            print("Evaluation period: {} epochs".format(env.period_eval))
            print("Previous maxECR: {:.2f} (e={})".format( maxECR,bestECRepoch))
            startIter = int(log_df.tail(1).timestep)+1     
        except IOError:
            print ("No previous model found, running default init")
            sess.run(init)
        try:
            per_weights = pickle.load(open( save_dir + "per_weights.p", "rb" ))
            imp_weights = pickle.load(open( save_dir + "imp_weights.p", "rb" ))

            # the PER weights, governing probability of sampling, and importance sampling
            # weights for use in the gradient descent updates
            df['prob'] = per_weights
            df['imp_weight'] = imp_weights
            print ("PER and Importance weights restored")
        except IOError:
            
            print("No PER weights found - default being used for PER and importance sampling")
    else:
        #print("Running default init")
        sess.run(init)
    print("Start Interation: {}".format(startIter))
    return df, env, log_df, maxECR, bestECRepoch, startIter
 

def process_eval_batch(env, df, data):
    a = data.copy(deep=True)   
    
    actions = np.squeeze(a.Action.tolist())
    next_actions = np.squeeze(a.next_action.tolist()) 
    rewards = np.squeeze(a[env.rewardFeat].tolist())
    done_flags = np.squeeze(a.done.tolist())
    
    if env.maxSeqLen > 1: # LSTM
        states = makeX_event_given_batch(df, a, env.stateFeat, env.pid, env.maxSeqLen)
        next_states = makeX_event_given_batch(df, a, env.nextStateFeat,  env.pid, env.maxSeqLen)   
    else:
        states = a.loc[:, env.stateFeat].values.tolist() 
        next_states =  a.loc[:, env.nextStateFeat].values.tolist()
    
    tGammas = np.squeeze(a.loc[:, env.discountFeat].tolist())
    yield (states, actions, rewards, next_states, next_actions, done_flags, tGammas, a)
     


