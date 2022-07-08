#!/usr/bin/env python
# coding: utf-8
"""
# ========================================================================================
# Author: Yeo Jin Kim
# Date: 01/20/2021
# File: Main function for Septic treament and Septic shock prevention,
#              using Temporal Abstraction with Time-aware deep Q-networks
# ========================================================================================
# * Development environment: 
#   - Linux: 3.10.0
#   - Main packages: Python 3.6.9, Tensorflow 1.3.0, Keras 2.0.8, pandas 0.25.3, numpy 1.16.4
# ========================================================================================
# * Offline training data: The EHRs cannot be shared
# ========================================================================================
# * Excution for each method:
# DQN: $ python cv_main_sepsis.py -method=DQN -func={Dense, LSTM} -trainMinTI=1 -trainMaxTI=1 -testMinTI=1 -testMaxTI=8
# TQN: $ python cv_main_sepsis.py -method=TQN -func={Dense, LSTM} -trainMinTI=1 -trainMaxTI=1 -testMinTI=1 -testMaxTI=8
# MTA-DQN: $ python cv_main_sepsis.py -method=TQN -func={Dense, LSTM} -trainMinTI=3 -trainMaxTI=5 -testMinTI=3 -testMaxTI=5
# MTA-TQN: $ python cv_main_sepsis.py -method=TQN -func={Dense, LSTM} -trainMinTI=3 -trainMaxTI=5 -testMinTI=3 -testMaxTI=5
#
# more options:  -b 0.1 -tf 2880 -d 0.97 -hu 128 -t 200000 -msl 5 -g 3 -cvf 0 -g 0
#   - b: belief for the temporal discount function
#   - tf: task time window for the temporal discount function
#   - d : constant discount
#   - hu: number of hidden units for deep function approximation
#   - t: max training update
#   - msl: max sequence length for LSTM/dense
#   - g: GPU ID
#   - cvf: fold id for a different random seed
# ========================================================================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import pandas as pd
import argparse
import datetime
import time
import pickle
import random

import SepticAgent as agent
import TQN as tq


def setExponentialDiscount(env, df, belief, focusTimeWindow, avgTimeInterval, outfile):
    if 'TD' not in df.columns:
        df.loc[:, 'TD'] = (df.groupby(env.pid).shift(-1)[env.timeFeat]-df[env.timeFeat]).fillna(0).values.tolist()  
    print("static gamma: {}".format(env.gamma))
    
    df.loc[:, env.discountFeat] = (env.belief**(df['TD']/focusTimeWindow)).values.tolist()
    
    print("mean TDD (before set fillna(0)): {:.4f}".format(df[env.discountFeat].mean()), end=' ')
    #df[decayfeat] *= df.groupby(pid).shift(1)[decayfeat].fillna(1) # fillna(1): keep tgamma for the first state
    df.loc[:, env.discountFeat] = df[env.discountFeat].fillna(0).tolist() # 0 for the terminal state (at the end of trajectory)
    print(" (after): {:.4f}".format(df[env.discountFeat].mean()))
    if outfile != '':
        df.to_csv(outfile, index=False)
    return df

# Set flag for splitter of numerical feature
def setData(env, df, testdf):
    
    avgTD = df[df.TD!=0].TD.mean()
    env.gamma = (env.belief)**(avgTD/env.targetFuture)
    print("belief: {}, targetFuture: {} min, avgTD: {:.4f}, discount: {}".format(env.belief, env.targetFuture, avgTD, env.gamma))


    # Temporal discount: calculate the corresponding discount with belief and concerned time window (targetFuture)
    if 'TQN' in env.method or 'TDiscount' in env.method:
        avgTimeItv = df[df.TD!=0].TD.mean()
        df = setExponentialDiscount(env, df, env.belief, env.targetFuture, avgTimeItv, outfile='')
        testdf = setExponentialDiscount(env, testdf, env.belief, env.targetFuture, avgTimeItv, outfile='')
        
    testAll = tq.process_eval_batch(env, testdf)
    print("testAll: {}".format(np.shape(testAll[0])))
    
    return df, testdf, testAll


def setDEBUG(df, val_df, env):    
    print("unique actions - train: {}, val: {}".format(df.Action.unique(), val_df.Action.unique()))
    df = df[df[env.pid].isin(df[env.pid].unique()[:100])]
    
    env.numSteps = 400
    env.period_save = 100000
    env.period_eval = 100
    env.kfold = 2
    return df, val_df, env


#Make paths for our model and results to be saved in.
def createResultPaths(save_dir, date):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    if not os.path.exists(save_dir+"results"):
        os.mkdir(save_dir+"results")           
    print(save_dir)
    
    
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

    # next states
    df[env.nextStateFeat] = df.groupby(env.pid).shift(-1).fillna(0)[env.stateFeat]
    
    # state-action Qs
    for f in env.Qfeat:
        df[f] = np.nan
        
    # Temporal Difference
    df.loc[:, 'TD'] = (df.groupby(env.pid)[env.timeFeat].shift(-1) - df[env.timeFeat]).fillna(0).tolist()
    
    otherFeat = env.numFeat+env.nextNumFeat+env.Qfeat+['Shock','Action', 'reward', 'next_action', 'done'] 
    usecols = ['VisitIdentifier', 'MinutesFromArrival','TD']+otherFeat
    if 'TQN' in env.method or 'TState' in env.method: 
        usecols += ['next_TD']

    df = df[usecols]
    return df

        

if __name__ == '__main__':
    print("=============\nStart experiment: {}".format(str(datetime.datetime.now().strftime('%m/%d %H:%M'))))
    simMaxSeqLen = 5
    parser = argparse.ArgumentParser()
    env = agent.parsingPolicy(parser)
    hdf = tq.saveHyperParameters(env) 
    
    # GPU setup
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = env.gpuID  # GPU-ID "0" or "0, 1" for multiple
    env.config = tf.ConfigProto()
    #env.config.gpu_options.allow_growth = True 
    env.config.gpu_options.per_process_gpu_memory_fraction = 0.1
  
    session = tf.Session(config=env.config) 
    startTime = time.time()

    
    cvID = int(env.cvFold/5)  # 5-cross validation : 1-fold : 5 time repetitions = total 25 repetitions per method
    
    print("-------------------------------------------------------------")
    #print(" Method: {} - Dueling: {} Double: {} PER: {}".format(env.method, env.DUELING, env.DOUBLE, env.per_flag))
    print(" Method: {} - cvID: {}".format(env.method, cvID))
    print("* Soft target update: {} / reward threshold: {}".format(env.tau, env.REWARD_THRESHOLD))
    if env.LEARNING_RATE:
        print("* learning rate - init: {} (annealing: {}, min: {}, factor: {}, period: {})".format(
              env.learnRate, env.LEARNING_RATE, env.learnRate_min, env.learnRateFactor, env.learnRatePeriod))
    else:
        print("* learning rate - init: {} (annealing: {})".format(env.learnRate, env.LEARNING_RATE))
    print("-------------------------------------------------------------")      

    
    # load data -----------------------------------------------------------   
    dfile = 'data/2_mayo_TD2m_SARS_all_051621.csv' # "data/1_mayo_TD2m_all_RLfeat_051521.csv"
    adf = initData_env(dfile, env)
    allposvids = adf[adf.Shock==1].VisitIdentifier.unique().tolist()
    allvids = adf.VisitIdentifier.unique().tolist()
    allnegvids = [v for v in allvids if v not in allposvids]
    print("all-pos:{} / neg: {} / all {}".format(len(allposvids), len(allnegvids), len(allvids)))
    
    cvPosNum = int(len(allposvids)/5)  # 441
    testvids = allposvids[cvPosNum*(cvID) : cvPosNum*(cvID+1)] + allnegvids[cvPosNum*(cvID) : cvPosNum*(cvID+1)]
    trainvids = [v for v in allvids if v not in testvids]
    
    df = adf[adf.VisitIdentifier.isin(trainvids)].copy(deep=True)
    # For test, do not use SARS data
    testdf = initData_env("../Sepsis/data/h48/2_mayo_TD2m_48h_RwdNorm_all_051521.csv", env)    
    testdf = testdf[testdf.VisitIdentifier.isin(testvids)].copy(deep=True)
    testPosNum = len(testdf[testdf.Shock==1].VisitIdentifier.unique())
    print("Double check - test: pos:{}/neg:{}".format(testPosNum, len(testdf.VisitIdentifier.unique())-testPosNum))
    del adf    

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
        
    testPath = 'data/ti{}/{}_test{}.p'.format(env.testMaxTI, modelName, 'cv{}'.format(cvID))
    if os.path.exists(testPath):
        with open (testPath, 'rb') as fp:
            env.testX = np.array(pickle.load(fp))
    else: # Load the test data and generate the TA data
        newTestPath = 'data/ti{}/'.format(env.testMaxTI)
        if not os.path.exists(newTestPath):
            os.makedirs(newTestPath)
        env.testX = env.make_MT_StateData(testdf, save_dir = newTestPath, keyword='cv{}'.format(cvID))    
    print("testX : {}".format(np.shape(env.testX)))    
        
    if env.DEBUG:
        df, testdf, env = setDEBUG(df, testdf, env)  
  
    df = tq.initPER(df, env)     
    df, testdf, testAll = setData(env, df, testdf)
    
    testdf.drop(columns = env.numFeat+env.nextNumFeat+['done'], inplace=True)
    df.drop(columns = ['prob', 'imp_weight'],inplace=True)
    #print(testdf.columns)
    
    runTime = []
    for i in range(env.cvFold,env.cvFold+1):
        print(" ******** ", i, " *********") 
        startTime = time.time()
        env.fold = i
        random.seed(env.fold+1) 
        
        env.save_dir = 'res_sepsis/'+env.date+'/'+str(env.fold)+'/'+env.filename+'/'
        createResultPaths(env.save_dir, env.date)
        
        hdf.to_csv(env.save_dir+'hyper_parameters.csv', index=False)
        print("Length: train({}), test({})".format(len(df), len(testdf))) 
        
        _= tq.mt_rl_learning(tf, env, df, testdf, testAll) # policy training 
               
        curRunTime = (time.time()-startTime)/60
        runTime.append(curRunTime)
        print("Learning Time: {:.2f} min".format(curRunTime))
        hdf.loc[len(hdf)] = ['learning_time', curRunTime]
        hdf.to_csv(env.save_dir+'hyper_parameters.csv', index=False)
        
    print("Learning Time: {}".format(runTime))            
    print("Avg. Learning Time: {:.2f} min".format(np.mean(runTime)))     
    print("End experiment: {}".format(str(datetime.datetime.now().strftime('%m/%d %H:%M'))))

