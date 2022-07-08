# ========================================================================================
# Author: Yeo Jin Kim
# Date: 01/20/2022
# File: Septic treatment agent class
# ========================================================================================


import argparse
import datetime
import numpy as np
import random
import os
import pickle
import pandas as pd
from collections import deque        
        
        
class SepticAgent(object):
    config = []
    pid = 'VisitIdentifier'
    label = 'Shock'
    timeFeat = 'MinutesFromArrival'
    discountFeat = 'DynamicDiscount' 
    rewardFeat = 'reward'
    
    TDmode = False       # use time interval as input feature for time-aware state approximation
    
    date = ''
    actionNum = 4 
    
    actions = [i for i in range(4)] 
    Qfeat = ['Q'+str(i) for i in actions]
    
    numFeat= ['HeartRate', 'RespiratoryRate','PulseOx', 'SystolicBP', 'DiastolicBP', 'MAP', 'Temperature', 'Bands',
               'BUN', 'Lactate', 'Platelet', 'Creatinine', 'BiliRubin','WBC', 'FIO2']
    
    #nextNumFeat = [f + '_next' for f in numFeat]
    nextNumFeat = ['next_'+f for f in numFeat]
    stateFeat = numFeat[:] 
    #nextStateFeat = [f + '_next' for f in stateFeat]
    nextStateFeat = ['next_'+f for f in stateFeat]

    train_posvids = []
    train_negvids = [] 
    train_totvids = []
    test_posvids = []
    test_negvids = [] 
    test_totvids = []
    
    def __init__(self, args):
        self.DEBUG = args.debug         # True: Turn on Debug mode
        self.method = args.method  #TQN, TState, TDiscount, RQN
        self.DUELING = False
        self.DOUBLE = False
        
        self.keyword = args.k
        self.character = args.c
        self.multiViewNum = args.mt
        
        if 'lstm' in self.keyword:   
            self.func_approx = 'LSTM'
        else:
            self.func_approx = 'FC'
        
        # Prioritized Experience Replay --------------
        self.per_flag = False
        self.per_alpha = 0.6    # PER hyperparameter
        self.per_epsilon = 0.01 # PER hyperparameter
        self.beta_start = 0.4 # the lower, the more prioritized
        self.beta_increment = 0.001 #not used
        self.beta_period  = 1000
        # ---------------------------------------------
        self.reg_lambda = 5
        self.Q_clipping = False # for Q-value clipping 
        self.Q_THRESHOLD = 1000 # for Q-value clipping
        self.REWARD_THRESHOLD = 1000
        
        self.huber_loss_threshold = 0
        
        self.batch_size = 32
        self.targetFuture = 2880 # task time window (minutes)
        self.tau = 0.001 #Rate to update target network toward primary network

        self.LEARNING_RATE = False # Annealing learning rate (False: disabled / True: enabled)
        self.learnRate = args.lr    # 0.001 / init_value (αk+1 = 0.99αk)
        self.learnRateFactor = 0.99 
        self.learnRate_min = 0.0005
        self.learnRatePeriod = 2000 
       
        self.belief = float(args.b)
        self.targetFuture = float(args.tf)
        self.gamma = float(args.d) 
        
        self.hidden_size = int(args.hu)
        self.numSteps = int(args.t)
        self.gpuID = str(args.g)

        self.maxSeqLen = int(args.msl)
        self.trainMinTI = args.trminTI
        self.trainMaxTI = args.trmaxTI
        self.testMinTI = args.teminTI
        self.testMaxTI = args.temaxTI
        self.normTI = 60 * self.trainMaxTI
        self.UPPER_TI = args.upperTI # 0: disabled (set the upper bound of time interval for temporal abstraction)
        self.valGen = args.valGen
        
        if ('TQN' in self.method or 'TState' in self.method):
            self.state_input_size = len(self.numFeat) +1
        else:
            self.state_input_size = len(self.numFeat) 
            
        self.load_model = True 
        self.date =  str(datetime.datetime.now().strftime('%m%d%H')) 
        
        self.memory = deque(maxlen=int(225922)) # same to the training set size
        self.train_start = 10000
        
        self.save_results = True
        
        self.period_save = 100000
        self.period_eval = 10000
        self.saveResultPeriod = 200000
     
        self.cvFold = int(args.cvf)
        self.filename = '{}_{}_seq{}_ti{}_{}_b{}_g{}_h{}'.format(self.method, self.keyword, self.maxSeqLen, self.trainMinTI,
                                                self.trainMaxTI, int(self.belief*10), int(self.gamma*100), self.hidden_size)
        if self.DEBUG:
            self.filename = 'DEBUG_' + self.filename
        
        #self.resAll = pd.DataFrame(columns = ['epoch', 'shockRate'])  
        
        
        
    def remember(self, state, action, time_interval, reward, next_state, done):
        #experience = state, action, time_interval, reward, next_state, done
        self.memory.append((experience))
            

    def update_ti_queue(self, time_interval, t_state_queue):
        if 'TQN' in self.method or 'TState' in self.method:   
            t_state_queue[-1][-1] = time_interval/(self.trainMaxTI*60)
        return t_state_queue

    def makeTempAbstract(self, obs, time_interval, workMEM):

        if self.method == 'TQN' or self.method == 'TState': 
            obs = np.array(obs.tolist()+[time_interval/(self.trainMaxTI*60)])
            
        workMEM.append(obs)        
        
        if len(workMEM) > self.maxSeqLen:
            workMEM = workMEM[-self.maxSeqLen:]
        
        # ----------------------------------------------
        # Convert t_state to RNN data format
        X = np.array(workMEM)
        if len(workMEM) < self.maxSeqLen:
            X = np.pad(X, [(self.maxSeqLen-len(workMEM), 0), (0,0)],
                        mode='constant')        
    

        if 'FC' in self.func_approx: #concatenate 'seqDim' number of recent states 
            X = X.flatten()

        t_state_lstm = np.expand_dims(X, axis=0)

        return t_state_lstm, workMEM    

    # make tuples (o, a, Delta t, r, o', done)
    def make_tuples(self, df):
        obs = np.array(df[self.numFeat])
        actions = np.array(df.Action.tolist())
        
        if 'next_action' not in df.columns:
            df['next_action'] = df.groupby('VisitIdentifier').shift(-1).Action.fillna(0)
        next_actions = np.array(df.next_action.tolist()) 
        
        rewards = np.array(df[self.rewardFeat].tolist())
        
        if 'done' not in df.columns:
            df['done'] = 0
            df.loc[df.groupby('VisitIdentifier').tail(1).index, 'done'] = 1
        done_flags = np.array(df.done.tolist())
        time_intv = np.array(df.TD.tolist()) 
        return obs, actions, time_intv, rewards, next_actions, done_flags 

    
 
    # Generate the state inputs for the validation/test data 
    def make_MT_StateData(self, df, save_dir, keyword=''):
        
        zeros = np.array([0]*len(self.numFeat))
        statePool = []
        # Init (seqLen * maxTI) size of sequence memory --------------------------------
        len_seqMEM = self.maxSeqLen * self.testMaxTI
        seqMEM = deque(maxlen=int(len_seqMEM))
        for i in range(len_seqMEM-1):
            seqMEM.append((zeros,0))
        # ------------------------------------------------------------------------------         
        # Make training tuples 
        obsL, actionL, tiL, rewardL, next_actionL, doneL = self.make_tuples(df)
        
        # For each event
        for i in range(0, len(obsL)):
            
            seqMEM.append((obsL[i],tiL[i]))
            
            # Select time steps for TA 
            sampleNum = np.min([self.maxSeqLen -1, len(seqMEM)-1])
            timeSteps = random.sample(range(0, len(seqMEM)-1), sampleNum) # select previous time steps
            timeSteps.append(len(seqMEM) - 1) # put a current time step    
            timeSteps.sort()
            
            # Init the working memory for temporal abstraction
            workMEM = []

            # Select (seqLen-1) number of observations and a current observation to generate a current Obs
            for e in range(len(timeSteps)):  
                if e < len(timeSteps)-1: # for previous observations
                    accTI = 0
                    for a in range(timeSteps[e],timeSteps[e+1]):
                        accTI += seqMEM[a][1]
                        
                        # Don't abstract too sparse observations for test
                        if self.UPPER_TI and (accTI >= self.UPPER_TI): 
                            break
            
                else: # for current observation
                    accTI = seqMEM[-1][1]

                cur_state, workMEM = self.makeTempAbstract(seqMEM[timeSteps[e]][0], accTI, workMEM) # Put a current observation
            
            statePool.append(cur_state[0])
    
            if doneL[i] == 1: # init seqMEM when a new visit starts
                # Init (seqLen * maxTI) size of sequence memory -------------------------
                seqMEM = deque(maxlen=int(len_seqMEM))
                for i in range(len_seqMEM-1):
                    seqMEM.append((zeros,0))
                # -----------------------------------------------------------------------

        # save the data -----------------------------------------------------------------
        print("data: {}".format(np.shape(statePool)))
        if ('reg' in self.keyword) or (self.testMinTI == self.testMaxTI):
            valPath = "{}/{}_{}{}_reg{}".format(save_dir, self.method, self.func_approx, self.maxSeqLen, self.testMaxTI)
        else:
            valPath = "{}/M{}_{}{}_ti{}_{}".format(save_dir, self.method, self.func_approx, self.maxSeqLen, 
                                                         self.testMinTI, self.testMaxTI)
        if self.UPPER_TI:
            valPath += '_UPPER{}'.format(self.UPPER_TI)

        valPath += '_test{}.p'.format(keyword)
        
        #if not os.path.exists(valPath):
        with open(valPath, 'wb') as f:
            pickle.dump(statePool, f)
        print(" ! == Generate validataion data: {} == !".format(valPath))
        return statePool
        
    # Make a training batch
    def get_mt_batch(self):        
            
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        if self.func_approx == 'LSTM':
            #print("*** state_input_size: {}".format(np.size(minibatch[0][0])))
            state = np.zeros((self.batch_size, self.maxSeqLen, self.state_input_size))
            next_state = np.zeros((self.batch_size, self.maxSeqLen, self.state_input_size))
        else: # Dense
            state = np.zeros((self.batch_size, self.state_input_size * self.maxSeqLen))
            next_state = np.zeros((self.batch_size, self.state_input_size* self.maxSeqLen))

        action, next_action, ti, reward, tGammas, done = [], [], [], [], [], []
        
        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop

        for i in range(len(minibatch)):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            ti.append(minibatch[i][2])
            reward.append(minibatch[i][3])
            next_state[i] = minibatch[i][4]
            next_action.append(minibatch[i][5])
            done.append(minibatch[i][6])
            if 'TQN' in self.method or 'TDiscount' in self.method: # Temporal discount function 
                tGammas.append(self.belief**(ti[i]/self.targetFuture))

        return state, action, ti, reward, next_state, next_action, np.array(done), tGammas

    # Get the best action 
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    
def parsingPolicy(parser):
    parser.add_argument("-method", type=str, choices={'DQN','TState','TDiscount','TQN'})   # Method: TQN, Tstate, Tdiscount, RQN
    parser.add_argument("-k", type=str, default = 'lstm')   # {'lstm', 'dense'} keyword for models & results
                                                            # For SARSA: add 'sarsa' in keyword (e.g. -k = sarsa_dense)
    parser.add_argument("-c",  type=str, default= '')       # characteristics of model
    parser.add_argument("-cvf", type=int, default = 0)      # fold 
    parser.add_argument("-debug", type=int, default = 0)    # Debug mode
     
    parser.add_argument("-g", type=str, default = '0')       # GPU ID
    # Network hyperparameters
    parser.add_argument("-msl", type=int, default = 3)       # max sequence length for LSTM
    parser.add_argument("-hu", type=int, default = 128)      # hidden_size
    parser.add_argument("-t",  type=int, default = 200000)   # training iteration
    parser.add_argument("-lr",  type=float, default = 0.001) # learning rate
    
    # Time-aware hyperparameters
    parser.add_argument("-d",  type=float, default = 0.97)   # discount factor gamma
    parser.add_argument("-b",  type=float, default = 0.1)    # belief for TQN
    parser.add_argument("-tf", type=int, default = 2880)     # task time window (minutes)
    
    # Multi-temporal view 
    parser.add_argument("-mt",  type=int, default = 3)        # number of multi-view
    parser.add_argument("-trminTI",  type=int, default = 1)   # training min time interval (TI)
    parser.add_argument("-trmaxTI",  type=int, default = 3)   # training max TI
    parser.add_argument("-teminTI",  type=int, default = 1)   # test min TI
    parser.add_argument("-temaxTI",  type=int, default = 3)   # test max TI
    parser.add_argument("-upperTI",  type=int, default = 0)   # set the upper bound TI for temporal abstraction (0: disabled)
    
    parser.add_argument("-valGen",  type=int, default = 0)   # generate a valdiation data set with given options
    args = parser.parse_args()

    env = SepticAgent(args)
    if env.method=='TQN' or env.method=='TState':
        #print("* Add time intervals as a state input feature")
        env.stateFeat = env.numFeat[:]+['TD']
        env.TDmode=True
        
    env.nextStateFeat = ['next_'+s for s in env.stateFeat]

    # Able to add the following network learning options with keywords 
    # P: prioritized experience replay (Disable for Temporal Abstraction methods)
    # Doub: Double DQN  (Able for all the methods)
    # Duel: Dueling DQN (Able for all the methods)
    if 'P' in env.filename:
        env.per_flag = True
    if 'Doub' in env.filename or 'DD' in env.filename:
        env.DOUBLE = True
    if 'Duel' in env.filename or 'DD' in env.filename:
        env.DUELING = True    

    return env



