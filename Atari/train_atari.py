##
# TQN

import pandas as pd
import os 
import random

import time
import argparse

import gym
import numpy as np
import torch
from torch import nn
import datetime
from collections import deque

# from torch.utils.tensorboard import SummaryWriter

# from torch.utils.data import DataLoader

from envs import ATARI_ENVS
from atari_wrappers import make_atari, wrap_deepmind

from replay_buffers.uniform_tqn import UniformBuffer, DatasetBuffer
from networks.nature_cnn import NatureCNN
from networks.nature_cnn_fc3 import NatureCNN_FC3
from networks.tqn_cnn import TQN_CNN
from networks.tqn_cnn_duel import TQN_CNN_DUEL
from networks.tqn_cnn_fc3 import TQN_CNN_FC3

from agents.tqn import TQN
from agents.dqn import DQN

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

DEBUG = False

    # Warning !! 
    # CANNOT LOAD and CONTINUE to learn a policy from the existing policy model 
    # because the replay buffer was gone. 

def main():
    """Main function. It runs the different algorithms in all the environemnts.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Process inputs",
    )

    parser.add_argument(
        "--env", type=str, choices=ATARI_ENVS, default="MsPacmanNoFrameskip-v4"
    )
    # ----------------------
    parser.add_argument("--method", type=str, choices=["DQN","TDiscount", "TState", "TQN"],default="TQN")    
    parser.add_argument("--mode", type=int,  default=0) # TQN or TState
    parser.add_argument("--keyword", type=str, default='')
    # learning starts with a specific-frame policy 
    parser.add_argument("--load_frame", type=int, default=0) 
    parser.add_argument("--belief", type=float, default=0.5)
    parser.add_argument("--focusTimeWindow", type=float, default=69) #69=np.log(args.belief)/np.log(0.99), # without domain knoweldge
        
    parser.add_argument("--minTI", type=int, default=1)
    parser.add_argument("--maxTI", type=int, default=8)
    parser.add_argument("--cminTI", type=int, default=1)
    parser.add_argument("--cmaxTI", type=int, default=8)
    parser.add_argument("--maxTI_low", type=int, default=1)
    parser.add_argument("--maxTI_high", type=int, default=32)
    
    parser.add_argument("--showPeriodEval", type=int, default=5000)

    # ----------------------
#     parser.add_argument("--prioritized", type=int, default=0)
    parser.add_argument("--double_q", type=int, default=0)
    parser.add_argument("--dueling", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=int(10e6))
    parser.add_argument("--clip_rewards", type=int, default=1) 
    parser.add_argument("--buffer_size", type=int, default=int(1e5))
    
    args = parser.parse_args()
    

    print("Arguments received:")
    print(args)
    if "TQN" in args.method or "TState" in args.method: # (TI as state_input)
        agent_class = TQN
    else: # DQN or TDiscount (no TI as state_input)
        agent_class = DQN
        if args.double_q:
            agent_class = DoubleDQN
    
    
    run_env(
        method = args.method,
        mode=args.mode, 
        keyword=args.keyword,
        load_frame = args.load_frame,
        minTI=args.minTI, maxTI=args.maxTI,
        cminTI = args.cminTI, cmaxTI = args.cmaxTI,
        maxTI_low = args.maxTI_low, maxTI_high = args.maxTI_high,
        
        belief = args.belief,
        focusTimeWindow = args.focusTimeWindow,
        
        env_name=args.env,
        agent_class=agent_class,
        
        buffer_size=args.buffer_size,
        dueling=args.dueling,
#         prioritized=args.prioritized,
        clip_rewards=args.clip_rewards,
        num_steps=args.num_steps,
        showPeriodEval = args.showPeriodEval
    )


def run_env(
    method, mode, keyword, load_frame,
    minTI, maxTI, cminTI, cmaxTI, maxTI_low, maxTI_high, 
    belief, focusTimeWindow,
    env_name, agent_class,
    
    buffer_size=int(1e5),
    dueling=False,
    clip_rewards=True,
    normalize_obs=True,
    num_steps= int(10e6),
    batch_size=32,
    initial_exploration=1.0,
    final_exploration=0.01,
    exploration_steps=int(2e6),
    learning_starts= int(1e4),
    train_freq=1,
    target_update_freq=int(1e4),
    save_ckpt_freq= int(10e6),
    showPeriodEval = 1000
): 
    """Runs an agent in a single environment to evaluate its performance.
    Args:
        env_name: str, name of a gym environment.
        agent_class: class object, one of the agents in the agent directory.
        clip_rewards: bool, whether to clip the rewards to {-1, 0, 1} or not.
    """
    
    random.seed(mode) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if env_name in ATARI_ENVS:
        env = make_atari(env_name)
        # wrap: scale=False --> no normalized observation /255
        env = wrap_deepmind(env, frame_stack=True, scale=False, clip_rewards=False)
    else:
        env = gym.make(env_name)

    if isinstance(env.action_space, gym.spaces.Discrete):
        num_state_feats = env.observation_space.shape
        num_actions = env.action_space.n

        replay_buffer = UniformBuffer(size=buffer_size, device=device)
        #----------------------------------
        if 'TQN' in method or 'TState' in method:
            TStateMode = 1
            if dueling:
                main_network = TQN_CNN_DUEL(num_actions, hsize=512).to(device)
                target_network = TQN_CNN_DUEL(num_actions, hsize=512).to(device)
            elif 'fc3' in keyword:
                main_network = TQN_CNN_FC3(num_actions, hsize=512).to(device)
                target_network = TQN_CNN_FC3(num_actions, hsize=512).to(device)
            else:
                main_network = TQN_CNN(num_actions).to(device)
                target_network = TQN_CNN(num_actions).to(device)
        else: # DQN or TDiscount
            TStateMode = 0
            if 'fc3' in keyword:
                main_network = NatureCNN_FC3(num_actions, hsize=512).to(device)
                target_network = NatureCNN_FC3(num_actions, hsize=512).to(device)            
            else:
                main_network = NatureCNN(num_actions).to(device)
                target_network = NatureCNN(num_actions).to(device)
        
        if "TQN" in method or "TDiscount" in method:
            TDiscountMode = 1
        else:
            TDiscountMode = 0
        
        #----------------------------------
        # Load the existing policy model
        save_path = "./saved_models/TI_fixed/{}/{}/ti{}_{}/b{}/".format(env_name[:-14], method,
             minTI, maxTI, int(belief*10))
        policy_file = save_path + "{}{}-ti{}-{}-{}-{}M.pt".format(method, mode, cminTI, cmaxTI, 
                 keyword, int(load_frame/1000000)) 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("create policy path: ", save_path)
            
                    
        target_network.load_state_dict(main_network.state_dict())
        target_network.eval()
        
        agent = agent_class(
            policy_file,
            num_actions=num_actions,
            main_nn=main_network,
            target_nn=target_network,
            device=device,
            discount=belief**((minTI + maxTI)/2/focusTimeWindow) # added 061021
        )

    # ------------------------------
    # Init logging
    outPath = 'out/TI_tuning/{}/{}/ti{}_{}/b{}/'.format(env_name, method, minTI, maxTI, int(belief*10))    
    logFileName = outPath + 'log_{}{}_{}.csv'.format(method, mode, keyword)
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    # Carefully use: LOAD and CONTINUE to learn a policy from the existing policy model 
    # but the replay buffer was gone. 
    if load_frame > 0: # learning starts with a specific-frame policy 
        logdf = pd.read_csv(logFileName, header=0)
        episode = int(logdf.episode.tolist()[-1])+1
        cur_frame = load_frame+1
        epsilon = logdf.epsilon.tolist()[-1]
        print("Loaded the existing policy: ", policy_file)
        print("Start from episode {} (frames: {}) - epsilon: {}".format(episode, cur_frame, epsilon))
    else: # learning starts from scratch
        logdf = pd.DataFrame(columns = ['episode', 'step', 'epsilon', 'minTI', 'maxTI', 'avgEpLen', 'returns',\
                                'returns_std', 'run_time', 'avgTimeInterval'])
        episode, cur_frame = 0, 0
        epsilon = initial_exploration
        
    # ------------------------------    
    returns = deque(maxlen=int(100))

    startEp = time.time()
    startTrainTime = startEp
    
    epLenList = deque(maxlen=int(100))
    tiList = deque(maxlen=int(100))# to calculate avg. time interval every 100 episodes
    preAvgEpLen = 1000000    
    preAvgReturn = 0  
    maxReturn = 0
    # init discount with a constant discount (for DQN/ TState)
    agent.discount = belief**((minTI + maxTI)/2/focusTimeWindow) #=0.99  
    
     
    print_env_info(env, env_name, minTI, maxTI, method, mode, main_network, focusTimeWindow, 
           keyword)
    
    
    # Start learning!
    while cur_frame <= num_steps:
        state = env.reset()
        done, ep_rew, clip_ep_rew = False, 0, 0

        episode_len = 0
        while not done:
            
            state_np = np.expand_dims(state, axis=0).transpose(0, 3, 2, 1)
            state_in = torch.from_numpy(state_np).to(device, non_blocking=True)
            # when the system does not support CUDA, remove cuda below: torch.FloatTensor
            state_in = torch.div(state_in.type(torch.cuda.FloatTensor), 255.0) #if normalize_obs
            
            # Sample action from policy and take that action in the env. with time interval
            # Generate a random time interval to the next state
            time_interval = random.randint(minTI, maxTI)
            tiList.append(time_interval)
            
            if TStateMode:
                normTI = np.array([time_interval/maxTI])  
                normTI = np.expand_dims(normTI, axis=1)    
                # when the system does not support CUDA, remove cuda below: torch.FloatTensor
                normTI_in = torch.from_numpy(normTI).to(device).type(torch.cuda.FloatTensor)
                action = agent.take_exploration_action(state_in, normTI_in, env, epsilon)                    
            else:
                action = agent.take_exploration_action(state_in, env, epsilon)                    
            
            cum_reward = 0 
            cum_reward_clip = 0
            for actual_TI in range(1, time_interval+1):   
                next_state, rew, done, info = env.step(action)
                cum_reward_clip += np.sign(rew)
                cum_reward += rew
                episode_len += 1
                if done:
                    break

            if (episode_len > 10000) and (cum_reward == 0): 
                break # too long steps without reward means stuck, so break the episode
                
            time_interval = np.array([np.float32(actual_TI)])  # update TI
            actualNormTI = time_interval/maxTI

            if TDiscountMode: # Temporal discount 
                agent.discount = belief**((actual_TI)/focusTimeWindow)

            # until 100K, keep the init maxTI, then control maxTI periodically with the following rules:
            # if the avg. episode length decreases, then reduce maxTI - 1, (the game has a shorter action timing window)
            # if the avg. ep reward increases, then increase maxTI + 1 (increase performance on higher abstraction)  
                        
            if clip_rewards: 
                reward_in = cum_reward_clip
            else:
                reward_in = cum_reward
                
            replay_buffer.add(state, actualNormTI, action, reward_in, next_state, done, agent.discount)
             

            if cur_frame % 1e6 == 0:
                print("cur_frame: {}M ({})".format(int(cur_frame/1e6), str(datetime.datetime.now().strftime('%m/%d %H:%M'))))
            
            cur_frame += 1  # = update 
            clip_ep_rew += cum_reward_clip
            ep_rew += cum_reward
            
            
            # Move to the next state -------------------------------------       
            state = next_state
            
            # train_freq=1: training every step
            if cur_frame > learning_starts: # and cur_frame % train_freq == 0:
                try:
                    st, normTimeInv, act, rew, next_st, d, tdiscounts = replay_buffer.sample(batch_size)

                except StopIteration:
                    iterator = iter(dataloader)
                    continue
                # when the system does not support CUDA, remove cuda below: torch.FloatTensor    
                st = torch.div(st.type(torch.cuda.FloatTensor), 255.0).to(device)
                next_st = torch.div(next_st.type(torch.cuda.FloatTensor), 255.0).to(device)
                # -------------- 
                if TStateMode:
                    loss_tuple = agent.train_step(st, normTimeInv, act, rew, next_st, d, tdiscounts)
                else: 
                    loss_tuple = agent.train_step(st, act, rew, next_st, d, tdiscounts)
                # --------------

            # Episode done -----------------------------
            epLenList.append(episode_len)
            
            # Update value of the exploration value epsilon.
            epsilon = decay_epsilon(
                epsilon, cur_frame, initial_exploration, final_exploration, exploration_steps)

            if cur_frame % target_update_freq == 0 and cur_frame > learning_starts:
                # Copy weights from main to target network.
                agent.target_nn.load_state_dict(agent.main_nn.state_dict())

            if cur_frame % save_ckpt_freq == 0 and cur_frame > learning_starts:
                policy_file = save_path + "{}{}-ti{}-{}-{}-{}M.pt".format(method, mode, minTI, maxTI, 
                 keyword, int(cur_frame/1000000)) 

                agent.save_checkpoint(policy_file)

        episode += 1
        returns.append(ep_rew)

        # Best model save --- 
        if (cur_frame > 5e6) and (np.mean(returns) > maxReturn):
            maxReturn = np.mean(returns)
            policy_file = save_path + "{}{}-ti{}-{}-{}-Best.pt".format(method, mode, minTI,
                                      maxTI, keyword) 
            agent.save_checkpoint(policy_file)
            print("* Best score: {} - {} F".format(maxReturn, cur_frame))
        
        # log every 100 episode    
        if episode % 100 == 0:        
            curAvgReturn = np.mean(returns)
            curAvgEpLen = np.mean(epLenList)
              
                     
            preAvgEpLen = curAvgEpLen
            preAvgReturn = curAvgReturn

            # --------------------------------
            # Log
            endEp = time.time()
            runTime = np.round((endEp-startEp),0)

            logdf.loc[len(logdf)] = [episode, cur_frame, epsilon, minTI, maxTI, curAvgEpLen,\
                                np.round(np.mean(returns),1), np.round(np.std(returns),1), runTime, np.mean(tiList)]

            logdf.to_csv(logFileName, index=False)
            # --------------------------------                   
            if episode % showPeriodEval == 0:                      
                print_result(method, mode, minTI, maxTI, env_name, epsilon, episode, cur_frame, returns, runTime, curAvgEpLen)
            
            startEp = time.time()
    
    logdf.loc[len(logdf)] = [episode, cur_frame, epsilon, minTI, maxTI, curAvgEpLen,\
                             np.mean(returns), np.std(returns), runTime, np.mean(tiList)]
    logdf.to_csv(logFileName, index=False)            
    print_result(method, mode, minTI, maxTI, env_name, epsilon, episode, cur_frame, returns, runTime, curAvgEpLen)    
    print("\n * Total Training Time: {:.1f} hours".format((time.time()-startTrainTime)/3600))
    print(" * End experiment: {}".format(str(datetime.datetime.now().strftime('%m/%d %H:%M'))))



def decay_epsilon(epsilon, step, initial_exp, final_exp, exp_steps):
    if step < exp_steps:
        epsilon -= (initial_exp - final_exp) / float(exp_steps)
    return epsilon


def print_result(method, mode, minTI, maxTI, env_name, epsilon, episode, step, returns, runTime, curAvgEpLen):
    print("{}{} [{},{}]  * {:>10s} ".format(method, mode, minTI, maxTI, env_name[:-14]), end='| ')
    print("{}k ep / {}k step (expl: {}%) - Last 100 ({:.1f} min, avgEpLen: {:.1f}) - return {:.1f} ".format(
                int(episode/1000), int(step/1000), int(epsilon * 100), runTime/60, curAvgEpLen, np.mean(returns)))
            


def print_env_info(env, env_name, minTI, maxTI, method, mode, network, focusTimeWindow, keyword):
    print("======================================")
    print("Environemnt: {} ({})".format(env_name[:-14], keyword))
    print("Agent: {} ({}) / TI:[{}, {}]".format(method, mode, minTI, maxTI))
    print("Network: {} (focusTimeWindow: {})".format(type(network).__name__, focusTimeWindow))
    print("Observation: {} / Action: {}".format(env.observation_space.shape, env.action_space))
    print("======================================")


if __name__ == "__main__":
    main()
