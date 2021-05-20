import warnings
warnings.filterwarnings("ignore")
import gym
import pybullet_envs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torch.multiprocessing as mp
import time
import random
import numpy as np
from collections import deque
from statistics import mean, stdev


ENV = gym.make("HalfCheetahBulletEnv-v0")
OBS_DIM = ENV.observation_space.shape[0]
ACT_DIM = ENV.action_space.shape[0]
ACT_LIMIT = ENV.action_space.high[0]
ENV.close()
GAMMA = 0.99

# SEED를 선택해서 학습하세요.
# SEED = {1,100,200,300,600}
# 선택한 SEED로 셋팅하세요.

# 본 template파일 자유롭게 수정 가능하며 score.py 와 문제없이 연동된다면 괜찮습니다.

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # TODO

    # Actor part
    def actor(self, x):
        
        return # TODO

    # Actor를 이용해 state를 받아서 action을 예측, 반환
    def get_action(self, x):

        return # TODO

    # Critic part
    def critic(self, x):
        
        return # TODO

    def learn(self, global_actor, state_lst, action_lst, q_val_lst, optimizer):
        """

            Computes advantages by subtracting a bseline(V(from critic)) from the estimated Q values
            추가로 해볼 수 있는 것 : advantage normalize
            Training a ActorCritic Agent refers to updating its actor using the given observations/actions
            and the calculated q_values/ advantages that come from the seen rewards

        """
        # TODO

def n_step_td(reward_lst, V):
    q_val_lst = []
    # TODO: n_step td return
    for t in #TODO:
        
    return q_val_lst

def reward_scaling(reward_lst):
    
    # TODO

    return reward_lst

def Worker(global_actor, num_episodes, n_steps):
    '''
    1. global_actor로부터 local_actor를 업데이트
    2. update the policy
       Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
    3. 추가로 해볼 수 있는 것 : reward scaling, + (n-step update, , advantage scaling, 
                                GAE(Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438))

    참고 논문 : Asynchronous Methods for Deep Reinforcement Learning(A3C), https://arxiv.org/abs/1602.01783).

    '''
    env = gym.make("HalfCheetahBulletEnv-v0")
    
    local_actor = ActorCritic()
    
    #TODO
    
    for episode in range(num_episodes):
        done = False
        s = env.reset()
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            
            # N-step rollout 
            for t in range(n_steps):
                # TODO
                # action = local_actor.get_action # TODO

                ########################## 이 부분은 수정하지 마세요 ###########################

                next_state, reward, done, _ = env.step(action)
                ########################## 이 부분은 수정하지 마세요 ###########################

                # TODO
                
                if done:
                    break

            # HINT : reward scaling part
            r_lst = reward_scaling(r_lst)

            # HINT : if done -> V = 0, else -> V = agent.critic(last state of N-step rollout trajectory)
            V = # TODO
            q_val_lst = n_step_td(r_lst, V)

            agent.learn(global_actor, s_lst, a_lst, q_val_lst, optimizer)

    env.close()
    print("Training process reached maximum episode.")
    
########################## 이 부분은 수정하지 마세요 ###########################
def Evaluate(global_actor, num_episodes):
    env = gym.make("HalfCheetahBulletEnv-v0")
    episode_rewards = deque(maxlen=100)
    start_time = time.time()
    epi_plot = []
    
    for episode in range(num_episodes):
        done = False
        state = env.reset()
        epi_rew = 0.0
        
        while not done:
            action = global_actor.get_action(torch.from_numpy(state).float())
            next_state, reward, done, _ = env.step(action)
            state = next_state
            epi_rew += reward
        
        episode_rewards.append(epi_rew)

        time.sleep(0.0005)

        if episode >= 100:
            mean_100_episode_reward = mean(episode_rewards)
            epi_plot.append(mean_100_episode_reward)
            if episode % 10 ==0:
                print("Episode: {}, avg score: {:.1f}".format(episode, mean_100_episode_reward))

            if mean_100_episode_reward >= 1000:
                print("Solved (3)!!!, Time : {:.2f}".format(time.time() - start_time))
                np.save("./multi_rs.npy", np.array(epi_plot))
                return
            time.sleep(1)

    env.close()
    print("Fail... Retry")
########################## 이 부분은 수정하지 마세요 ###########################

def run(num_episodes):
    n_steps = 32 #TODO up to you

    # Worker Multiprocessing, Evaluate them 
    global_actor = ActorCritic()

    # TODO
    # Worker를 만들어 학습하세요.
    # 하나의 worker는 Evaluate을 위해 사용하세요.


if __name__ =='__main__':
    run(2000)