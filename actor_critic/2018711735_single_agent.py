import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, seed=100, std=0.0):
        super(ActorCritic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.critic = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)

        self.apply(init_weights)

    def forward(self, x):
        self.value = self.critic(x)
        self.mu = self.actor(x)
        self.std = self.log_std.exp().expand_as(self.mu)  #
        self.dist = Normal(self.mu, self.std)
        return self.dist, self.value

    # Actor를 이용해 state를 받아서 action을 예측, 반환
    def get_action(self, x):
        return  # TODO

    def learn(self, state_lst, logprob_lst, q_val_lst, entropy, optimizer):
        """

            Computes advantages by subtracting a bseline(V(from critic)) from the estimated Q values
            추가로 해볼 수 있는 것 : advantage normalize
            Training a ActorCritic Agent refers to updating its actor using the given observations/actions
            and the calculated q_values/ advantages that come from the seen rewards

        """
        # TODO

        self.log_probs = torch.cat(logprob_lst)
        self.returns = torch.cat(q_val_lst).detach()
        self.values = torch.cat(state_lst)

        self.advantage = self.returns - self.values

        self.actor_loss = -(self.log_probs * self.advantage.detach()).mean()
        self.critic_loss = self.advantage.pow(2).mean()

        self.loss = self.actor_loss + 0.5 * self.critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        self.loss.backward()
        optimizer.step()

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def n_step_td(reward_lst, V):
    q_val_lst = []
    # TODO: n_step td return
    R = V
    for step in reversed(range(len(reward_lst))):  # TODO:
        R = reward_lst[step] + GAMMA * R
        q_val_lst.insert(0, R)
    return q_val_lst


# episode_rewards, epi_plot, epi_reward는 나중에 plot할 때 필요한 정보이기 때문에 수정하지 말아주세요.
def Worker(num_episodes, n_steps):
    env = gym.make("HalfCheetahBulletEnv-v0")

    agent = ActorCritic(OBS_DIM, ACT_DIM).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)

    ##########################이 부분은 수정하지 마세요###########################
    episode_rewards = deque(maxlen=100)
    start_time = time.time()
    epi_plot = []
    finish = False
    ##########################이 부분은 수정하지 마세요###########################

    # TODO
    for episode in range(num_episodes):
        done = False
        state = env.reset()
        epi_reward = 0.

        while not done:
            s_lst, a_lst, r_lst = [], [], []
            entropy = 0  #
            masks = []

            # N-step rollout
            for t in range(n_steps):
                # TODO
                # action = agent.get_action # TODO
                # while env takes in/out in numpy, nn.module does in tensor, convert!
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                dist, value = agent(state)

                action = dist.sample()
                ##########################이 부분은 수정하지 마세요###########################
                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                epi_reward += reward
                ##########################이 부분은 수정하지 마세요###########################

                # TODO
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                a_lst.append(log_prob)
                s_lst.append(value)

                # scalar reward, done to tensor
                r_lst.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
                # masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))

                if done:
                    break

                state = next_state

            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            _, value = agent(next_state)
            # HINT : if done -> V = 0, else -> V = agent.critic(last state of N-step rollout trajectory)
            V = 0 if done else value  # TODO

            # q_val_lst = compute_gae(V, r_lst, masks, s_lst)
            q_val_lst = n_step_td(r_lst, V)

            agent.learn(s_lst, a_lst, q_val_lst, entropy, optimizer)

        ###################################이 부분은 수정하지 마세요###################################

        episode_rewards.append(epi_reward)

        if episode >= 100:
            mean_100_episode_reward = mean(episode_rewards)
            epi_plot.append(mean_100_episode_reward)
            if episode % 10 == 0:
                print("Episode: {}, avg score: {:.1f}".format(episode, mean_100_episode_reward))

            if mean_100_episode_reward >= 500:
                finish = True
                print("Solved (1)!!!, Time : {:.2f}".format(time.time() - start_time))
                np.save("./single.npy", np.array(epi_plot))
                return
    env.close()
    print("Fail... Retry")


def run(num_episodes):
    n_steps = 5  # TODO up to you

    Worker(num_episodes, n_steps)


if __name__ == '__main__':
    run(2000)
###################################이 부분은 수정하지 마세요###################################
