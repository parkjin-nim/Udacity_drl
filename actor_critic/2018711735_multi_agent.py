import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#from multiprocessing_env import SubprocVecEnv
from multiprocessing import Process, Pipe

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
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


# 본 template파일 자유롭게 수정 가능하며 score.py 와 문제없이 연동된다면 괜찮습니다.
### parallelism ###
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class VecEnv(object):
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        pass
    def step_async(self, actions):
        pass
    def step_wait(self):
        pass
    def close(self):
        pass
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs

### model ###
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
        self.std = self.log_std.exp().expand_as(self.mu)
        self.dist = Normal(self.mu, self.std)
        return self.dist, self.value


    def learn(self, state_lst, logprob_lst, q_val_lst, entropy, optimizer):
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

def n_step_td(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def Worker(envs, num_envs, num_episodes, n_steps):
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
    #env = gym.make("HalfCheetahBulletEnv-v0")
    state_size = envs.observation_space.shape[0]
    action_size = envs.action_space.shape[0]
    high_limit = envs.action_space.high[0]
    print("input space:", state_size)
    print("input space:", action_size)
    print("high_limit:", high_limit)

    agent = ActorCritic(state_size, action_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)

    ##########################이 부분은 수정하지 마세요###########################
    episode_rewards = deque(maxlen=100)
    start_time = time.time()
    epi_plot = []
    finish = False
    ##########################이 부분은 수정하지 마세요###########################

    for episode in range(num_episodes):

        states = envs.reset()
        scores = np.zeros(num_envs)  # initialize the score (for each agent)
        dones = np.zeros(num_envs, dtype=bool)
        while not np.any(dones):

            log_probs = []
            values = []
            nstep_rewards = []
            masks = []
            entropy = 0

            for step in range(n_steps):

                dist, value = agent(torch.FloatTensor(states).to(device))
                actions = dist.sample()

                next_states, rewards, dones, _ = envs.step(actions.cpu().numpy())
                scores += rewards
                states = next_states

                log_prob = dist.log_prob(actions)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                nstep_rewards.append(torch.FloatTensor(rewards).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - dones).unsqueeze(1).to(device))

                if np.any(dones):  # exit loop if episode finished
                    break

            _, next_values = agent(torch.FloatTensor(next_states).to(device))
            next_values = torch.zeros(num_envs).unsqueeze(1) if np.any(dones) else next_values

            returns = n_step_td(next_values, nstep_rewards, masks)

            agent.learn(values, log_probs, returns, entropy, optimizer)

        episode_rewards.append(scores.mean())
        #print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(episode_rewards)), end="")

        if episode >= 100:
            mean_100_episode_reward = mean(episode_rewards)
            epi_plot.append(mean_100_episode_reward)
            if episode % 10 ==0:
                print("Episode: {}, avg score: {:.1f}".format(episode, mean_100_episode_reward))

            if mean_100_episode_reward >= 800:
                print("Solved (2)!!!, Time : {:.2f}".format(time.time() - start_time))
                np.save("./multi.npy", np.array(epi_plot))
                return

    envs.close()
    print("Fail... Retry")
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

            if mean_100_episode_reward >= 800:
                print("Solved (2)!!!, Time : {:.2f}".format(time.time() - start_time))
                np.save("./multi.npy", np.array(epi_plot))
                return
            time.sleep(1)
    env.close()
    print("Fail... Retry")
########################## 이 부분은 수정하지 마세요 ###########################  


def run(num_episodes):
    n_steps = 5 #TODO up to you

    env_name = "HalfCheetahBulletEnv-v0"

    # TODO
    # Worker를 만들어 학습하세요.
    # 하나의 worker는 Evaluate을 위해 사용하세요.
    num_envs = 12
    def make_env():
        def _thunk():
            env = gym.make(env_name)
            return env

        return _thunk

    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    Worker(envs, num_envs, num_episodes, n_steps)

if __name__ =='__main__':
    
    run(2000)

