#from agent import Agent
from sarsa_agent import Agent
from monitor import interact
import gym
import numpy as np

from bayes_opt import BayesianOptimization

env = gym.make('Taxi-v2')
#agent = Agent(epsilon=0.1, alpha=0.1, gamma=0.9)
#avg_rewards, best_avg_reward = interact(env, agent)

def interact_wrapper(epsilon, alpha, gamma):
    agent = Agent(epsilon=epsilon, alpha=alpha, gamma=gamma)
    avg_rewards, best_avg_reward = interact(env, agent)
    return best_avg_reward

pbounds = {'epsilon': (0.01, 0.995), 'alpha': (0.1, 0.5), 'gamma': (0.5, 1.0)}

optimizer = BayesianOptimization(
    f=interact_wrapper,
    pbounds=pbounds,
    random_state=47
)

optimizer.probe(
    params={'epsilon': 0.1, 'alpha': 0.1, 'gamma': 0.9},
    lazy=True,
)

optimizer.maximize(
    init_points=3,
    n_iter=25
)