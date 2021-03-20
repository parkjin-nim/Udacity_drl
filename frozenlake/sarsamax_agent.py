import gym
import numpy as np
from collections import deque
import sys

class Agent:
    def __init__(self, Q, env, mode, eps=1.0, alpha=0.01, gamma=1.0):
        self.env = env
        self.Q = Q

        self.eps_min = 0.00001
        self.eps = eps
        self.gamma = gamma

        self.shape = int(np.sqrt(self.env.nS))
        if gym.envs.registration.registry.env_specs['FrozenLake-v3']._kwargs['is_slippery'] and self.shape == 4:
            self.num_episodes, self.alpha = (100000, 0.01)
        elif gym.envs.registration.registry.env_specs['FrozenLake-v3']._kwargs['is_slippery'] and self.shape == 8:
            self.num_episodes, self.alpha = (100000, 0.05)
        else:
            self.num_episodes, self.alpha = 20000, alpha


    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def exp_decay(self, i_episode):
        return max(self.eps ** i_episode, self.eps_min)

    def epsilon_greedy_probs(self, Q_s, i_episode, eps=None):

        epsilon = self.exp_decay(i_episode)
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.env.nA) * epsilon / self.env.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.env.nA)
        return policy_s

    def select_action(self, Q_s):
        best_a = np.argmax(self.Q[Q_s])
        return best_a

    def learn(self, plot_every=100):
        tmp_scores = deque(maxlen=plot_every)
        scores = deque(maxlen=self.num_episodes)

        for i_episode in range(1, self.num_episodes + 1):

            if i_episode % 100 == 0:
                print("\rEpisode {}/{} || average reward {}".format(i_episode, self.num_episodes,
                      np.mean(tmp_scores)), end="")
                sys.stdout.flush()

            score = 0
            state = self.env.reset()
            while True:

                policy_s = self.epsilon_greedy_probs(self.Q[state], i_episode, None)
                action = np.random.choice(np.arange(self.env.nA), p=policy_s)
                next_state, reward, done, info = self.env.step(action)
                score += reward

                # Sarsamax takes no sampling for the next_state, bux max.:
                self.Q[state][action] = self.update_Q(self.Q[state][action], np.max(self.Q[next_state]),
                                                      reward, self.alpha, self.gamma)

                state = next_state
                if done:
                    tmp_scores.append(score)
                    break

            if (i_episode % plot_every == 0):
                scores.append(np.mean(tmp_scores))

        self.print_policy()

        return

    def print_policy(self):
        policy_sarsamax = np.array([np.argmax(self.Q[key]) if key in self.Q else -1 \
                                    for key in np.arange(self.env.nS)]).reshape((self.shape, self.shape))
        print("\nEstimated Optimal Policy (UP = 3, RIGHT = 2, DOWN = 1, LEFT = 0):")
        print(policy_sarsamax)
