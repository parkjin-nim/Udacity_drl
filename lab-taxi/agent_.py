import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        self.eps = 1.
        self.eps_min = 0.00001
        self.i_episode = 1
        self.alpha = 0.07
        self.gamma = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        # make eps policy w/ existing epsilon
        self.policy = np.ones(self.nA) * self.eps / self.nA
        best_a = np.argmax(self.Q[state])
        self.policy[best_a] = (1 - self.eps)+(self.eps / self.nA)
        
        # return new action
        return np.random.choice(np.arange(self.nA), p = self.policy)                          

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Q-learning is implemented
        current = self.Q[state][action]  # estimate in Q-table (for current state, action pair)
        
        if not done:
            #Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0  # value of next state 
            #Qsa_next = np.dot(self.Q[next_state], self.policy)
            #next_action = np.random.choice(np.arange(self.nA), p = self.policy)
            next_action  = self.select_action(next_state)
            Qsa_next = self.Q[next_state][next_action]

            target = reward + (self.gamma * Qsa_next)               # construct TD target, gamma is 1.
            new_value = current + (self.alpha * (target - current)) # get updated value, alpha is .01
            self.Q[state][action] = new_value
        
        if done:
            Qsa_next = 0.
            target = reward + (self.gamma * Qsa_next)               # construct TD target, gamma is 1.
            new_value = current + (self.alpha * (target - current)) # get updated value, alpha is .01
            self.Q[state][action] = new_value

            # an episode ends, decay epsilon                      
            self.i_episode += 1
            self.eps = max(0.995**self.i_episode, self.eps_min)
            #self.eps = max(1./self.i_episode, 0.1)

        