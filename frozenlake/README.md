# FrozenLake-V3 Problem

### Getting Started

Read the description of the environment [here](https://gym.openai.com/envs/FrozenLake-v0/).  You can verify that the description in the paper matches the OpenAI Gym environment by peeking at the code [here](https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py).


### Instructions

The repository contains two files:
- `sarsamax_agent.py`: Develop your reinforcement learning agent here.  This is the only file that you should modify.
- `main.py`: Run this file in the terminal to check the performance of your agent.

Begin by running the following command in the terminal:
```
python main.py
```

When you run `main.py`, it first takes environment setting inputs where you set either yes or no for the stochasticity, then either 4x4 or 8x8 for the lake size. The agent that you specify in `sarsamax_agent.py` interacts with the environment.  

Then you will see the menu as below. With menu 1, the manual operation can be checked. With menu 2, your q-learning model can be trained. With menu 3, you can test your model working.
```
1. Checking Frozen_Lake 
2. Q-learning
3. Testing after learning 
4. Exit
```

The result first shows the optimal policy trained(UP = 3, RIGHT = 2, DOWN = 1, LEFT = 0). Then the average is the score how well your agent works from 0 to 1.

- Use the `__init__()` method to define any needed instance variables.  Currently,  the agent is initialized with the action values (`Q`) to an empty dictionary of arrays.  Feel free to add more instance variables; for example, you may find it useful to define the value of epsilon if the agent uses an epsilon-greedy policy for selecting actions.
- The `learn()` method is where q-learning is implemented. For deterministic setting(4x4, 8x8), the interation number is 20,000 episodes. For stochastic setting, it is 100,000. The method accepts a (`state`, `action`, `reward`, `next_state`) tuple from the environment, along with the `done` variable, which is `True` if the episode has ended. 
- The `epsilon_greedy_probs()` method accepts the environment state as input and returns the agent's choice of action.  This code provided selects an action by epsilon greedy policy with the exponential decay every episode. Note that q-learning takes its next greedy action and update the Q table, before it actually takes to the environment.

OpenAI Gym [defines "solving"](https://gym.openai.com/envs/Taxi-v1/) this task as getting average return of 0.7 over 100 consecutive trials for the stochastic case.

### Run example 
```
Estimated Optimal Policy (UP = 3, RIGHT = 2, DOWN = 1, LEFT = 0)
[[0 3 3 3] 
 [0 0 0 0] 
 [3 1 0 0] 
 [0 2 1 0]]
 
1. Checking Frozen_Lake 
2. Q-learning
3. Testing after learning 
4. Exit
select: 3 avg: 0.827


Estimated Optimal Policy (UP = 3, RIGHT = 2, DOWN = 1, LEFT = 0):
[[3 2 3 2 2 3 2 2]
 [3 3 3 3 3 3 3 2] 
 [0 3 0 0 2 3 2 2] 
 [0 0 0 3 0 0 2 2] 
 [0 3 3 0 3 1 3 2] 
 [0 0 0 2 2 0 0 2] 
 [0 0 2 0 0 3 0 2] 
 [0 1 0 0 1 1 2 0]]
 
1. Checking Frozen_Lake 
2. Q-learning
3. Testing after learning 
4. Exit
select: 3 avg: 1.0
```
