# Taxi Problem

### Getting Started

Read the description of the environment in subsection 3.1 of [this paper](https://arxiv.org/pdf/cs/9905014.pdf).  You can verify that the description in the paper matches the OpenAI Gym environment by peeking at the code [here](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py).


### Instructions

The repository contains three files:
- `agent.py`: Develop your reinforcement learning agent here.  This is the only file that you should modify.
- `monitor.py`: The `interact` function tests how well your agent learns from interaction with the environment.
- `main.py`: Run this file in the terminal to check the performance of your agent.

Begin by running the following command in the terminal:
```
python main.py
```

When you run `main.py`, the agent that you specify in `agent.py` interacts with the environment for 20,000 episodes.  The details of the interaction are specified in `monitor.py`, which returns two variables: `avg_rewards` and `best_avg_reward`.
- `avg_rewards` is a deque where `avg_rewards[i]` is the average (undiscounted) return collected by the agent from episodes `i+1` to episode `i+100`, inclusive.  So, for instance, `avg_rewards[0]` is the average return collected by the agent over the first 100 episodes.
- `best_avg_reward` is the largest entry in `avg_rewards`.  This is the final score that you should use when determining how well your agent performed in the task.

Your assignment is to modify the `agents.py` file to improve the agent's performance.
- Use the `__init__()` method to define any needed instance variables.  Currently, we define the number of actions available to the agent (`nA`) and initialize the action values (`Q`) to an empty dictionary of arrays.  Feel free to add more instance variables; for example, you may find it useful to define the value of epsilon if the agent uses an epsilon-greedy policy for selecting actions.
- The `select_action()` method accepts the environment state as input and returns the agent's choice of action.  The default code that we have provided randomly selects an action.
- The `step()` method accepts a (`state`, `action`, `reward`, `next_state`) tuple as input, along with the `done` variable, which is `True` if the episode has ended.  The default code (which you should certainly change!) increments the action value of the previous state-action pair by 1.  You should change this method to use the sampled tuple of experience to update the agent's knowledge of the problem.

Once you have modified the function, you need only run `python main.py` to test your new agent.

OpenAI Gym [defines "solving"](https://gym.openai.com/envs/Taxi-v1/) this task as getting average return of 9.7 over 100 consecutive trials.  

### Run example

|   iter    |  target   |   alpha   |  epsilon  |   gamma   |

-------------------------------------------------------------
Episode 20000/20000 || Best average reward 9.215

|  1        |  9.21     |  0.1      |  0.1      |  0.9      |
Episode 20000/20000 || Best average reward 9.286

|  2        |  9.2      |  0.1454   |  0.9699   |  0.8644   |
Episode 20000/20000 || Best average reward 9.286

|  3        |  9.2      |  0.2406   |  0.707    |  0.8998   |
Episode 20000/20000 || Best average reward 9.292

|  4        |  9.29     |  0.3582   |  0.4184   |  0.853    |
Episode 20000/20000 || Best average reward 9.395

|  5        |  9.39     |  0.5      |  0.01     |  0.5      |
Episode 20000/20000 || Best average reward 9.025

|  6        |  9.02     |  0.5      |  0.995    |  0.5      |
Episode 20000/20000 || Best average reward 9.219

|  7        |  9.21     |  0.4731   |  0.01709  |  0.9798   |
Episode 20000/20000 || Best average reward 9.217

|  8        |  9.21     |  0.1      |  0.489    |  0.5      |
Episode 20000/20000 || Best average reward 9.617

|  9        |  9.61     |  0.1      |  0.01     |  0.5      |
Episode 20000/20000 || Best average reward 9.325

|  10       |  9.32     |  0.5      |  0.995    |  1.0      |
Episode 20000/20000 || Best average reward 9.212

|  11       |  9.21     |  0.2061   |  0.01     |  0.5      |
Episode 20000/20000 || Best average reward 9.377

|  12       |  9.37     |  0.5      |  0.4691   |  0.5      |
Episode 20000/20000 || Best average reward 9.158

|  13       |  9.15     |  0.1534   |  0.8065   |  0.7767   |
Episode 20000/20000 || Best average reward 9.189

|  14       |  9.18     |  0.5      |  0.5531   |  1.0      |
Episode 20000/20000 || Best average reward 9.242

|  15       |  9.24     |  0.4415   |  0.06551  |  0.5809   |
Episode 20000/20000 || Best average reward 9.165

|  16       |  9.16     |  0.1005   |  0.4936   |  0.9736   |
Episode 20000/20000 || Best average reward 9.213

|  17       |  9.2      |  0.3388   |  0.3383   |  0.9435   |
Episode 20000/20000 || Best average reward 9.462

|  18       |  9.46     |  0.1      |  0.995    |  0.5      |
Episode 20000/20000 || Best average reward 9.129

|  19       |  9.12     |  0.1      |  0.1867   |  0.5      |
Episode 20000/20000 || Best average reward 9.221

|  20       |  9.22     |  0.1703   |  0.2376   |  0.6747   |
Episode 20000/20000 || Best average reward 9.185

|  21       |  9.18     |  0.1      |  0.01     |  0.6792   |
Episode 20000/20000 || Best average reward 9.194

|  22       |  9.19     |  0.2814   |  0.6749   |  0.756    |
Episode 20000/20000 || Best average reward 9.356

|  23       |  9.35     |  0.3702   |  0.3796   |  0.5846   |
Episode 20000/20000 || Best average reward 9.295

|  24       |  9.29     |  0.2703   |  0.2741   |  0.6746   |
Episode 20000/20000 || Best average reward 9.191

|  25       |  9.19     |  0.4627   |  0.915    |  0.9147   |
Episode 20000/20000 || Best average reward 9.275

|  26       |  9.27     |  0.5      |  0.2773   |  0.734    |
Episode 20000/20000 || Best average reward 9.378

|  27       |  9.37     |  0.3638   |  0.7186   |  0.6848   |
Episode 20000/20000 || Best average reward 9.163

|  28       |  9.16     |  0.2794   |  0.759    |  0.5      |
Episode 20000/20000 || Best average reward 9.275

|  29       |  9.27     |  0.5      |  0.6193   |  0.6715   |

=============================================================