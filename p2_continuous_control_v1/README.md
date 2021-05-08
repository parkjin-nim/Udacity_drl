[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Continuous Control

This is the second project in the Udacity Deep Reinforcement Learning Nanodegree.

This project works with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

## Background
In this environment, a double-jointed arm can move to target locations.

Reward: A reward of +0.1 is provided for each step that the agent's hand is in the goal location.

Goal: The goal of your agent is to maintain its position at the target location for as many time steps as possible.

Observation Space: The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints.

Action Space: Every entry in the action vector should be a number between -1 and 1.

In order to consider the environment has been solved, the agent must get an average score of +30 over 100 consecutive episodes.

## Getting Started
It is recommended to follow the Udacity DRL ND dependencies [instructions here](https://github.com/udacity/deep-reinforcement-learning#dependencies) 

This project utilises [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [NumPy](http://www.numpy.org/) and [PyTorch](https://pytorch.org/) 

A prebuilt simulator is required in be installed. You need only select the environment that matches your operating system:

### Version 1: One (1) Agent
Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

The file needs to placed in the root directory of the repository and unzipped.

Next, before starting the environment utilising the corresponding prebuilt app from Udacity  **Before running the code cell in the notebook**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.

## Instructions

Run the `Continuous_Control_v1.ipynb` notebook to train the DDPG agent using the drlnd kernel.

Once trained, the actor critic model weights are saved in checkpoint_actor_v1.pth and checkpint_critic_v1.pth.

[![Watch the vide](https://i9.ytimg.com/vi_webp/2K0p_ZrE2vs/mqdefault.webp?time=1620479700000&sqp=CNSd2oQG&rs=AOn4CLBoEef-mEk-RHtm6nv-eI1ubv6s7w)](https://youtu.be/2K0p_ZrE2vs)


