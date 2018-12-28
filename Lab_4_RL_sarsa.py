#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 08:09:50 2018

@author: salihemredevrim
"""
import gym
import numpy as np
import matplotlib.pyplot as plt

#Mountain car using Sarsa

slow = False
gym.envs.register(
    id='MountainCarMyEasyVersion-v0',

    entry_point='gym.envs.classic_control:MountainCarEnv',

    max_episode_steps=10000,      # MountainCar-v0 uses 200
);

env = gym.make('MountainCarMyEasyVersion-v0');

print("Number of actions:", env.action_space); 
print("State components:", env.observation_space); 

#Parameters and Initialization
alpha = 0.5; #learning rate
gamma = 1; #discount factor
states_num = 25; #number of states for discretization
rewards=[]; #keeps rewards
Q = np.zeros((states_num,states_num, 3)); #initialize Q table, all set as 0 
#20 by 20 state matrix for 3 actions go left, do nothing or go right 
eps = 0; #probability for random moves
max_iter = 10000; #maximum number of iterations

#Discretization
#Find high and low values in the environment, then size of the each state
low = env.observation_space.low;
high = env.observation_space.high;
size = (high - low) / states_num;

def discrete(observation):
    # observation[0] keeps position, observation[1] keeps velocity
    pos = int((observation[0] - low[0]) / size[0]);
    vel = int((observation[1] - low[1]) / size[1]);
    
    if pos >= states_num: #must be in given state intervals
        pos = states_num - 1; 
    elif pos < 0:
        pos = 0;
    
    if vel >= states_num:
        vel = states_num - 1;
    elif vel < 0:
        vel = 0;    
        
    state1 = [pos, vel];
    return state1

def get_action(observation, t):
    if np.random.uniform(0, 1) < eps: #with a small probability, an action is selected at random
        return env.action_space.sample();
    
    state1 = discrete(observation);
    return np.argmax(np.array(Q[tuple(state1)]));

#SARSA
#https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html    
def sarsa(observation_0, observation_1, reward, action_0 , action_1, t):
    
    state1 = discrete(observation_1);
    state0 = discrete(observation_0);
    
    Q_next = Q[tuple(state1)][action_1];
    delta1 = reward + gamma * Q_next - Q[tuple(state0)][action_0]; #temporal difference
    Q[tuple(state0)][action_0] = Q[tuple(state0)][action_0] + alpha * delta1;

#Iterations
for i in range(max_iter):
    observation = env.reset();
    done = False;
    timesteps = 0;
    
    while not done:
        if slow: 
            env.render();
        action = get_action(observation, i);
        observation1, reward, done, info = env.step(action);
        next_action = get_action(observation1, i);
        sarsa(observation, observation1, reward, action, next_action, i); #implement sarsa and update the state and action
        observation = observation1;
        action = next_action;
        timesteps+=1; 
        
        if done:

            rewards.append(timesteps); #save the number of moves until to reach the flag
            break
        
        
    if i % 500 == 0: #prints per 500 iterations       
        print("Episode finished after ", timesteps, "timesteps.");
        
        if slow: 
            print(observation);
        if slow: 
            print(reward);
        if slow: 
            print(done);
            
#State-value visualization
map1 = np.amax(Q, axis=2);
plt.imshow(map1, origin='lower');
plt.colorbar();
plt.show();

#Best matrix
best = np.argmax(Q, axis=2);
print(best);

#Rewards plot
plt.plot(rewards); 
