# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:47:53 2020

@author: GKSch
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("results/ppo_results/output", 'rb') as file:
    ppo = pickle.load(file)
with open("results/pd_1/output", 'rb') as file:
    pd_1 = pickle.load(file)
with open("results/pd_2/output", 'rb') as file:
    pd_2 = pickle.load(file)
with open("results/control/output", 'rb') as file:
    con = pickle.load(file)
    
# Process data
ppo_episode_rewards = []
pd_1_episode_rewards = []
pd_2_episode_rewards = []
con_episode_rewards = []
for current_agent in range(len(ppo['logbook']['agents'])):
    ppo_episode_rewards.append(ppo['logbook']['data'][current_agent]['r_per_episode'])
    pd_1_episode_rewards.append(pd_1['logbook']['data'][current_agent]['r_per_episode'])
    pd_2_episode_rewards.append(pd_2['logbook']['data'][current_agent]['r_per_episode'])
    con_episode_rewards.append(con['logbook']['data'][current_agent]['r_per_episode'])
ppo_episode_rewards = np.array(ppo_episode_rewards)
ppo_mean = np.mean(ppo_episode_rewards)
ppo_stdev = np.std(ppo_episode_rewards)
pd_1_episode_rewards = np.array(pd_1_episode_rewards)
pd_1_mean = np.mean(pd_1_episode_rewards)
pd_1_stdev = np.std(pd_1_episode_rewards)
pd_2_episode_rewards = np.array(pd_2_episode_rewards)
pd_2_mean = np.mean(pd_2_episode_rewards)
pd_2_stdev = np.std(pd_2_episode_rewards)
con_episode_rewards = np.array(con_episode_rewards)
con_mean = np.mean(con_episode_rewards)
con_stdev = np.std(con_episode_rewards)

# Plot data
objects = ('Control', 'PD', 'Tuned PD', 'PPO')
y_pos = np.arange(len(objects))
performance = [con_mean, pd_1_mean, pd_2_mean, ppo_mean]+1.01*abs(con_mean)
error = [con_stdev, pd_1_stdev, pd_2_stdev, ppo_stdev]
plt.barh(y_pos, performance, xerr=error, capsize=5.0, align='center')
plt.yticks(y_pos, objects)
plt.xlabel('Average Reward per Simulation Step')
plt.title('Controller Relative Performance with Constant Target Front Velocity, N=10000')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/relative_performance.png', dpi = 500)
plt.close()