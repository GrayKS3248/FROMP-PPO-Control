# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:00:57 2020

@author: GKSch
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load data
with open("results/ppo_1/output", 'rb') as file:
    data_1 = pickle.load(file)  
set_1=data_1['logbook']['data'][0]['r_per_episode']

with open("results/ppo_2/output", 'rb') as file:
    data_2 = pickle.load(file)  
set_2=data_2['logbook']['data'][0]['r_per_episode']

with open("results/ppo_3/output", 'rb') as file:
    data_3 = pickle.load(file)  
set_3=data_3['logbook']['data'][0]['r_per_episode']

with open("results/ppo_4/output", 'rb') as file:
    data_4 = pickle.load(file)  
set_4=data_4['logbook']['data'][0]['r_per_episode']

with open("results/ppo_5/output", 'rb') as file:
    data_5 = pickle.load(file)  
set_5=data_5['logbook']['data'][0]['r_per_episode']

# with open("results/ppo_6/output", 'rb') as file:
#     data_6 = pickle.load(file)  
# set_6=data_6['logbook']['data'][0]['r_per_episode']

# with open("results/ppo_7/output", 'rb') as file:
#     data_7 = pickle.load(file)  
# set_7=data_7['logbook']['data'][0]['r_per_episode']

# with open("results/ppo_8/output", 'rb') as file:
#     data_8 = pickle.load(file)  
# set_8=data_8['logbook']['data'][0]['r_per_episode']

# Combine actor data
total_set = np.concatenate((set_1, set_2, set_3, set_4, set_5))#, set_6, set_7, set_8))

# Process actor data
ts = pd.Series(total_set)
mean_10 = ts.rolling(window=10).mean()
mean_20 = ts.rolling(window=20).mean()
mean_50 = ts.rolling(window=50).mean()
mean_100 = ts.rolling(window=100).mean()
std_10 = ts.rolling(window=10).std()
std_20 = ts.rolling(window=20).std()
std_50 = ts.rolling(window=50).std()
std_100 = ts.rolling(window=100).std()

# Plot actor mean data
plt.clf()
title_str = "Actor Learning Curve, Episode Normalized"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Average Reward per Simulation Step")
plt.plot([*range(len(total_set))],total_set)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_mean_1.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=10"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_10))],mean_10)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_mean_10.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=20"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_20))],mean_20)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_mean_20.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=50"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_50))],mean_50)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_mean_50.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=100"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_100))],mean_100)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_mean_100.png', dpi = 500)
plt.close()

# Plot actor std data
plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=10"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_10))],std_10)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_std_10.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=20"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_20))],std_20)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_std_20.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=50"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_50))],std_50)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_std_50.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=100"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_100))],std_100)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_std_100.png', dpi = 500)
plt.close()

# Plot combined actor data
plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=10"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_10))],mean_10)
plt.fill_between([*range(len(mean_10))],mean_10+std_10,mean_10-std_10,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_mean_std_10.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=20"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_20))],mean_20)
plt.fill_between([*range(len(mean_20))],mean_20+std_20,mean_20-std_20,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_mean_std_20.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=50"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_50))],mean_50)
plt.fill_between([*range(len(mean_50))],mean_50+std_50,mean_50-std_50,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_mean_std_50.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=100"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_100))],mean_100)
plt.fill_between([*range(len(mean_100))],mean_100+std_100,mean_100-std_100,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/actor_mean_std_100.png', dpi = 500)
plt.close()

# Load critic data
set_1=data_1['logbook']['data'][0]['value_error'][0]
set_2=data_2['logbook']['data'][0]['value_error'][0]
set_3=data_3['logbook']['data'][0]['value_error'][0]
set_4=data_4['logbook']['data'][0]['value_error'][0]
set_5=data_5['logbook']['data'][0]['value_error'][0]
# set_6=data_6['logbook']['data'][0]['value_error'][0]
# set_7=data_7['logbook']['data'][0]['value_error'][0]
# set_8=data_8['logbook']['data'][0]['value_error'][0]

# Combine critic data
total_set = np.concatenate((set_1, set_2, set_3, set_4, set_5))#, set_6, set_7, set_8))

# Plot critic data
plt.clf()
title_str = "Critic Learning Curve"
plt.title(title_str)
plt.xlabel("Optimization Step")
plt.ylabel("Critic MSE Loss")
plt.plot([*range(len(total_set))],total_set)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/critic.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Critic Learning Curve"
plt.title(title_str)
plt.xlabel("Optimization Step")
plt.ylabel("Critic MSE Loss")
plt.plot([*range(len(total_set))],total_set)
plt.yscale("log")
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/critic_log.png', dpi = 500)
plt.close()

# Load loc stdev data
set_1=data_1['logbook']['data'][0]['loc_rate_stdev']
set_2=data_2['logbook']['data'][0]['loc_rate_stdev']
set_3=data_3['logbook']['data'][0]['loc_rate_stdev']
set_4=data_4['logbook']['data'][0]['loc_rate_stdev']
set_5=data_5['logbook']['data'][0]['loc_rate_stdev']
# set_6=data_6['logbook']['data'][0]['loc_rate_stdev']
# set_7=data_7['logbook']['data'][0]['loc_rate_stdev']
# set_8=data_8['logbook']['data'][0]['loc_rate_stdev']

# Combine loc stdev data
total_set = np.concatenate((set_1, set_2, set_3, set_4, set_5))#, set_6, set_7, set_8))

# Plot loc stdev data
plt.clf()
title_str = "Laser Position Rate Stdev"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Laser Position Rate Stdev [m/s]")
plt.plot([*range(len(total_set))],data_1['logbook']['envs'][0].loc_rate_scale*total_set)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/loc_rate_stdev.png', dpi = 500)
plt.close()

# Load mag stdev data
set_1=data_1['logbook']['data'][0]['mag_stdev']
set_2=data_2['logbook']['data'][0]['mag_stdev']
set_3=data_3['logbook']['data'][0]['mag_stdev']
set_4=data_4['logbook']['data'][0]['mag_stdev']
set_5=data_5['logbook']['data'][0]['mag_stdev']
# set_6=data_6['logbook']['data'][0]['mag_stdev']
# set_7=data_7['logbook']['data'][0]['mag_stdev']
# set_8=data_8['logbook']['data'][0]['mag_stdev']

# Combine mag stdev data
total_set = np.concatenate((set_1, set_2, set_3, set_4, set_5))#, set_6, set_7, set_8))

# Plot mag stdev data
plt.clf()
title_str = "Laser Magnitude Stdev"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel('Laser Magnitude Stdev [K/s]')
plt.plot([*range(len(total_set))],data_1['logbook']['envs'][0].mag_scale*data_1['logbook']['envs'][0].max_input_mag*total_set)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/mag_stdev.png', dpi = 500)
plt.close()