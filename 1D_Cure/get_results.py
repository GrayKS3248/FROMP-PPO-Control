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

with open("results/ppo_6/output", 'rb') as file:
    data_6 = pickle.load(file)  
set_6=data_6['logbook']['data'][0]['r_per_episode']

with open("results/ppo_7/output", 'rb') as file:
    data_7 = pickle.load(file)  
set_7=data_7['logbook']['data'][0]['r_per_episode']


# Combine data
total_set = np.concatenate((set_1, set_2, set_3, set_4, set_5, set_6, set_7))

# Process data
ts = pd.Series(total_set)
mean_10 = ts.rolling(window=10).mean()
mean_20 = ts.rolling(window=20).mean()
mean_50 = ts.rolling(window=50).mean()
mean_100 = ts.rolling(window=100).mean()
std_10 = ts.rolling(window=10).std()
std_20 = ts.rolling(window=20).std()
std_50 = ts.rolling(window=50).std()
std_100 = ts.rolling(window=100).std()

# Plot mean data
plt.clf()
title_str = "Actor Learning Curve, Episode Normalized"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Average Reward per Simulation Step")
plt.plot([*range(len(total_set))],total_set)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/mean_1.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=10"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_10))],mean_10)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/mean_10.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=20"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_20))],mean_20)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/mean_20.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=50"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_50))],mean_50)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/mean_50.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=100"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_100))],mean_100)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/mean_100.png', dpi = 500)
plt.close()

# Plot std data
plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=10"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_10))],std_10)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/std_10.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=20"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_20))],std_20)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/std_20.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=50"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_50))],std_50)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/std_50.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=100"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_100))],std_100)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/std_100.png', dpi = 500)
plt.close()

# Plot combined data
plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=10"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_10))],mean_10)
plt.fill_between([*range(len(mean_10))],mean_10+std_10,mean_10-std_10,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/mean_std_10.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=20"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_20))],mean_20)
plt.fill_between([*range(len(mean_20))],mean_20+std_20,mean_20-std_20,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/mean_std_20.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=50"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_50))],mean_50)
plt.fill_between([*range(len(mean_50))],mean_50+std_50,mean_50-std_50,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/mean_std_50.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=100"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_100))],mean_100)
plt.fill_between([*range(len(mean_100))],mean_100+std_100,mean_100-std_100,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/ppo_results/mean_std_100.png', dpi = 500)
plt.close()