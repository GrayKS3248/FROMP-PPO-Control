# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:00:57 2020

@author: GKSch
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Save path
path = 'PPO-Results'
n_trainings = 4

# Load actor data
data_set = np.array([])
for current_training_session in range(n_trainings):
    current_folder = "ppo_" + str(current_training_session+1)
    with open("results/"+current_folder+"/output", 'rb') as file:
        data = pickle.load(file)  
    data_set = np.append(data_set, data['logbook']['data'][0]['r_per_episode'])
    
# Format data
data_set = data_set.squeeze()

# Process actor data
ts = pd.Series(data_set)
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
plt.plot([*range(len(data_set))],data_set)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_mean_1.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=10"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_10))],mean_10)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_mean_10.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=20"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_20))],mean_20)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_mean_20.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=50"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_50))],mean_50)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_mean_50.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=100"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_100))],mean_100)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_mean_100.png', dpi = 500)
plt.close()

# Plot actor std data
plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=10"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_10))],std_10)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_std_10.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=20"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_20))],std_20)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_std_20.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=50"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_50))],std_50)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_std_50.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Std N=100"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Std Reward per Simulation Step")
plt.plot([*range(len(std_100))],std_100)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_std_100.png', dpi = 500)
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
plt.savefig('results/'+path+'/actor_mean_std_10.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=20"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_20))],mean_20)
plt.fill_between([*range(len(mean_20))],mean_20+std_20,mean_20-std_20,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_mean_std_20.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=50"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_50))],mean_50)
plt.fill_between([*range(len(mean_50))],mean_50+std_50,mean_50-std_50,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_mean_std_50.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Actor Learning Curve, Rolling Mean N=100"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward per Simulation Step")
plt.plot([*range(len(mean_100))],mean_100)
plt.fill_between([*range(len(mean_100))],mean_100+std_100,mean_100-std_100,alpha=0.6)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/actor_mean_std_100.png', dpi = 500)
plt.close()

# Load critic data
data_set = np.array([])
for current_training_session in range(n_trainings):
    current_folder = "ppo_" + str(current_training_session+1)
    with open("results/"+current_folder+"/output", 'rb') as file:
        data = pickle.load(file)  
    data_set = np.append(data_set, data['logbook']['data'][0]['value_error'])
    
# Format data
data_set = data_set.squeeze()

# Plot critic data
plt.clf()
title_str = "Critic Learning Curve"
plt.title(title_str)
plt.xlabel("Optimization Step")
plt.ylabel("Critic MSE Loss")
plt.plot([*range(len(data_set))],data_set)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/critic.png', dpi = 500)
plt.close()

plt.clf()
title_str = "Critic Learning Curve"
plt.title(title_str)
plt.xlabel("Optimization Step")
plt.ylabel("Critic MSE Loss")
plt.plot([*range(len(data_set))],data_set)
plt.yscale("log")
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/critic_log.png', dpi = 500)
plt.close()

# Load location stdev data data
data_set = np.array([])
for current_training_session in range(n_trainings):
    current_folder = "ppo_" + str(current_training_session+1)
    with open("results/"+current_folder+"/output", 'rb') as file:
        data = pickle.load(file)  
    data_set = np.append(data_set, data['logbook']['data'][0]['x_loc_rate_stdev'])
    
# Format data
data_set = data_set.squeeze()

# Plot loc stdev data
plt.clf()
title_str = "Laser X Position Rate Stdev"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Laser X Position Rate Stdev [m/s]")
plt.plot([*range(len(data_set))],data['logbook']['envs'][0].loc_rate_scale*data_set)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/x_loc_rate_stdev.png', dpi = 500)
plt.close()

# Load location stdev data data
data_set = np.array([])
for current_training_session in range(n_trainings):
    current_folder = "ppo_" + str(current_training_session+1)
    with open("results/"+current_folder+"/output", 'rb') as file:
        data = pickle.load(file)  
    data_set = np.append(data_set, data['logbook']['data'][0]['y_loc_rate_stdev'])
    
# Format data
data_set = data_set.squeeze()

# Plot loc stdev data
plt.clf()
title_str = "Laser Y Position Rate Stdev"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel("Laser Y Position Rate Stdev [m/s]")
plt.plot([*range(len(data_set))],data['logbook']['envs'][0].loc_rate_scale*data_set)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/y_loc_rate_stdev.png', dpi = 500)
plt.close()

# Load magnitude stdev data data
data_set = np.array([])
for current_training_session in range(n_trainings):
    current_folder = "ppo_" + str(current_training_session+1)
    with open("results/"+current_folder+"/output", 'rb') as file:
        data = pickle.load(file)  
    data_set = np.append(data_set, data['logbook']['data'][0]['mag_stdev'])
    
# Format data
data_set = data_set.squeeze()

# Plot mag stdev data
plt.clf()
title_str = "Laser Magnitude Stdev"
plt.title(title_str)
plt.xlabel("Episode")
plt.ylabel('Laser Magnitude Stdev [K/s]')
plt.plot([*range(len(data_set))],data['logbook']['envs'][0].mag_scale*data['logbook']['envs'][0].max_input_mag*data_set)
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/'+path+'/mag_stdev.png', dpi = 500)
plt.close()