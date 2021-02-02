# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:00:57 2020

@author: GKSch
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import shutil

# Find or create save paths
done = False
n_trainings = 1
while not done:
    path = "results/PPO_"+str(n_trainings)
    if not os.path.isdir(path):
        done = True
    else:
        n_trainings = n_trainings + 1
n_trainings = n_trainings - 1
if n_trainings == 0:
    raise RuntimeError("No training data found")
path = 'results/PPO_Training'
if not os.path.isdir(path):
    os.mkdir(path)
else:
    shutil.rmtree(path)
    os.mkdir(path)

# Load actor data
data_set = np.array([])
for current_training_session in range(n_trainings):
    current_folder = "PPO_" + str(current_training_session+1)
    with open("results/"+current_folder+"/output", 'rb') as file:
        load_file = pickle.load(file)
    if 'data' in load_file:
        data_set = np.append(data_set, load_file['data']['r_per_episode'])
    else:
        data_set = np.append(data_set, load_file['logbook']['data'][0]['r_per_episode'])
    
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
plt.title("Actor Learning Curve, Episode Normalized",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Average Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(data_set))],data_set,lw=2.0,c='r')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_mean_1.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Actor Learning Curve, Rolling Mean N=10",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Average Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(mean_10))],mean_10,lw=2.0,c='r')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_mean_10.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Actor Learning Curve, Rolling Mean N=20",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Average Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(mean_20))],mean_20,lw=2.0,c='r')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_mean_20.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Actor Learning Curve, Rolling Mean N=50",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Average Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(mean_50))],mean_50,lw=2.0,c='r')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_mean_50.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Actor Learning Curve, Rolling Mean N=100",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Average Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(mean_100))],mean_100,lw=2.0,c='r')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_mean_100.png', dpi = 500)
plt.close()

# Plot actor std data
plt.clf()
plt.title("Actor Learning Curve, Rolling Stdev N=10",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Stdev of Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(std_10))],std_10,lw=2.0,c='r')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_std_10.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Actor Learning Curve, Rolling Stdev N=20",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Stdev of Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(std_20))],std_20,lw=2.0,c='r')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_std_20.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Actor Learning Curve, Rolling Stdev N=50",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Stdev of Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(std_50))],std_50,lw=2.0,c='r')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_std_50.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Actor Learning Curve, Rolling Stdev N=100",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Stdev of Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(std_100))],std_100,lw=2.0,c='r')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_std_100.png', dpi = 500)
plt.close()

# Plot combined actor data
plt.clf()
plt.title("Actor Learning Curve, Rolling Mean N=10",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(mean_10))],mean_10,c='r')
plt.fill_between([*range(len(mean_10))],mean_10+std_10,mean_10-std_10,alpha=0.5,color='r',linewidth=0.0)
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_mean_std_10.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Actor Learning Curve, Rolling Mean N=20",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(mean_20))],mean_20,c='r')
plt.fill_between([*range(len(mean_20))],mean_20+std_20,mean_20-std_20,alpha=0.5,color='r',linewidth=0.0)
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_mean_std_20.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Actor Learning Curve, Rolling Mean N=20",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(mean_50))],mean_50,c='r')
plt.fill_between([*range(len(mean_50))],mean_50+std_50,mean_50-std_50,alpha=0.5,color='r',linewidth=0.0)
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_mean_std_50.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Actor Learning Curve, Rolling Mean N=20",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Reward per Simulation Step",fontsize='large')
plt.plot([*range(len(mean_100))],mean_100,c='r')
plt.fill_between([*range(len(mean_100))],mean_100+std_100,mean_100-std_100,alpha=0.5,color='r',linewidth=0.0)
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/actor_mean_std_100.png', dpi = 500)
plt.close()

# Load critic data
data_set = np.array([])
for current_training_session in range(n_trainings):
    current_folder = "PPO_" + str(current_training_session+1)
    with open("results/"+current_folder+"/output", 'rb') as file:
        load_file = pickle.load(file)  
    if 'data' in load_file:
        data_set = np.append(data_set, load_file['data']['value_error'][0])
    else:
        data_set = np.append(data_set, load_file['logbook']['data'][0]['value_error'][0])
    
# Format data
data_set = data_set.squeeze()

# Plot critic data
plt.clf()
plt.title("Critic Learning Curve",fontsize='xx-large')
plt.xlabel("Optimization Step",fontsize='large')
plt.ylabel("Critic MSE Loss",fontsize='large')
plt.plot([*range(len(data_set))],data_set,c='r',lw=2.0)
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/critic.png', dpi = 500)
plt.close()

plt.clf()
plt.title("Critic Learning Log Curve",fontsize='xx-large')
plt.xlabel("Optimization Step",fontsize='large')
plt.ylabel("Critic MSE Loss",fontsize='large')
plt.plot([*range(len(data_set))],data_set,c='r',lw=2.0)
plt.yscale("log")
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/critic_log.png', dpi = 500)
plt.close()

# Load location stdev data data
data_set = np.array([])
for current_training_session in range(n_trainings):
    current_folder = "PPO_" + str(current_training_session+1)
    with open("results/"+current_folder+"/output", 'rb') as file:
        load_file = pickle.load(file)  
    if 'data' in load_file:
        data_set = np.append(data_set, load_file['data']['x_rate_stdev'])
    else:
        data_set = np.append(data_set, load_file['logbook']['data'][0]['x_loc_rate_stdev'])
    
# Format data
data_set = data_set.squeeze()

# Plot loc stdev data
plt.clf()
plt.title("Laser X Position Rate Stdev",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Laser X Position Rate Stdev [m/s]",fontsize='large')
if 'env' in load_file:
    plt.plot([*range(len(data_set))],load_file['env'].loc_rate_scale*data_set,c='r',lw=2.0)
else:
    plt.plot([*range(len(data_set))],load_file['logbook']['envs'][0].loc_rate_scale*data_set,c='r',lw=2.0)
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/x_rate_stdev.png', dpi = 500)
plt.close()

# Load location stdev data data
data_set = np.array([])
for current_training_session in range(n_trainings):
    current_folder = "PPO_" + str(current_training_session+1)
    with open("results/"+current_folder+"/output", 'rb') as file:
        load_file = pickle.load(file)  
    if 'data' in load_file:
        data_set = np.append(data_set, load_file['data']['y_rate_stdev'])
    else:
        data_set = np.append(data_set, load_file['logbook']['data'][0]['y_loc_rate_stdev'])
    
# Format data
data_set = data_set.squeeze()

# Plot loc stdev data
plt.clf()
plt.title("Laser Y Position Rate Stdev",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel("Laser Y Position Rate Stdev [m/s]",fontsize='large')
if 'env' in load_file:
    plt.plot([*range(len(data_set))],load_file['env'].loc_rate_scale*data_set,c='r',lw=2.0)
else:
    plt.plot([*range(len(data_set))],load_file['logbook']['envs'][0].loc_rate_scale*data_set,c='r',lw=2.0)
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/y_rate_stdev.png', dpi = 500)
plt.close()

# Load magnitude stdev data data
data_set = np.array([])
for current_training_session in range(n_trainings):
    current_folder = "PPO_" + str(current_training_session+1)
    with open("results/"+current_folder+"/output", 'rb') as file:
        load_file = pickle.load(file)  
    if 'data' in load_file:
        data_set = np.append(data_set, load_file['data']['mag_stdev'])
    else:
        data_set = np.append(data_set, load_file['logbook']['data'][0]['mag_stdev'])
    
# Format data
data_set = data_set.squeeze()

# Plot mag stdev data
plt.clf()
plt.title("Laser Magnitude Stdev",fontsize='xx-large')
plt.xlabel("Episode",fontsize='large')
plt.ylabel('Laser Magnitude Stdev [K/s]',fontsize='large')
if 'env' in load_file:
    plt.plot([*range(len(data_set))],load_file['env'].mag_scale*load_file['env'].max_input_mag*data_set,c='r',lw=2.0)
else:
    plt.plot([*range(len(data_set))],load_file['logbook']['envs'][0].mag_scale*load_file['logbook']['envs'][0].max_input_mag*data_set,c='r',lw=2.0)
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig(path+'/mag_stdev.png', dpi = 500)
plt.close()