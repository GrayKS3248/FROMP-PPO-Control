# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:47:53 2020

@author: GKSch
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("results/pd_05/output", 'rb') as file:
    pd_299 = pickle.load(file)
with open("results/pd_1/output", 'rb') as file:
    pd_300 = pickle.load(file)
with open("results/pd_15/output", 'rb') as file:
    pd_301 = pickle.load(file)
with open("results/pd_2/output", 'rb') as file:
    pd_302 = pickle.load(file)
with open("results/pd_25/output", 'rb') as file:
    pd_303 = pickle.load(file)
with open("results/pd_3/output", 'rb') as file:
    pd_304 = pickle.load(file)
with open("results/pd_35/output", 'rb') as file:
    pd_305 = pickle.load(file)
with open("results/pd_4/output", 'rb') as file:
    pd_306 = pickle.load(file)
    
# Process data
pd_299_mean = np.mean(np.array(pd_299['logbook']['data'][0]['r_per_episode']))
pd_299_stdev = np.std(np.array(pd_299['logbook']['data'][0]['r_per_episode']))
pd_300_mean = np.mean(np.array(pd_300['logbook']['data'][0]['r_per_episode']))
pd_300_stdev = np.std(np.array(pd_300['logbook']['data'][0]['r_per_episode']))
pd_301_mean = np.mean(np.array(pd_301['logbook']['data'][0]['r_per_episode']))
pd_301_stdev = np.std(np.array(pd_301['logbook']['data'][0]['r_per_episode']))
pd_302_mean = np.mean(np.array(pd_302['logbook']['data'][0]['r_per_episode']))
pd_302_stdev = np.std(np.array(pd_302['logbook']['data'][0]['r_per_episode']))
pd_303_mean = np.mean(np.array(pd_303['logbook']['data'][0]['r_per_episode']))
pd_303_stdev = np.std(np.array(pd_303['logbook']['data'][0]['r_per_episode']))
pd_304_mean = np.mean(np.array(pd_304['logbook']['data'][0]['r_per_episode']))
pd_304_stdev = np.std(np.array(pd_304['logbook']['data'][0]['r_per_episode']))
pd_305_mean = np.mean(np.array(pd_305['logbook']['data'][0]['r_per_episode']))
pd_305_stdev = np.std(np.array(pd_305['logbook']['data'][0]['r_per_episode']))
pd_306_mean = np.mean(np.array(pd_306['logbook']['data'][0]['r_per_episode']))
pd_306_stdev = np.std(np.array(pd_306['logbook']['data'][0]['r_per_episode']))

# Plot data
objects = ('0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0')
y_pos = np.arange(len(objects))
performance = [pd_299_mean, pd_300_mean, pd_301_mean, pd_302_mean, pd_303_mean, pd_304_mean, pd_305_mean, pd_306_mean]
error = [pd_299_stdev, pd_300_stdev, pd_301_stdev, pd_302_stdev, pd_303_stdev, pd_304_stdev, pd_305_stdev, pd_306_stdev]
plt.barh(y_pos, performance, xerr=error, capsize=5.0, align='center')
plt.yticks(y_pos, objects)
plt.xlabel('Average Reward per Simulation Step')
plt.ylabel('K_P for Temperature Controller')
plt.title('PD Controller Relative Performance with Random Target Front Velocity, N=500')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/pd_results/kp-temp_performance.png', dpi = 500)
plt.close()