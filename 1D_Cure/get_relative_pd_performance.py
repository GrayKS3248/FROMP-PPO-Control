# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:47:53 2020

@author: GKSch
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("results/pd_1/output", 'rb') as file:
    pd_1 = pickle.load(file)
with open("results/pd_2/output", 'rb') as file:
    pd_2 = pickle.load(file)
with open("results/pd_3/output", 'rb') as file:
    pd_3 = pickle.load(file)
with open("results/pd_4/output", 'rb') as file:
    pd_4 = pickle.load(file)
with open("results/pd_5/output", 'rb') as file:
    pd_5 = pickle.load(file)
with open("results/pd_6/output", 'rb') as file:
    pd_6 = pickle.load(file)
with open("results/pd_7/output", 'rb') as file:
    pd_7 = pickle.load(file)
with open("results/pd_8/output", 'rb') as file:
    pd_8 = pickle.load(file)
    
# Process data
pd_1_mean = np.mean(np.array(pd_1['logbook']['data'][0]['r_per_episode']))
pd_1_stdev = np.std(np.array(pd_1['logbook']['data'][0]['r_per_episode']))
pd_2_mean = np.mean(np.array(pd_2['logbook']['data'][0]['r_per_episode']))
pd_2_stdev = np.std(np.array(pd_2['logbook']['data'][0]['r_per_episode']))
pd_3_mean = np.mean(np.array(pd_3['logbook']['data'][0]['r_per_episode']))
pd_3_stdev = np.std(np.array(pd_3['logbook']['data'][0]['r_per_episode']))
pd_4_mean = np.mean(np.array(pd_4['logbook']['data'][0]['r_per_episode']))
pd_4_stdev = np.std(np.array(pd_4['logbook']['data'][0]['r_per_episode']))
pd_5_mean = np.mean(np.array(pd_5['logbook']['data'][0]['r_per_episode']))
pd_5_stdev = np.std(np.array(pd_5['logbook']['data'][0]['r_per_episode']))
pd_6_mean = np.mean(np.array(pd_6['logbook']['data'][0]['r_per_episode']))
pd_6_stdev = np.std(np.array(pd_6['logbook']['data'][0]['r_per_episode']))
pd_7_mean = np.mean(np.array(pd_7['logbook']['data'][0]['r_per_episode']))
pd_7_stdev = np.std(np.array(pd_7['logbook']['data'][0]['r_per_episode']))
pd_8_mean = np.mean(np.array(pd_8['logbook']['data'][0]['r_per_episode']))
pd_8_stdev = np.std(np.array(pd_8['logbook']['data'][0]['r_per_episode']))

# Plot data
objects = ('1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0')
y_pos = np.arange(len(objects))
performance = [pd_1_mean, pd_2_mean, pd_3_mean, pd_4_mean, pd_5_mean, pd_6_mean, pd_7_mean, pd_8_mean]
error = [pd_1_stdev, pd_2_stdev, pd_3_stdev, pd_4_stdev, pd_5_stdev, pd_6_stdev, pd_7_stdev, pd_8_stdev]
plt.barh(y_pos, performance, xerr=error, capsize=5.0, align='center')
plt.yticks(y_pos, objects)
plt.xlabel('Average Reward per Simulation Step')
plt.ylabel('Performance Parameter')
plt.title('PD Controller Relative Performance, N=500')
plt.gcf().set_size_inches(8.5, 5.5)
plt.savefig('results/PD_Results/performance.png', dpi = 500)
plt.close()