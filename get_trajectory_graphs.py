# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 12:33:49 2021

@author: Grayson Schaer
"""

import Finite_Element_Solver_2D as fes_2
import PPO_Agent_3_Output as ppo_3
import Finite_Element_Solver_1D as fes_1 
import PPO_Agent_2_Output as ppo_2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load data
with open("results/PPO-Results/output_1c", 'rb') as file:
    data = pickle.load(file)  
Control_1D = np.array(data['logbook']['data'][0]['front_velocity'])
Control_1D_Time = np.array(data['logbook']['data'][0]['time'])

with open("results/PPO-Results/output_1p", 'rb') as file:
    data = pickle.load(file)  
PPO_1D = np.array(data['logbook']['data'][0]['front_velocity'])

with open("results/PPO-Results/output_2c", 'rb') as file:
    data = pickle.load(file)  
Control_2D = np.array(data['logbook']['data'][0]['front_velocity'])
Control_2D_Time = np.array(data['logbook']['data'][0]['time'])

with open("results/PPO-Results/output_2p", 'rb') as file:
    data = pickle.load(file)  
PPO_2D = np.array(data['logbook']['data'][0]['front_velocity'])

# Plot 1D Trajectory
plt.clf()
plt.title("Front Velocity in 1D", fontsize='x-large')
plt.xlabel("Simulation Time [s]", size='x-large')
plt.ylabel("Front Velocity [mm/s]", size='x-large')
plt.plot(Control_1D_Time, 1000.0*Control_1D, c='b', zorder=-100, linewidth=2.0)
plt.plot(Control_1D_Time, 1000.0*PPO_1D, c='r', zorder=-50, linewidth=2.0)
plt.hlines(0.15, 0.0, 360.0, colors='k', linestyles='dashed', linewidth=2.5)
plt.legend(('Uncontrolled', 'Controlled', 'Target'),loc='lower right', fontsize='x-large')
plt.ylim(0.0, 0.20)
plt.xlim(0.0, 360.0)
plt.gca().tick_params(axis='both', which='major', labelsize=13)
plt.gcf().set_size_inches(7.50, 5.0)
plt.savefig('results/data/1D.png', dpi = 500)
plt.close()

# Plot 2D Trajectory
plt.clf()
plt.title("Front Velocity in 2D", fontsize='x-large')
plt.xlabel("Simulation Time [s]", size='x-large')
plt.ylabel("Front Velocity [mm/s]", size='x-large')
plt.plot(Control_2D_Time, 1000.0*np.mean(Control_2D,axis=1), c='b', zorder=-100, linewidth=2.0)
plt.plot(Control_2D_Time, 1000.0*np.mean(PPO_2D,axis=1), c='r', zorder=-50, linewidth=2.0)
plt.hlines(0.15, 0.0, 360.0, colors='k', linestyles='dashed', linewidth=2.5)
plt.legend(('Uncontrolled', 'Controlled', 'Target'),loc='lower right', fontsize='x-large')
plt.ylim(0.0, 0.20)
plt.xlim(0.0, 360.0)
plt.gca().tick_params(axis='both', which='major', labelsize=13)
plt.gcf().set_size_inches(7.50, 5.0)
plt.savefig('results/data/2D.png', dpi = 500)
plt.close()