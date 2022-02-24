# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:25:12 2022

@author: GKSch
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

front_width = 8.400008400008401

with open("../results/SIM_1/output", 'rb') as file:
    dat = pickle.load(file)

pos = dat['global_fine_mesh_x'][:,0]
time = dat['time']
temp = dat['interpolated_temp_field'][:,:,len(dat['interpolated_temp_field'][0,0,:])//2]
cure = dat['interpolated_cure_field'][:,:,len(dat['interpolated_cure_field'][0,0,:])//2]
a0 = np.round(np.mean(dat['cure_field'][0]),2)
T0 = dat['initial_temperature'] - 273.15

t_prof = []
c_prof = []
p_prof = []
speed = []
radius = (np.argmin(abs(pos-2e-3))+1)//2
for i in [28.0]:
    
    ind = np.argmin(abs(time - i))
    mid_ind = int(np.round(0.50*(np.argmax(abs(np.diff(cure[ind])))+0.5 + np.argmax(abs(np.diff(temp[ind]))))))
    
    if front_width == 0.0:
        left_ind = np.argmin(abs(cure[ind]-0.99))
        right_ind = 2*mid_ind-left_ind
    else:
        left_ind = np.argmin(abs(pos - (pos[mid_ind]-5.0e-4*front_width)))
        right_ind = np.argmin(abs(pos - (pos[mid_ind]+5.0e-4*front_width)))
    
    t_prof.append(temp[ind][left_ind:right_ind])
    c_prof.append(cure[ind][left_ind:right_ind])
    p_prof.append(1000.0*(pos[left_ind:right_ind]-pos[left_ind]))
    speed.append(dat['front_velocity'][ind])
  
speed = 1000.0*np.mean(np.array(speed))
plt.clf()
plt.gcf().set_size_inches(8.5, 5.5)
title_str = "PDE Front Profile: T0 = " + str(T0) + " °C, α0 = " + str(a0)
plt.title(title_str,fontsize='xx-large')
plt.xlabel("Position [mm]",fontsize='large')
plt.ylabel("θ, α [-]",fontsize='large')

for i in range(len(t_prof)):
    if i == 0:
        plt.plot(p_prof[i], t_prof[i], color='red', label='θ', lw=2.5)
        plt.plot(p_prof[i], c_prof[i], color='blue', label='α', lw=2.5)
        text_str = str(np.round(speed,3)) + " mm/s"
        plt.text((np.mean(p_prof[-1]-p_prof[-1][0]) + 0.05*(np.max(p_prof[-1]-p_prof[-1][0])-np.min(p_prof[-1]-p_prof[-1][0]))), 0.5, text_str, fontsize='x-large')
        plt.arrow((np.mean(p_prof[-1]-p_prof[-1][0]) + 0.05*(np.max(p_prof[-1]-p_prof[-1][0])-np.min(p_prof[-1]-p_prof[-1][0]))), 0.45, 1, 0, width=0.01, head_length=0.125, fc='k')
    else:
        plt.plot(p_prof[i], t_prof[i], color='red', lw=2.5)
        plt.plot(p_prof[i], c_prof[i], color='blue', lw=2.5)
    
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.legend(loc='upper right',fontsize='large')
plt.tight_layout()
plt.savefig("../results/PDE_Profiles.png", dpi = 500)
    
# t = np.mean(np.array(t_prof),axis=0)
# c = np.mean(np.array(c_prof),axis=0)
# pos = 1000.0*pos[0:2*radius]