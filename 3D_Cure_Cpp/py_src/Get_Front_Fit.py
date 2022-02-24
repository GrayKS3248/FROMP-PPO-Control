# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:10:18 2022

@author: GKSch
"""

import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(0.01,0.15,15)
T = np.linspace(8,36,8)
v = np.transpose(np.array([[0.959, 1.103, 1.265, 1.448, 1.655, 1.888, 2.151, 2.446],
[0.897, 1.032, 1.185, 1.359, 1.556, 1.777, 2.027, 2.308],
[0.836, 0.964, 1.108, 1.273, 1.459, 1.669, 1.906, 2.173],
[0.778, 0.898, 1.035, 1.189, 1.365, 1.563, 1.788, 2.041],
[0.724, 0.836, 0.964, 1.110, 1.276, 1.463, 1.675, 1.914],
[0.672, 0.777, 0.897, 1.035, 1.190, 1.367, 1.567, 1.793],
[0.623, 0.722, 0.834, 0.963, 1.109, 1.276, 1.464, 1.678],
[0.577, 0.669, 0.775, 0.895, 1.033, 1.189, 1.367, 1.568],
[0.534, 0.620, 0.719, 0.832, 0.961, 1.107, 1.274, 1.464],
[0.493, 0.574, 0.666, 0.772, 0.893, 1.030, 1.187, 1.365],
[0.456, 0.530, 0.617, 0.715, 0.828, 0.957, 1.104, 1.272],
[0.420, 0.490, 0.571, 0.663, 0.768, 0.889, 1.027, 1.184],
[0.387, 0.452, 0.527, 0.613, 0.712, 0.825, 0.954, 1.101],
[0.357, 0.417, 0.487, 0.567, 0.659, 0.765, 0.885, 1.023],
[0.328, 0.384, 0.449, 0.524, 0.609, 0.708, 0.821, 0.950]]))

a_mesh, T_mesh = np.meshgrid(a, T)
a = a_mesh.flatten()
T = T_mesh.flatten()

A = np.transpose(np.array([a*0+1, a, T, a**2, a*T, T**2, a**2*T, a*T**2]))
B = v.flatten()

coeff, r, rank, s = np.linalg.lstsq(A, B,rcond=None)
fit = coeff[0]*(a_mesh*0+1) + coeff[1]*(a_mesh) + coeff[2]*(T_mesh) + coeff[3]*(a_mesh**2) + coeff[4]*(a_mesh*T_mesh) + coeff[5]*(T_mesh**2) + coeff[6]*(a_mesh**2*T_mesh) + coeff[7]*(a_mesh*T_mesh**2)

rms_err = np.sqrt(np.mean((v-fit)**2))
norm_rms_err = rms_err / (np.max(v) - np.min(v))

plt.cla()
plt.clf()
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
fig.set_size_inches(12,8.5)
min_v = np.round((np.min(v) // 0.1) * 0.1,1)
max_v = np.round((np.max(v) // 0.1) * 0.1 + 0.1,1)

c0 = ax0.pcolormesh(a_mesh, T_mesh, v, shading='nearest', cmap='jet', vmin=min_v, vmax=max_v)
cbar0 = fig.colorbar(c0, ax=ax0)
cbar0.set_label("Front Speed [mm/s]",labelpad=20,fontsize='large')
cbar0.ax.tick_params(labelsize=12)
ax0.set_xlabel('α0 [-]',fontsize='large')
ax0.set_ylabel('T0 [°C]',fontsize='large')
ax0.tick_params(axis='x',labelsize=12)
ax0.tick_params(axis='y',labelsize=12)
ax0.set_title("ODE Front Speed",fontsize='large')
ax0.set_aspect(0.0009, adjustable='box')

a_fine = np.linspace(0.01,0.15,150)
T_fine = np.linspace(8,36,80)
a_fine_mesh, T_fine_mesh = np.meshgrid(a_fine, T_fine)
fit_fine = coeff[0]*(a_fine_mesh*0+1) + coeff[1]*(a_fine_mesh) + coeff[2]*(T_fine_mesh) + coeff[3]*(a_fine_mesh**2) + coeff[4]*(a_fine_mesh*T_fine_mesh) + coeff[5]*(T_fine_mesh**2) + coeff[6]*(a_fine_mesh**2*T_fine_mesh) + coeff[7]*(a_fine_mesh*T_fine_mesh**2)
c1 = ax1.pcolormesh(a_fine_mesh, T_fine_mesh, fit_fine, shading='gouraud', cmap='jet', vmin=min_v, vmax=max_v)
cbar1 = fig.colorbar(c1, ax=ax1)
cbar1.set_label("Front Speed [mm/s]",labelpad=20,fontsize='large')
cbar1.ax.tick_params(labelsize=12)
ax1.set_xlabel('α0 [-]',fontsize='large')
ax1.set_ylabel('T0 [°C]',fontsize='large')
ax1.tick_params(axis='x',labelsize=12)
ax1.tick_params(axis='y',labelsize=12)
ax1.set_title("Fit Front Speed",fontsize='large')
ax1.set_aspect(0.0009, adjustable='box')        

c2 = ax2.pcolormesh(a_mesh, T_mesh, fit-v, shading='gouraud', cmap='coolwarm', vmin=np.round((np.min(fit-v) // 0.001) * 0.001,3), vmax=np.round((np.max(fit-v) // 0.001) * 0.001+0.001,3))
cbar2 = fig.colorbar(c2, ax=ax2)
cbar2.set_label("Error [mm/s]",labelpad=20,fontsize='large')
cbar2.ax.tick_params(labelsize=12)
ax2.set_xlabel('α0 [-]',fontsize='large')
ax2.set_ylabel('T0 [°C]',fontsize='large')
ax2.tick_params(axis='x',labelsize=12)
ax2.tick_params(axis='y',labelsize=12)
ax2.set_title("Error",fontsize='large')
ax2.set_aspect(0.0009, adjustable='box')

# Set title and save
fit_str = "{:.2f}".format(coeff[0]) + ' + ' + "{:.2f}".format(coeff[1]) +'(α) + '+"{:.2e}".format(coeff[2]) +'(T) + '+"{:.2f}".format(coeff[3]) +'(α²) + '+"{:.2f}".format(coeff[4]) +'(αT) + '+"{:.2e}".format(coeff[5]) +'(T²) + '+"{:.2f}".format(coeff[6]) +'(α²T) + '+"{:.2e}".format(coeff[7]) +'(αT²)'
err_str = "Range Normalized RMS Error = " + str(np.round(norm_rms_err*100.0,2))+"%"
fig.suptitle(fit_str+"\n"+err_str,fontsize='large')
plt.savefig('../results/speed_fit.png', dpi=500)
plt.close()