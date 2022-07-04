# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:10:18 2022

@author: GKSch
"""

import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(0.07,0.15,9)
T = np.linspace(281.15,309.15,8)
v = np.transpose(np.array([[0.69,0.80,0.92,1.06,1.22,1.41,1.64,1.94],
[0.64,0.74,0.85,0.99,1.14,1.31,1.51,1.76],
[0.59,0.69,0.79,0.91,1.06,1.22,1.40,1.62],
[0.55,0.64,0.74,0.85,0.98,1.13,1.30,1.50],
[0.51,0.59,0.68,0.79,0.91,1.05,1.21,1.40],
[0.47,0.55,0.63,0.73,0.85,0.98,1.13,1.30],
[0.44,0.51,0.59,0.68,0.79,0.91,1.05,1.21],
[0.40,0.47,0.54,0.63,0.73,0.84,0.97,1.12],
[0.37,0.43,0.50,0.58,0.68,0.78,0.90,1.04]]))

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
fig.set_size_inches(10,12)
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

a_fine = np.linspace(0.07,0.15,90)
T_fine = np.linspace(281.15,309.15,80)
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