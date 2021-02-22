# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 09:34:49 2021

@author: GKSch
"""
import Finite_Element_Solver_3D as fes 
import PPO_Agent_3_Output as ppo
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import signal
from scipy.stats import multivariate_normal

with open('results/PPO_1/output', 'rb') as file:
    input_data = pickle.load(file)
data = input_data['data']
env = input_data['env']
agent = input_data['agent']
video_path = 'results/PPO_1/video/'
dpi = 100

# Make videos of the best temperature field trajecotry and cure field trajectories as function of time
print("Rendering...")
min_temp = 0.99*np.min(data['temperature_field'])-273.15
max_temp = 1.01*np.max(data['temperature_field'])-273.15

# Determine front shape deltas
front_mean_loc = np.mean(100.0*np.array(data['front_location']),axis=(1,2))
min_loc = 1.025*min(np.min(100.0*np.array(data['front_location']),axis=(1,2)) - front_mean_loc)
max_loc = 1.025*max(np.max(100.0*np.array(data['front_location']),axis=(1,2)) - front_mean_loc)

# Determine front speed deltas
max_vel = 1.025*(np.max(100.0*np.array(data['front_velocity'])))

# Make custom color map for normalized data
min_round = 10.0*round(min_temp//10.0)
max_round = 10.0*round(max_temp/10.0)
for curr_step in range(len(data['time'])):
       
    # Calculate input field
    input_percent = data['input_percent'][curr_step]
    input_location = data['input_location'][curr_step]
    input_mesh = input_percent*env.max_input_mag*np.exp(((env.mesh_x[:,:,0]-input_location[0])**2*env.exp_const) + 
                                                          (env.mesh_y[:,:,0]-input_location[1])**2*env.exp_const)
    input_mesh[input_mesh<0.01*env.max_input_mag] = 0.0
    
    # Make fig for temperature, cure, and input
    plt.cla()
    plt.clf()
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
    fig.set_size_inches(11,8.5)
    
    # Plot temperature
    c0 = ax0.pcolormesh(100.0*env.mesh_x[:,:,0], 100.0*env.mesh_y[:,:,0], np.array(data['temperature_field'][curr_step])-273.15, shading='gouraud', cmap='jet', vmin=min_round, vmax=max_round)
    cbar0 = fig.colorbar(c0, ax=ax0)
    cbar0.set_label('Temperature [C]',labelpad=20,fontsize='large')
    cbar0.ax.tick_params(labelsize=12)
    ax0.set_xlabel('X Position [cm]',fontsize='large')
    ax0.set_ylabel('Y Position [cm]',fontsize='large')
    ax0.tick_params(axis='x',labelsize=12)
    ax0.tick_params(axis='y',labelsize=12)
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_title('Max Temperature = '+'{:.2f}'.format(np.max(data['temperature_field'][curr_step]-273.15))+' C',fontsize='large')
    
    # Plot cure
    c1 = ax1.pcolormesh(100.0*env.mesh_x[:,:,0], 100.0*env.mesh_y[:,:,0], data['cure_field'][curr_step], shading='gouraud', cmap='YlOrBr', vmin=0.0, vmax=1.0)
    cbar1 = fig.colorbar(c1, ax=ax1)
    cbar1.set_label('Degree Cure [-]', labelpad=20,fontsize='large')
    cbar1.ax.tick_params(labelsize=12)
    ax1.set_xlabel('X Position [cm]',fontsize='large')
    ax1.set_ylabel('Y Position [cm]',fontsize='large')
    ax1.tick_params(axis='x',labelsize=12)
    ax1.tick_params(axis='y',labelsize=12)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot input
    c2 = ax2.pcolormesh(100.0*env.mesh_x[:,:,0], 100.0*env.mesh_y[:,:,0], 1.0e-3*input_mesh, shading='gouraud', cmap='coolwarm', vmin=0.0, vmax=1.0e-3*env.max_input_mag)
    ax2.plot(100.0*np.array(data['front_location'][curr_step][:,0]).reshape(env.num_vert_width,1), 100.0*env.mesh_y[0,:,0], 'k-', lw=1.5)
    cbar2 = fig.colorbar(c2, ax=ax2)
    cbar2.set_label('Input Heat [KW/m^2]',labelpad=20,fontsize='large')
    cbar2.ax.tick_params(labelsize=12)
    ax2.set_xlabel('X Position [cm]',fontsize='large')
    ax2.set_ylabel('Y Position [cm]',fontsize='large')
    ax2.tick_params(axis='x',labelsize=12)
    ax2.tick_params(axis='y',labelsize=12)
    ax2.set_aspect('equal', adjustable='box')
    
    # Set title and save
    title_str = "Time From Trigger: "+'{:.2f}'.format(data['time'][curr_step])+'s'
    fig.suptitle(title_str,fontsize='xx-large')
    plt.savefig(video_path+str(curr_step).zfill(4)+'.png', dpi=dpi)
    plt.close()
    
    # Make fig for front location and velocity
    plt.cla()
    plt.clf()
    fig, (ax0, ax1) = plt.subplots(1,2,subplot_kw={"projection": "3d"})
    fig.set_size_inches(14.0,8.0)
    
    # Convolve front location data
    back_msaa_index = np.clip(curr_step-5,0,len(data['time'])-1)
    front_msaa_index = np.clip(curr_step+5,0,len(data['time'])-1)
    front_delta_loc = np.mean(100.0*np.array(data['front_location'][back_msaa_index:front_msaa_index]),axis=0) - np.mean(front_mean_loc[back_msaa_index:front_msaa_index])
    front_delta_min = np.min(front_delta_loc)
    front_delta_max = np.max(front_delta_loc)
    if not ((front_delta_loc<=1.0e-5).all() and (front_delta_loc>=-1.0e-5).all()):
        dim = 21
        x,y=np.meshgrid(np.linspace(-1,1,dim),np.linspace(-1,1,dim))
        win=multivariate_normal.pdf(np.dstack((x,y)),mean=[0,0],cov=[[1.0,0.0],[0.0,1.0]])
        padded = front_delta_loc
        for i in range(int((dim+1)/2)-1):
            padded = np.append(padded[:,0].reshape(len(padded[:,0]),1),padded,axis=1)
            padded = np.append(padded[0,:].reshape(1,len(padded[0,:])),padded,axis=0)
            padded = np.append(padded,padded[:,-1].reshape(len(padded[:,-1]),1),axis=1)
            padded = np.append(padded,padded[-1,:].reshape(1,len(padded[-1,:])),axis=0)
        out = signal.convolve2d(padded,win,mode='valid')
        out=out*((front_delta_max-front_delta_min)/(np.max(out)-np.min(out)))
        out=out-np.mean(out)
    else:
        out = front_delta_loc

    # Plot front location
    surf = ax0.plot_surface(100.0*env.mesh_y[0,:,:], 100.0*env.mesh_z[0,:,:],out,cmap='coolwarm',vmin=min_loc,vmax=max_loc,alpha=1.0)
    ax0.set_xlabel('Y Position [cm]',fontsize='large',labelpad=15)
    ax0.set_ylabel('Z Position [cm]',fontsize='large',labelpad=15)
    ax0.set_zlabel('Lengthwise Delta [cm]',fontsize='large',labelpad=20)
    ax0.tick_params(axis='x',labelsize=12,pad=10)
    ax0.tick_params(axis='y',labelsize=12,pad=10)
    ax0.tick_params(axis='z',labelsize=12,pad=10)
    ax0.set_zlim(min_loc,max_loc)
    ax0.set_title("Front Shape",fontsize='xx-large')
    
    # Covolve front speed data
    back_msaa_index = np.clip(curr_step-5,0,len(data['time'])-1)
    front_msaa_index = np.clip(curr_step+5,0,len(data['time'])-1)
    curr_front_vel = np.mean(100.0*np.array(data['front_velocity'][back_msaa_index:front_msaa_index]),axis=0)
    front_vel_min = np.min(curr_front_vel)
    front_vel_max = np.max(curr_front_vel)
    if not ((curr_front_vel<=1.0e-5).all() and (curr_front_vel>=-1.0e-5).all()):
        dim = 21
        x,y=np.meshgrid(np.linspace(-1,1,dim),np.linspace(-1,1,dim))
        win=multivariate_normal.pdf(np.dstack((x,y)),mean=[0,0],cov=[[1.0,0.0],[0.0,1.0]])
        padded = curr_front_vel
        for i in range(int((dim+1)/2)-1):
            padded = np.append(padded[:,0].reshape(len(padded[:,0]),1),padded,axis=1)
            padded = np.append(padded[0,:].reshape(1,len(padded[0,:])),padded,axis=0)
            padded = np.append(padded,padded[:,-1].reshape(len(padded[:,-1]),1),axis=1)
            padded = np.append(padded,padded[-1,:].reshape(1,len(padded[-1,:])),axis=0)
        out = signal.convolve2d(padded,win,mode='valid')
        out=out*((front_vel_max-front_vel_min)/(np.max(out)-np.min(out)))
        out=out-np.mean(out)+np.mean(curr_front_vel)
    else:
        out = curr_front_vel
    
    # Plot front speed
    surf = ax1.plot_surface(100.0*env.mesh_y[0,:,:],100.0*env.mesh_z[0,:,:],out,cmap='coolwarm',vmin=0.0,vmax=max_vel,alpha=1.0)
    ax1.set_xlabel('Y Position [cm]',fontsize='large',labelpad=15)
    ax1.set_ylabel('Z Position [cm]',fontsize='large',labelpad=15)
    ax1.set_zlabel('Front Speed [cm/s]',fontsize='large',labelpad=20)
    ax1.tick_params(axis='x',labelsize=12,pad=10)
    ax1.tick_params(axis='y',labelsize=12,pad=10)
    ax1.tick_params(axis='z',labelsize=12,pad=10)
    ax1.set_zlim(0.0,max_vel)
    ax1.set_title("Front Speed",fontsize='xx-large')
    
    # Set title and save
    title_str = "Time From Trigger: "+'{:.2f}'.format(data['time'][curr_step])+'s'
    fig.suptitle(title_str,fontsize='xx-large')
    plt.savefig(video_path+"f_"+str(curr_step).zfill(4)+'.png', dpi=dpi)
    plt.close()