# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:50:34 2020

@author: Grayson Schaer
"""
import Finite_Element_Solver_3D as fes 
import PPO_Agent_3_Output as ppo
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import signal
from scipy.stats import multivariate_normal

def run(env, agent, total_trajectories, execution_rate, frame_multiplier):

    # Create dict to store data from simulation
    data = {
        'r_per_episode' : [],
        'value_error' : [],
        'x_rate_stdev': [],
        'y_rate_stdev': [],
        'mag_stdev': [],
        'input_location': [],
        'input_percent':[],
        'temperature_field': [],
        'cure_field': [],
        'front_location': [],
        'front_velocity': [],
        'target_velocity': [],
        'time': [],
        'best_episode': [],
    }
    trajectory = {
        'input_location': [],
        'input_percent':[],
        'temperature_field': [],
        'cure_field': [],
        'front_location': [],
        'front_velocity': [],
        'target_velocity': [],
        'time': [],
        }
    
    # Create variables used to keep track of learning
    r_total = 0.0
    curr_step = 0
    best_episode = 0.0
    
    # Run a set of episodes
    for curr_episode in range(total_trajectories):
        
        # # User readout parameters
        percent_complete = curr_episode / total_trajectories
        
        # User readout
        if curr_episode >= 1:
            print_str = (('{:03.1f}'.format(100.0 * percent_complete) + "% Complete").ljust(16) + 
                ("| Traj: " + str(curr_episode+1) + "/" + str(total_trajectories)).ljust(20) + 
                ("| R/Step: " + '{:.2f}'.format(data['r_per_episode'][-1])).ljust(16) + 
                ("| Avg_R/Step: " + '{:.2f}'.format(np.mean(data['r_per_episode'][-100:]))).ljust(20) + 
                ("| Best R: " + '{:.1f}'.format(best_episode)).ljust(17) + 
                ("| Avg R: " + '{:.1f}'.format(np.mean(data['r_per_episode'][-100:])*agent.steps_per_trajectory)).ljust(16) + 
                "|")
            print(print_str, end="\r", flush=True)
        else:
            print_str = (('{:03.1f}'.format(100.0 * percent_complete) + "% Complete").ljust(16) + 
                ("| Traj: " + str(curr_episode+1) + "/" + str(total_trajectories)).ljust(20) + 
                ("| R/Step: " + '{:.2f}'.format(0.0)).ljust(16) + 
                ("| Avg_R/Step: " + '{:.2f}'.format(0.0)).ljust(20) + 
                ("| Best R: " + '{:.1f}'.format(best_episode)).ljust(17) + 
                ("| Avg R: " + '{:.1f}'.format(0.0)).ljust(16) + 
                "|")
            print(print_str, end="\r", flush=True)
        
        # Initialize simulation
        s = env.reset()
        episode_reward = r_total
                
        # Simulate until episode is done
        done = False
        step_in_episode = 0
        while not done:
            
            # Get action, do action, learn
            if step_in_episode % execution_rate == 0:
                x_loc_rate_action, x_loc_rate_stdev, y_loc_rate_action, y_loc_rate_stdev, mag_action, mag_stdev = agent.get_action(s)
            (s2, r, done) = env.step(np.array([x_loc_rate_action, y_loc_rate_action, mag_action]))
            if step_in_episode % execution_rate == 0:
                agent.update_agent(s, np.array([x_loc_rate_action, y_loc_rate_action, mag_action]), r)
                r_total = r_total + r
                curr_step = curr_step + 1
            
            # Update logs
            if step_in_episode % int(frame_multiplier*execution_rate) == 0 or done:
                trajectory['input_location'].append(np.copy(env.input_location))
                trajectory['temperature_field'].append(np.copy(env.temp_mesh[:,:,0]))
                trajectory['input_percent'].append(np.copy(env.input_percent))
                trajectory['cure_field'].append(np.copy(env.cure_mesh[:,:,0]))
                trajectory['front_location'].append(np.copy(env.front_loc))
                trajectory['front_velocity'].append(np.copy(env.front_vel))
                trajectory['target_velocity'].append(np.copy(env.current_target_front_vel))
                trajectory['time'].append(np.copy(env.current_time))
            
            # Update state and step
            s = s2
            step_in_episode = step_in_episode + 1
    
        # Calculate reward per epoch
        episode_reward = r_total - episode_reward
        if episode_reward > best_episode or curr_episode == 0:
            best_episode = episode_reward
            data['input_location'] = trajectory['input_location']
            data['temperature_field'] = trajectory['temperature_field']
            data['input_percent'] = trajectory['input_percent']
            data['cure_field'] = trajectory['cure_field']
            data['front_location'] = trajectory['front_location']
            data['front_velocity'] = trajectory['front_velocity']
            data['target_velocity'] = trajectory['target_velocity']
            data['time'] = trajectory['time']
            data['best_episode'] = episode_reward
        
        # Update the logs
        data['r_per_episode'].append(episode_reward / agent.steps_per_trajectory)
        data['x_rate_stdev'].append(x_loc_rate_stdev)
        data['y_rate_stdev'].append(y_loc_rate_stdev)
        data['mag_stdev'].append(mag_stdev)
        
        # Reset the trajectory memory
        trajectory['input_location'] = []
        trajectory['temperature_field'] = []
        trajectory['input_percent'] = []
        trajectory['cure_field'] = []
        trajectory['front_location'] = []
        trajectory['front_velocity'] = []
        trajectory['target_velocity'] = []
        trajectory['time'] = []
   
    # Store the training data
    data['value_error'].append(agent.value_estimation_error)
        
    # User readout
    percent_complete = 1.0
    print_str = (('{:03.1f}'.format(100.0 * percent_complete) + "% Complete").ljust(16) + 
        ("| Traj: " + str(curr_episode+1) + "/" + str(total_trajectories)).ljust(20) + 
        ("| R/Step: " + '{:.2f}'.format(data['r_per_episode'][-1])).ljust(16) + 
        ("| Avg_R/Step: " + '{:.2f}'.format(np.mean(data['r_per_episode'][-100:]))).ljust(20) + 
        ("| Best R: " + '{:.1f}'.format(best_episode)).ljust(17) + 
        ("| Avg R: " + '{:.1f}'.format(np.mean(data['r_per_episode'][-100:])*agent.steps_per_trajectory)).ljust(16) + 
        "|")
    print(print_str, end="\n", flush=True)
    
    return data, agent, env

if __name__ == '__main__':
    
    # Simulation parameters
    load_previous_agent = False
    reset_stdev = False
        
    # Agent parameters
    total_trajectories = 1
    steps_per_trajecotry = 240
    trajectories_per_batch = 10
    num_epochs = 10
    gamma = 0.99
    lamb = 0.95
    epsilon = 0.20
    start_alpha = 5.0e-4
    end_alpha = 4.0e-4
    
    # Rendering parameters
    frame_multiplier = 1.0/6.0
    dpi = 100
    
    # Calculated env and agent parameters
    env = fes.FES()
    num_states = ((env.num_vert_length)//6)*((env.num_vert_width)//4) + 25 + 2*((env.num_vert_width)//4) + 3
    decay_rate = (end_alpha/start_alpha)**(trajectories_per_batch/total_trajectories)
    agent_temporal_precision = (env.sim_duration / float(steps_per_trajecotry))
    execution_rate = int(agent_temporal_precision / env.time_step)
    minibatch_size = (trajectories_per_batch * steps_per_trajecotry) // num_epochs

    # Check inputs
    if ((int(agent_temporal_precision / env.time_step) - (agent_temporal_precision / env.time_step))!=0):
        raise RuntimeError("Agent execution rate is not multiple of simulation rate")
   
    # Find save paths
    done = False
    curr_folder = 1
    while not done:
        path = "results/PPO_"+str(curr_folder)
        video_path = "results/PPO_"+str(curr_folder)+"/video/"
        if not os.path.isdir(path):
            os.mkdir(path)
            os.mkdir(video_path)
            done = True
        else:
            curr_folder = curr_folder + 1
            
    # Find load path
    if load_previous_agent:
        load_path = "results/PPO_"+str(curr_folder-1)
        if not os.path.isdir(load_path):
            raise RuntimeError("Could not find previous folder: " + load_path)
        load_path = load_path + "/output"
        if not os.path.exists(load_path):
            raise RuntimeError("Could not find previous output file: " + load_path)
        with open(load_path, 'rb') as file:
            input_data = pickle.load(file)
        if 'agent' in input_data:
            old_agent = input_data['agent']
        else:
            old_agent = input_data['logbook']['agents'][0]

    # Create agents, run simulation, save results
    agent = ppo.PPO_Agent(num_states, steps_per_trajecotry, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, start_alpha, decay_rate)
    if load_previous_agent:
        print("Loading previous agent data...")
        agent.copy(old_agent, reset_stdev)
    print("Training agent...")
    data, agent, env = run(env, agent, total_trajectories, execution_rate, frame_multiplier)
    
    # Pickle all important outputs
    print("Saving...")
    output = { 'data':data, 'agent':agent, 'env':env }
    save_file = path + "/output"
    with open(save_file, 'wb') as file:
        pickle.dump(output, file)
        
    # Plot the trajectory
    print("Plotting...")
    plt.clf()
    plt.title("Front Velocity",fontsize='xx-large')
    plt.xlabel("Simulation Time [s]",fontsize='large')
    plt.ylabel("Front Velocity [mm/s]",fontsize='large')
    plt.plot(data['time'], 1000.0*np.array(np.mean(np.mean(data['front_velocity'],axis=1),axis=1)),c='k',lw=2.0)
    plt.plot(data['time'], 1000.0*np.array(data['target_velocity']),c='b',ls='--',lw=2.0)
    plt.legend(('Actual','Target'),loc='lower right',fontsize='large')
    plt.ylim(0.0, max(1.25*1000.0*np.array(data['target_velocity'])))
    plt.xlim(0.0, env.sim_duration)
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.gcf().set_size_inches(8.5, 5.5)
    save_file = path + "/trajectory.png"
    plt.savefig(save_file, dpi = 500)
    plt.close()
        
    #Plot actor learning curve
    plt.clf()
    plt.title("Actor Learning Curve, Episode-Wise",fontsize='xx-large')
    plt.xlabel("Episode",fontsize='large')
    plt.ylabel("Average Reward per Simulation Step",fontsize='large')
    plt.plot([*range(len(data['r_per_episode']))],data['r_per_episode'],lw=2.0,c='r')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.gcf().set_size_inches(8.5, 5.5)
    save_file = path + "/actor_learning.png"
    plt.savefig(save_file, dpi = 500)
    plt.close()

    # Plot value learning curve
    plt.clf()
    title_str = "Critic Learning Curve"
    plt.title(title_str,fontsize='xx-large')
    plt.xlabel("Optimization Step",fontsize='large')
    plt.ylabel("MSE Loss",fontsize='large')
    plt.plot([*range(len(data['value_error'][0]))],data['value_error'][0],lw=2.0,c='r')
    plt.yscale("log")
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.gcf().set_size_inches(8.5, 5.5)
    save_file = path + "/critic_learning.png"
    plt.savefig(save_file, dpi = 500)
    plt.close()

    # Plot x rate stdev curve
    plt.clf()
    plt.title("Laser X Position Rate Stdev",fontsize='xx-large')
    plt.xlabel("Episode",fontsize='large')
    plt.ylabel("Laser X Position Rate Stdev [m/s]",fontsize='large')
    plt.plot([*range(len(data['x_rate_stdev']))],env.loc_rate_scale*np.array(data['x_rate_stdev']),lw=2.0,c='r')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.gcf().set_size_inches(8.5, 5.5)
    save_file = path + "/x_rate_stdev.png"
    plt.savefig(save_file, dpi = 500)
    plt.close()

    # Plot y rate stdev curve
    plt.clf()
    plt.title("Laser Y Position Rate Stdev",fontsize='xx-large')
    plt.xlabel("Episode",fontsize='large')
    plt.ylabel("Laser Y Position Rate Stdev [m/s]",fontsize='large')
    plt.plot([*range(len(data['y_rate_stdev']))],env.loc_rate_scale*np.array(data['y_rate_stdev']),lw=2.0,c='r')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.gcf().set_size_inches(8.5, 5.5)
    save_file = path + "/y_rate_stdev.png"
    plt.savefig(save_file, dpi = 500)
    plt.close()

    # Plot magnitude stdev curve
    plt.clf()
    plt.title("Laser Magnitude Stdev",fontsize='xx-large')
    plt.xlabel("Episode",fontsize='large')
    plt.ylabel('Laser Magnitude Stdev [K/s]',fontsize='large')
    plt.plot([*range(len(data['mag_stdev']))],env.mag_scale*env.max_input_mag*np.array(data['mag_stdev']),lw=2.0,c='r')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.gcf().set_size_inches(8.5, 5.5)
    save_file = path + "/mag_stdev.png"
    plt.savefig(save_file, dpi = 500)
    plt.close()
    
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
            dim = 5
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
            dim = 5
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
    
    print("Done!")