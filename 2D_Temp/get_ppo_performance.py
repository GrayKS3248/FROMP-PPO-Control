# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:50:34 2020

@author: Grayson Schaer
"""
import Finite_Element_Solver_2D as fes 
import PPO_Agent_3_Output as ppo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import pickle
import os
import shutil

def run(env, agent, total_trajectories, execution_rate, frame_multiplier):

    # Create dict to store data from simulation
    data = {
        'r_per_episode' : [],
        'input_location': [],
        'input_magnitude':[],
        'temperature_field': [],
        'temperature_target': [],
        'temperature_rel_error': [],
        'temperature_max_error': [],
        'temperature_min_error': [],
        'time': [],
        'best_episode': [],
    }
    trajectory = {
        'input_location': [],
        'input_magnitude':[],
        'temperature_field': [],
        'temperature_target': [],
        'temperature_rel_error': [],
        'temperature_max_error': [],
        'temperature_min_error': [],
        'time': [],
        }
    
    # Create variables used to keep track of learning
    r_total = 0
    episode_reward = 0
    curr_step = 0
    percent_complete = 0.0
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
                x_loc_rate_action, y_loc_rate_action, mag_action = agent.get_greedy_action(s)
            (s2, r, done) = env.step(np.array([x_loc_rate_action, y_loc_rate_action, mag_action]))
            if step_in_episode % execution_rate == 0:
                r_total = r_total + r
                curr_step = curr_step + 1
            
            # Update logs
            if step_in_episode % int(frame_multiplier*execution_rate) == 0 or done:
                trajectory['input_location'].append(np.copy(env.input_location))
                trajectory['input_magnitude'].append(np.copy(env.input_magnitude))
                trajectory['temperature_field'].append(np.copy(env.temp_mesh))
                trajectory['temperature_target'].append(np.copy(env.target_temp_mesh))
                trajectory['temperature_rel_error'].append(np.copy(env.temp_error))
                trajectory['temperature_max_error'].append(np.copy(env.temp_error_max))
                trajectory['temperature_min_error'].append(np.copy(env.temp_error_min))
                trajectory['time'].append(np.copy(env.current_time))
            
            # Update state and step
            s = s2
            step_in_episode = step_in_episode + 1
    
        # Calculate reward per epoch
        episode_reward = r_total - episode_reward
        if episode_reward > best_episode or curr_episode == 0:
            best_episode = episode_reward
            data['input_location'] = trajectory['input_location']
            data['input_magnitude'] = trajectory['input_magnitude']
            data['temperature_field'] = trajectory['temperature_field']
            data['temperature_target'] = trajectory['temperature_target']
            data['temperature_rel_error'] = trajectory['temperature_rel_error']
            data['temperature_max_error'] = trajectory['temperature_max_error']
            data['temperature_min_error'] = trajectory['temperature_min_error']
            data['time'] = trajectory['time']
            data['best_episode'] = episode_reward
        
        # Update the logs
        data['r_per_episode'].append(episode_reward / agent.steps_per_trajectory)
        
        # Reset trajecotry memory
        trajectory['input_location'] = []
        trajectory['input_magnitude'] = []
        trajectory['temperature_field'] = []
        trajectory['temperature_target'] = []
        trajectory['temperature_rel_error'] = []
        trajectory['temperature_max_error'] = []
        trajectory['temperature_min_error'] = []
        trajectory['time'] = []
        
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
    
    # Agent hyperparameters
    total_trajectories = 10
    steps_per_trajecotry = 240
    trajectories_per_batch = 10
    num_epochs = 10
    gamma = 0.99
    lamb = 0.95
    epsilon = 0.20
    start_alpha = 1.0e-3
    end_alpha = 5.0e-4
    
    # Rendering parameters
    frame_multiplier = 1.0/6.0
    dpi = 100
        
    # Calculated env and agent parameters
    env = fes.FES()
    num_states = ((env.num_vert_length-1)//8)*((env.num_vert_width-1)//8) + 52
    decay_rate = (end_alpha/start_alpha)**(trajectories_per_batch/total_trajectories)
    agent_temporal_precision = (env.sim_duration / float(steps_per_trajecotry))
    execution_rate = int(agent_temporal_precision / env.time_step)
    minibatch_size = (trajectories_per_batch * steps_per_trajecotry) // num_epochs
    
    # Check inputs
    if ((int(agent_temporal_precision / env.time_step) - (agent_temporal_precision / env.time_step))!=0):
        raise RuntimeError("Agent execution rate is not multiple of simulation rate")
    
    # Find save paths
    path = "results/PPO_Performance"
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)
    video_path = "results/PPO_Performance/video"
    os.mkdir(video_path)
            
    # Find load path
    done = False
    curr_folder = 1
    while not done:
        load_path = "results/PPO_"+str(curr_folder)
        if not os.path.isdir(load_path):
            done = True
        else:
            curr_folder = curr_folder + 1
    curr_folder = curr_folder - 1
    if curr_folder == 0:
        raise RuntimeError("No training data found")
    load_path = "results/PPO_"+str(curr_folder)
    load_path = load_path + "/output"
    if not os.path.exists(load_path):
        raise RuntimeError("Could not find previous output file: " + load_path)
    with open(load_path, 'rb') as file:
        input_data = pickle.load(file)  
    old_agent = input_data['agent']

    # Create agents, run simulation, save results
    agent = ppo.PPO_Agent(num_states, steps_per_trajecotry, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, start_alpha, decay_rate)
    print("Loading previous agent data...")
    agent.copy(old_agent, False)
    print("Gathering performance data...")
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
    plt.title("Relative Difference Trajectory",fontsize='xx-large')
    plt.xlabel("Time [s]",fontsize='large')
    plt.ylabel("Relative Difference from Target Temperature [%]",fontsize='large')
    plt.plot(data['time'],100.0*np.array(data['temperature_rel_error']),c='k',lw=2.0)
    plt.plot(data['time'],100.0*np.array(data['temperature_max_error']),c='r',ls='--',lw=2.0)
    plt.plot(data['time'],100.0*np.array(data['temperature_min_error']),c='b',ls='--',lw=2.0)
    plt.legend(('Average', 'Maximum', 'Minimum'),loc='upper left',fontsize='large')
    plt.grid(which='major',axis='y')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.gcf().set_size_inches(8.5, 5.5)
    save_file = path + "/trajectory.png"
    plt.savefig(save_file, dpi = 500)
    plt.close()
    
    # Plot actor learning curve
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
    
    # Make videos of the best temperature field trajecotry and cure field trajectories as function of time
    print("Rendering...")
    min_temp = 0.99*np.min(data['temperature_field'])
    max_temp = max(1.01*np.max(data['temperature_field']), 1.05*np.max(env.target_temp_mesh))
    normalized_temperature = 100.0*np.array(data['temperature_field'])/np.array(data['temperature_target'])
    min_normalized_temp = np.min(0.99*normalized_temperature)
    max_normalized_temp = np.max(1.01*normalized_temperature)
    
    # Make custom color map for normalized data
    lower = min(5.0*round(min_normalized_temp//5.0), 90.0)
    mid_lower = 95.0
    mid_upper = 105.0
    upper = max(5.0*round(max_normalized_temp/5.0), 110.0)
    lower_delta = mid_lower - lower
    upper_delta = upper - mid_upper
    n_colors_in_lower = round((lower_delta / (lower_delta + upper_delta)) * 10.0)
    n_color_in_upper = round((upper_delta / (lower_delta + upper_delta)) * 10.0)
    lower_ticks = np.round(np.linspace(lower,mid_lower,n_colors_in_lower+1))
    upper_ticks = np.round(np.linspace(mid_upper,upper,n_color_in_upper+1))
    ticks = np.concatenate((lower_ticks, upper_ticks))
    norm = clr.BoundaryNorm(ticks, 11)
    base_color_array = ["navy","blue","deepskyblue","lightseagreen","forestgreen","limegreen","yellow","orange","orangered","firebrick"]
    base_color_array.insert(n_colors_in_lower, "fuchsia")
    cmap = clr.ListedColormap(base_color_array)
    for curr_step in range(len(data['time'])):
           
        # Calculate input field
        input_magnitude = data['input_magnitude'][curr_step]
        input_location = data['input_location'][curr_step]
        input_mesh = input_magnitude * env.max_input_mag * np.exp(((env.mesh_cens_x_cords - input_location[0])**2 * env.exp_const) + 
                                                                      (env.mesh_cens_y_cords - input_location[1])**2 * env.exp_const)
        input_mesh[input_mesh<0.01*env.max_input_mag] = 0.0
        
        # Make fig for temperature, cure, and input
        plt.cla()
        plt.clf()
        fig, (ax0, ax1) = plt.subplots(2, 1)
        fig.set_size_inches(8.5,7.5)
        
        # Plot temperature
        c0 = ax0.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, normalized_temperature[curr_step], shading='auto', cmap=cmap, norm=norm)
        cbar0 = fig.colorbar(c0, ax=ax0)
        cbar0.set_label('Percent of Target', labelpad=20,fontsize='large')
        cbar0.set_ticks(ticks)
        cbar0.ax.tick_params(labelsize=12)
        ax0.set_xlabel('X Position [cm]',fontsize='large')
        ax0.set_ylabel('Y Position [cm]',fontsize='large')
        ax0.tick_params(axis='x',labelsize=12)
        ax0.tick_params(axis='y',labelsize=12)
        ax0.set_aspect('equal', adjustable='box')
        
        # Plot input
        c1 = ax1.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, 1.0e-6*input_mesh, shading='auto', cmap='coolwarm', vmin=0.0, vmax=1.0e-6*env.max_input_mag)
        cbar1 = fig.colorbar(c1, ax=ax1)
        cbar1.set_label('Input Heat Rate Density [MW/m^3]', labelpad=20,fontsize='large')
        cbar1.ax.tick_params(labelsize=12)
        ax1.set_xlabel('X Position [cm]',fontsize='large')
        ax1.set_ylabel('Y Position [cm]',fontsize='large')
        ax1.tick_params(axis='x',labelsize=12)
        ax1.tick_params(axis='y',labelsize=12)
        ax1.set_aspect('equal', adjustable='box')
        
        # Set title and save
        title_str = "Simulation Time: "+'{:.2f}'.format(data['time'][curr_step])+'s'
        fig.suptitle(title_str,fontsize='x-large')
        plt.savefig(video_path+'/time_'+'{:.2f}'.format(data['time'][curr_step])+'.png', dpi=dpi)
        plt.close()
    
    print("Done!")