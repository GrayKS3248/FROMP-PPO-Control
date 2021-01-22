# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:50:34 2020

@author: Grayson Schaer
"""
import Finite_Element_Solver_2D as fes 
import PPO_Agent_3_Output as ppo
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gc

def run(env, agent, total_trajectories, execution_rate, frame_multiplier):

    # Create dict to store data from simulation
    data = {
        'r_per_step' : [],
        'r_per_episode' : [],
        'input_location': [],
        'input_magnitude':[],
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
        'input_magnitude':[],
        'temperature_field': [],
        'cure_field': [],
        'front_location': [],
        'front_velocity': [],
        'target_velocity': [],
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
        x_loc_rate_action = 0.0
        y_loc_rate_action = 0.0
        mag_action = 0.0
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
                trajectory['temperature_field'].append(np.copy(env.temp_mesh))
                trajectory['input_magnitude'].append(np.copy(env.input_magnitude))
                trajectory['cure_field'].append(np.copy(env.cure_mesh))
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
            data['input_magnitude'] = trajectory['input_magnitude']
            data['cure_field'] = trajectory['cure_field']
            data['front_location'] = trajectory['front_location']
            data['front_velocity'] = trajectory['front_velocity']
            data['target_velocity'] = trajectory['target_velocity']
            data['time'] = trajectory['time']
            data['best_episode'] = episode_reward
        
        # Update the logs
        data['r_per_episode'].append(episode_reward / agent.steps_per_trajectory)
        data['r_per_step'].append(r_total / (curr_step+1))
        trajectory['input_location'] = []
        trajectory['temperature_field'] = []
        trajectory['input_magnitude'] = []
        trajectory['cure_field'] = []
        trajectory['front_location'] = []
        trajectory['front_velocity'] = []
        trajectory['target_velocity'] = []
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
    
    # Environment parameters
    random_target = False
    target_switch = False
    control = False
    for_pd = False
        
    # Agent parameters
    num_agents = 1
    total_trajectories = 10
    steps_per_trajecotry = 240
    trajectories_per_batch = 10
    num_epochs = 10
    gamma = 0.99
    lamb = 0.95
    epsilon = 0.20
    start_alpha = 2.0e-4
    end_alpha = 1.0e-4
    
    # Rendering parameters
    frame_multiplier = 1.0/6.0
    dpi = 100
    path="PPO-Results"
    
    # Calculated env and agent parameters
    env = fes.FES(random_target=random_target, target_switch=target_switch, control=control, for_pd=for_pd)
    num_states = ((env.num_vert_length-1)//9)*((env.num_vert_width-1)//5) + 25 + 2*((env.num_vert_width-1)//5) + 3
    decay_rate = (end_alpha/start_alpha)**(trajectories_per_batch/total_trajectories)
    agent_temporal_precision = (env.sim_duration / float(steps_per_trajecotry))
    execution_rate = int(agent_temporal_precision / env.time_step)
    minibatch_size = (trajectories_per_batch * steps_per_trajecotry) // num_epochs

    # Check inputs
    if ((int(agent_temporal_precision / env.time_step) - (agent_temporal_precision / env.time_step))!=0):
        raise RuntimeError("Agent execution rate is not multiple of simulation rate")
        
    # Simulation parameters
    logbook = {
        'data': [],
        'agents': [],
        'envs': [],
    }
    best_overall_episode = -1e20
    best_overall_agent = 0

    # Load old agent
    with open("results/PPO-Controller/output", 'rb') as file:
        data = pickle.load(file)  
    old_agent = data['logbook']['agents'][0]
        
    # Create agents, run simulations, save results
    for curr_agent in range(num_agents):
        print("Agent " + str(curr_agent+1) + " / " + str(num_agents))
        agent = ppo.PPO_Agent(num_states, steps_per_trajecotry, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, start_alpha, decay_rate)
        agent.copy(old_agent)
        data, agent, env = run(env, agent, total_trajectories, execution_rate, frame_multiplier)
        logbook['data'].append(data)
        logbook['agents'].append(agent)
        logbook['envs'].append(env)

    # Pickle all important outputs
    print("Saving...")
    outputs = {
    'total_trajectories' : total_trajectories, 
    'steps_per_trajecotry' : steps_per_trajecotry, 
    'trajectories_per_batch' : trajectories_per_batch,
    'num_epochs' : num_epochs, 
    'gamma' : gamma,
    'lambda' : lamb,
    'epsilon' : epsilon,
    'alpha' : start_alpha,
    'decay_rate': decay_rate,
    'logbook' : logbook
    }
    with open("results/"+path+"/output", 'wb') as file:
        pickle.dump(outputs, file)  

    print("Plotting...")
    # Plot front rate trajectory
    plt.clf()
    title_str = "Front Velocity"
    plt.title(title_str)
    plt.xlabel("Simulation Time [s]")
    plt.ylabel("Front Velocity [mm/s]")
    plt.plot(logbook['data'][best_overall_agent]['time'], 1000.0*np.array(np.mean(logbook['data'][best_overall_agent]['front_velocity'],axis=1)), c='k')
    plt.plot(logbook['data'][best_overall_agent]['time'], 1000.0*np.array(logbook['data'][best_overall_agent]['target_velocity']), c='b', ls='--')
    plt.legend(('Actual','Target'),loc='lower right')
    plt.ylim(0.0, max(1.25*1000.0*np.array(logbook['data'][best_overall_agent]['target_velocity'])))
    plt.xlim(0.0, env.sim_duration)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('results/'+path+'/front_velocity.png', dpi = 500)
    plt.close()
    
    # Make videos of the best temperature field trajecotry and cure field trajectories as function of time
    print("Rendering...")
    min_temp = 0.99*np.min(logbook['data'][best_overall_agent]['temperature_field'])
    max_temp = max(1.05*np.max(logbook['data'][best_overall_agent]['temperature_field']), 1.05*env.temperature_limit)
    for curr_step in range(len(logbook['data'][best_overall_agent]['time'])):
           
        # Calculate input field
        input_magnitude = logbook['data'][best_overall_agent]['input_magnitude'][curr_step]
        input_location = logbook['data'][best_overall_agent]['input_location'][curr_step]
        input_mesh = input_magnitude * env.max_input_mag * np.exp(((env.mesh_cens_x_cords - input_location[0])**2 * env.exp_const) + 
                                                                      (env.mesh_cens_y_cords - input_location[1])**2 * env.exp_const)
        input_mesh[input_mesh<0.01*env.max_input_mag] = 0.0
        
        # Make fig for temperature, cure, and input
        plt.cla()
        plt.clf()
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        fig.set_size_inches(11,8.5)
        
        # Plot temperature
        c0 = ax0.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, logbook['data'][best_overall_agent]['temperature_field'][curr_step], shading='auto', cmap='jet', vmin=min_temp, vmax=max_temp)
        cbar0 = fig.colorbar(c0, ax=ax0)
        cbar0.set_label('Temperature [K]', labelpad=20)
        ax0.set_xlabel('X Position [cm]')
        ax0.set_ylabel('Y Position [cm]')
        ax0.set_aspect('equal', adjustable='box')
        
        # Plot cure
        c1 = ax1.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, logbook['data'][best_overall_agent]['cure_field'][curr_step], shading='auto', cmap='YlOrBr', vmin=0.0, vmax=1.0)
        cbar1 = fig.colorbar(c1, ax=ax1)
        cbar1.set_label('Degree Cure [-]', labelpad=20)
        ax1.set_xlabel('X Position [cm]')
        ax1.set_ylabel('Y Position [cm]')
        ax1.set_aspect('equal', adjustable='box')
        
        # Plot input
        c2 = ax2.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, 1.0e-6*input_mesh, shading='auto', cmap='coolwarm', vmin=0.0, vmax=1.0e-6*env.max_input_mag)
        ax2.plot(100.0*logbook['data'][best_overall_agent]['front_location'][curr_step].reshape(env.num_vert_width-1,1), 100.0*env.mesh_cens_y_cords[0,:], 'k-', lw=1.5)
        cbar2 = fig.colorbar(c2, ax=ax2)
        cbar2.set_label('Input Heat Rate Density [MW/m^3]', labelpad=20)
        ax2.set_xlabel('X Position [cm]')
        ax2.set_ylabel('Y Position [cm]')
        ax2.set_aspect('equal', adjustable='box')
        
        # Set title and save
        title_str = "Time from Trigger: "+'{:.2f}'.format(logbook['data'][best_overall_agent]['time'][curr_step])+'s'
        fig.suptitle(title_str)
        plt.savefig('results/'+path+'/video/time_'+'{:.2f}'.format(logbook['data'][best_overall_agent]['time'][curr_step])+'.png', dpi=dpi)
        plt.close()
        
        # Collect garbage
        del input_magnitude, input_location, input_mesh, fig, ax0, ax1, ax2, c0, c1, c2, cbar0, cbar1, cbar2, title_str
        gc.collect()
    
    print("Done!")