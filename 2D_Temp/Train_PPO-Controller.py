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
        'value_error' : [],
        'x_loc_rate_stdev': [],
        'y_loc_rate_stdev': [],
        'mag_stdev': [],
        'input_location': [],
        'input_magnitude':[],
        'temperature_field': [],
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
        x_loc_rate_stdev = 0.0
        y_loc_rate_action = 0.0
        y_loc_rate_stdev = 0.0
        mag_action = 0.0
        mag_stdev = 0.0
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
                trajectory['temperature_field'].append(np.copy(env.temp_mesh))
                trajectory['input_magnitude'].append(np.copy(env.input_magnitude))
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
            data['time'] = trajectory['time']
            data['best_episode'] = episode_reward
        
        # Update the logs
        data['r_per_episode'].append(episode_reward / agent.steps_per_trajectory)
        data['x_loc_rate_stdev'].append(x_loc_rate_stdev)
        data['y_loc_rate_stdev'].append(y_loc_rate_stdev)
        data['mag_stdev'].append(mag_stdev)
        data['r_per_step'].append(r_total / (curr_step+1))
        trajectory['input_location'] = []
        trajectory['temperature_field'] = []
        trajectory['input_magnitude'] = []
        trajectory['cure_field'] = []
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
        
    # Agent parameters
    num_agents = 1
    total_trajectories = 20000
    steps_per_trajecotry = 240
    trajectories_per_batch = 10
    num_epochs = 10
    gamma = 0.99
    lamb = 0.95
    epsilon = 0.20
    start_alpha = 1.0e-3
    end_alpha = 1.0e-4
    
    # Rendering parameters
    frame_multiplier = 1.0/6.0
    dpi = 100
    
    # Calculated env and agent parameters
    env = fes.FES()
    num_states = ((env.num_vert_length-1)//9)*((env.num_vert_width-1)//5) + 28
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

    # Create agents, run simulations, save results
    for curr_agent in range(num_agents):
        print("Agent " + str(curr_agent+1) + " / " + str(num_agents))
        agent = ppo.PPO_Agent(num_states, steps_per_trajecotry, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, start_alpha, decay_rate)
        data, agent, env = run(env, agent, total_trajectories, execution_rate, frame_multiplier)
        logbook['data'].append(data)
        logbook['agents'].append(agent)
        logbook['envs'].append(env)
    
    # Average results from all agents
    print('Processing...')
    if num_agents > 1:
        r_per_step_stdev = np.zeros((num_agents, len(logbook['data'][curr_agent]['r_per_step'])))
        r_per_episode_stdev = np.zeros((num_agents, len(logbook['data'][curr_agent]['r_per_episode'])))
        value_learning_stdev = np.zeros((num_agents, len(logbook['data'][curr_agent]['value_error'][0])))
    average_r_per_step = np.array([0.0]*len(logbook['data'][curr_agent]['r_per_step']))
    average_r_per_episode = np.array([0.0]*len(logbook['data'][curr_agent]['r_per_episode']))
    average_value_learning = np.array([0.0]*len(logbook['data'][curr_agent]['value_error'][0]))
    average_x_loc_rate_stdev = np.array([0.0]*len(logbook['data'][curr_agent]['x_loc_rate_stdev']))
    average_y_loc_rate_stdev = np.array([0.0]*len(logbook['data'][curr_agent]['y_loc_rate_stdev']))
    average_mag_stdev = np.array([0.0]*len(logbook['data'][curr_agent]['mag_stdev']))
    for curr_agent in range(num_agents):
        if num_agents > 1:
            r_per_step_stdev[curr_agent,:] = logbook['data'][curr_agent]['r_per_step']
            r_per_episode_stdev[curr_agent,:] = logbook['data'][curr_agent]['r_per_episode']
            value_learning_stdev[curr_agent,:] = logbook['data'][curr_agent]['value_error'][0]
        average_r_per_step = average_r_per_step + np.array(logbook['data'][curr_agent]['r_per_step'])
        average_r_per_episode = average_r_per_episode + np.array(logbook['data'][curr_agent]['r_per_episode'])
        average_value_learning = average_value_learning + np.array(logbook['data'][curr_agent]['value_error'][0])
        average_x_loc_rate_stdev = average_x_loc_rate_stdev + np.array(logbook['data'][curr_agent]['x_loc_rate_stdev'])
        average_y_loc_rate_stdev = average_y_loc_rate_stdev + np.array(logbook['data'][curr_agent]['y_loc_rate_stdev'])
        average_mag_stdev = average_mag_stdev + np.array(logbook['data'][curr_agent]['mag_stdev'])
        if logbook['data'][curr_agent]['best_episode'] >= best_overall_episode:
            best_overall_episode = logbook['data'][curr_agent]['best_episode']
            best_overall_agent = curr_agent
    if num_agents > 1:
        r_per_step_stdev = np.std(r_per_step_stdev,axis=0)
        r_per_episode_stdev = np.std(r_per_episode_stdev,axis=0)
        value_learning_stdev = np.std(value_learning_stdev,axis=0)
    average_r_per_step = average_r_per_step / float(num_agents)
    average_r_per_episode = average_r_per_episode / float(num_agents)
    average_value_learning = average_value_learning / float(num_agents)
    average_x_loc_rate_stdev = average_x_loc_rate_stdev / float(num_agents)
    average_y_loc_rate_stdev = average_y_loc_rate_stdev / float(num_agents)
    average_mag_stdev = average_mag_stdev / float(num_agents)

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
    with open("results/PPO-Controller/output", 'wb') as file:
        pickle.dump(outputs, file)  

    print("Plotting...")  
    # Plot learning curve 1
    plt.clf()
    title_str = "Actor Learning Curve, Simulation Normalized"
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Simulation Step")
    if num_agents==1:
        plt.plot([*range(len(average_r_per_step))],average_r_per_step)
    else:
        plt.plot([*range(len(average_r_per_step))],average_r_per_step)
        plt.fill_between([*range(len(average_r_per_step))],average_r_per_step+r_per_step_stdev,average_r_per_step-r_per_step_stdev,alpha=0.6)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('results/PPO-Controller/actor_learning_1.png', dpi = 500)
    plt.close()
    
    # Plot learning curve 2
    plt.clf()
    title_str = "Actor Learning Curve, Episode Normalized"
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Simulation Step")
    if num_agents==1:
        plt.plot([*range(len(average_r_per_episode))],average_r_per_episode)
    else:
        plt.plot([*range(len(average_r_per_episode))],average_r_per_episode)
        plt.fill_between([*range(len(average_r_per_episode))],average_r_per_episode+r_per_episode_stdev,average_r_per_episode-r_per_episode_stdev,alpha=0.6)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('results/PPO-Controller/actor_learning_2.png', dpi = 500)
    plt.close()
    
    # Plot value learning curve
    plt.clf()
    title_str = "Critic Learning Curve"
    plt.title(title_str)
    plt.xlabel("Optimization Step")
    plt.ylabel("MSE Loss")
    if num_agents==1:
        plt.plot([*range(len(average_value_learning))],average_value_learning)
    else:
        plt.plot([*range(len(average_value_learning))],average_value_learning)
        plt.fill_between([*range(len(average_value_learning))],average_value_learning+value_learning_stdev,average_value_learning-value_learning_stdev,alpha=0.6)
    plt.yscale("log")
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('results/PPO-Controller/critic_learning.png', dpi = 500)
    plt.close()
   
    # Plot stdev curve
    plt.clf()
    title_str = "Laser X Position Rate Stdev"
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel("Laser X Position Rate Stdev [m/s]")
    plt.plot([*range(len(average_x_loc_rate_stdev))],env.loc_rate_scale*average_x_loc_rate_stdev)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('results/PPO-Controller/x_loc_rate_stdev.png', dpi = 500)
    plt.close()
    
    # Plot stdev curve
    plt.clf()
    title_str = "Laser Y Position Rate Stdev"
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel("Laser Y Position Rate Stdev [m/s]")
    plt.plot([*range(len(average_y_loc_rate_stdev))],env.loc_rate_scale*average_y_loc_rate_stdev)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('results/PPO-Controller/y_loc_rate_stdev.png', dpi = 500)
    plt.close()
    
    # Plot stdev curve
    plt.clf()
    title_str = "Laser Magnitude Stdev"
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel('Laser Magnitude Stdev [K/s]')
    plt.plot([*range(len(average_mag_stdev))],env.mag_scale*env.max_input_mag*average_mag_stdev)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('results/PPO-Controller/mag_stdev.png', dpi = 500)
    plt.close()
    
    # Make videos of the best temperature field trajecotry and cure field trajectories as function of time
    print("Rendering...")
    min_temp = 0.99*np.min(logbook['data'][best_overall_agent]['temperature_field'])
    max_temp = max(1.05*np.max(logbook['data'][best_overall_agent]['temperature_field']), 1.05*np.max(env.target_temp_mesh))
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
        fig, (ax0, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(7.33,8.5)
        
        # Plot temperature
        c0 = ax0.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, logbook['data'][best_overall_agent]['temperature_field'][curr_step], shading='auto', cmap='jet', vmin=min_temp, vmax=max_temp)
        cbar0 = fig.colorbar(c0, ax=ax0)
        cbar0.set_label('Temperature [K]', labelpad=20)
        ax0.set_xlabel('X Position [cm]')
        ax0.set_ylabel('Y Position [cm]')
        ax0.set_aspect('equal', adjustable='box')
        
        # Plot input
        c2 = ax2.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, 1.0e-6*input_mesh, shading='auto', cmap='coolwarm', vmin=0.0, vmax=1.0e-6*env.max_input_mag)
        cbar2 = fig.colorbar(c2, ax=ax2)
        cbar2.set_label('Input Heat Rate Density [MW/m^3]', labelpad=20)
        ax2.set_xlabel('X Position [cm]')
        ax2.set_ylabel('Y Position [cm]')
        ax2.set_aspect('equal', adjustable='box')
        
        # Set title and save
        title_str = "Time from Trigger: "+'{:.2f}'.format(logbook['data'][best_overall_agent]['time'][curr_step])+'s'
        fig.suptitle(title_str)
        plt.savefig('results/PPO-Controller/video/time_'+'{:.2f}'.format(logbook['data'][best_overall_agent]['time'][curr_step])+'.png', dpi=dpi)
        plt.close()
        
        # Collect garbage
        del input_magnitude, input_location, input_mesh, fig, ax0, ax2, c0, c2, cbar0, cbar2, title_str
        gc.collect()
    
    print("Done!")