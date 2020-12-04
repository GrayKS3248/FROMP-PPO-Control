# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:50:34 2020

@author: Grayson Schaer
"""
import Finite_Element_Solver_1D as fes 
import PPO_Agent_2_Output as ppo
import numpy as np
import matplotlib.pyplot as plt
from collections import deque 

def main(env, agent, total_trajectories, execution_rate):

    # Create dict to store data from simulation
    data = {
        'r_per_step' : [],
        'r_per_episode' : [],
        'value_error' : [],
        'pos_rate_stdev': [],
        'mag_rate_stdev': [],
        'input_location': [],
        'input_magnitude':[],
        'temperature_field': [],
        'cure_field': [],
        'front_location': [],
        'front_velocity': [],
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
        'time': [],
        }
    
    r_per_step_history = deque()
    r_per_episode_history = deque()
    
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
        r_per_step_history.append(episode_reward/(agent.steps_per_trajectory * execution_rate))
        r_per_episode_history.append(episode_reward)
        if len(r_per_step_history) >= 100:
            r_per_step_history.popleft()
            r_per_episode_history.popleft()
        
        # User readout
        print_str = (('{:03.2f}'.format(100.0 * percent_complete) + "% Complete").ljust(17) + 
            ("| Episode: " + str(curr_episode+1) + "/" + str(total_trajectories)).ljust(24) + 
            ("| Reward: " + '{:.0f}'.format(r_total)).ljust(22) + 
            ("| R/Step: " + '{:.2f}'.format(sum(r_per_step_history) / len(r_per_step_history))).ljust(17) + 
            ("| R/Episode: " + '{:.0f}'.format(sum(r_per_episode_history) / len(r_per_episode_history))).ljust(21) + 
            ("| Best: " + '{:.0f}'.format(best_episode)).ljust(14) + 
            "|")
        print(print_str, end="\r", flush=True)
        
        # Initialize simulation
        s = env.reset()
        pos_rate_action = 0.0
        pos_rate_stdev = 0.0
        mag_rate_action = 0.0
        mag_rate_stdev = 0.0
        episode_reward = r_total
                
        # Simulate until episode is done
        done = False
        step_in_episode = 0
        while not done:
            
            # Get action, do action, learn
            if step_in_episode % execution_rate == 0:
                pos_rate_action, pos_rate_stdev, mag_rate_action, mag_rate_stdev = agent.get_action(s)
            (s2, r, done) = env.step(np.array([pos_rate_action, mag_rate_action]))
            if step_in_episode % execution_rate == 0:
                agent.update_agent(s, np.array([pos_rate_action, mag_rate_action]), r)
            
            # Update logs
            r_total = r_total + r
            trajectory['input_location'].append(env.input_location)
            trajectory['temperature_field'].append(env.temperature_grid)
            trajectory['input_magnitude'].append(env.input_magnitude)
            trajectory['cure_field'].append(env.cure_grid)
            trajectory['front_location'].append(env.front_position)
            trajectory['front_velocity'].append(env.front_rate)
            trajectory['time'].append(env.current_time)
            
            # Update state and step
            s = s2
            curr_step = curr_step + 1
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
            data['time'] = trajectory['time']
            data['best_episode'] = episode_reward
        
        # Update the logs
        data['r_per_episode'].append(episode_reward)
        data['pos_rate_stdev'].append(pos_rate_stdev)
        data['mag_rate_stdev'].append(mag_rate_stdev)
        data['r_per_step'].append(r_total / (curr_step+1))
        trajectory['input_location'] = []
        trajectory['temperature_field'] = []
        trajectory['input_magnitude'] = []
        trajectory['cure_field'] = []
        trajectory['front_location'] = []
        trajectory['front_velocity'] = []
        trajectory['time'] = []
   
    # Store the training data
    data['value_error'].append(agent.value_estimation_error)
        
    # User readout
    percent_complete = 1.0
    print_str = (("100.00% Complete").ljust(17) + 
        ("| Episode: " + str(total_trajectories) + "/" + str(total_trajectories)).ljust(24) + 
        ("| Reward: " + '{:.0f}'.format(r_total)).ljust(22) + 
        ("| R/Step: " + '{:.2f}'.format(sum(r_per_step_history) / len(r_per_step_history))).ljust(17) + 
        ("| R/Episode: " + '{:.0f}'.format(sum(r_per_episode_history) / len(r_per_episode_history))).ljust(21) + 
        ("| Best: " + '{:.0f}'.format(best_episode)).ljust(14) + 
        "|")
    print(print_str, end="\n", flush=True)
    
    return data, agent, env


if __name__ == '__main__':
    
    # Create environment
    env = fes.FES()
    num_states = int(env.spacial_precision/10 + 5)
        
    # Set agent parameters
    total_trajectories = 1
    steps_per_trajecotry = 240
    trajectories_per_batch = 40
    num_epochs = 10
    gamma = 0.99
    lamb = 0.95
    epsilon = 0.20
    alpha = 1.0e-4
    
    # Calculated agent parameters
    agent_temporal_precision = (env.simulation_time / float(steps_per_trajecotry))
    execution_rate = int(agent_temporal_precision / env.temporal_precision)
    minibatch_size = (trajectories_per_batch * steps_per_trajecotry) // num_epochs

    # Check inputs
    if ((int(agent_temporal_precision / env.temporal_precision) - (agent_temporal_precision / env.temporal_precision))!=0):
        raise RuntimeError("Agent execution rate is not multiple of simulation rate")
        
    # Simulation parameters
    num_agents = 1
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
        agent = ppo.PPO_Agent(num_states, steps_per_trajecotry, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, alpha)
        data, agent, env = main(env, agent, total_trajectories, execution_rate)
        logbook['data'].append(data)
        logbook['agents'].append(agent)
        logbook['envs'].append(env)
    
    # Average results from all agents
    print('Processing...')
    average_r_per_step = np.array([0.0]*len(logbook['data'][curr_agent]['r_per_step']))
    average_r_per_episode = np.array([0.0]*len(logbook['data'][curr_agent]['r_per_episode']))
    average_value_learning = np.array([0.0]*len(logbook['data'][curr_agent]['value_error'][0]))
    average_pos_rate_stdev = np.array([0.0]*len(logbook['data'][curr_agent]['pos_rate_stdev']))
    average_mag_rate_stdev = np.array([0.0]*len(logbook['data'][curr_agent]['mag_rate_stdev']))
    for curr_agent in range(num_agents):
        average_r_per_step = average_r_per_step + np.array(logbook['data'][curr_agent]['r_per_step'])
        average_r_per_episode = average_r_per_episode + np.array(logbook['data'][curr_agent]['r_per_episode'])
        average_value_learning = average_value_learning + np.array(logbook['data'][curr_agent]['value_error'][0])
        average_pos_rate_stdev = average_pos_rate_stdev + np.array(logbook['data'][curr_agent]['pos_rate_stdev'])
        average_mag_rate_stdev = average_mag_rate_stdev + np.array(logbook['data'][curr_agent]['mag_rate_stdev'])
        if logbook['data'][curr_agent]['best_episode'] >= best_overall_episode:
            best_overall_episode = logbook['data'][curr_agent]['best_episode']
            best_overall_agent = curr_agent
    average_r_per_step = average_r_per_step / float(num_agents)
    average_r_per_episode = average_r_per_episode / float(num_agents)
    average_value_learning = average_value_learning / float(num_agents)
    average_pos_rate_stdev = average_pos_rate_stdev / float(num_agents)
    average_mag_rate_stdev = average_mag_rate_stdev / float(num_agents)

    print("Plotting...")
    
    # Make video of the best temperature field trajecotry as function of time
    y_min_temperature = 0.99*np.min(logbook['data'][best_overall_agent]['temperature_field'])
    y_max_temperature = max(1.01*np.max(logbook['data'][best_overall_agent]['temperature_field']), env.maximum_temperature)
    for curr_step in range(len(logbook['data'][best_overall_agent]['temperature_field'])):
        if curr_step % 5 == 0 or curr_step == len(logbook['data'][best_overall_agent]['temperature_field']) - 1:
            plt.clf()
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            title_str = "Temperature and Cure: t = "+'{:.2f}'.format(curr_step*env.temporal_precision)+'s'
            plt.title(title_str)
            color = 'k'
            ax1.set_xlabel('Position [m]')
            ax1.set_ylabel('Temperature [K]', color=color)
            temp = ax1.plot(env.spacial_grid, logbook['data'][best_overall_agent]['temperature_field'][curr_step], color=color,label='Temp')
            max_temp = ax1.axhline(y=env.maximum_temperature,c='k',ls=':',label='Limit')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xlim(0.0, env.field_length)
            ax1.set_ylim(y_min_temperature, y_max_temperature)
            color = 'b'
            ax2.set_ylabel('Degree Cure [-]', color=color)
            cure = ax2.plot(env.spacial_grid, logbook['data'][best_overall_agent]['cure_field'][curr_step], color=color, label = 'Cure')
            front = plt.axvline(x=logbook['data'][best_overall_agent]['front_location'][curr_step],c='b',ls=':',label='Front')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0.0, 1.01)
            input_location = logbook['data'][best_overall_agent]['input_location'][curr_step]
            input_magnitude = logbook['data'][best_overall_agent]['input_magnitude'][curr_step]
            input_center = ax2.axvline(x=input_location,c='r',alpha=input_magnitude/env.peak_thermal_rate,label='Center')
            input_edge = ax2.axvline(x=input_location+env.radius_of_input,c='r',ls=':',alpha=input_magnitude/env.peak_thermal_rate, label='Edge')
            plt.axvline(x=input_location-env.radius_of_input,c='r',ls=':',alpha=input_magnitude/env.peak_thermal_rate)
            lns=(temp[0],max_temp,cure[0],front,input_center,input_edge)
            labs=(temp[0].get_label(),max_temp.get_label(),cure[0].get_label(),front.get_label(),input_center.get_label(),input_edge.get_label())
            ax1.legend(lns, labs, loc=1)
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig('Results/fields/fields_'+'{:2.4}'.format(curr_step*env.temporal_precision)+'.png', dpi = 100)
            plt.close()
        
    # Plot front rate trajectory
    plt.clf()
    title_str = "Front Velocity: t = "+'{:2.4}'.format(curr_step*env.temporal_precision)+'s'
    plt.title(title_str)
    plt.xlabel("Simulation Time [s]")
    plt.ylabel("Front Rate [m/s]")
    plt.plot(logbook['data'][best_overall_agent]['time'], logbook['data'][best_overall_agent]['front_velocity'], c='k')
    plt.axhline(y=env.desired_front_rate, c='b', ls='--')
    plt.legend(('Actual','Target'))
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('Results/front_velocity.png', dpi = 500)
    plt.close()
        
    # Plot learning curve 1
    plt.clf()
    title_str = "Actor Learning Curve, Simulation Step Normalized"
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel("Reward per Simulation Step")
    plt.plot([*range(len(average_r_per_step))],average_r_per_step)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('Results/actor_learning_1.png', dpi = 500)
    plt.close()
    
    # Plot learning curve 2
    plt.clf()
    title_str = "Actor Learning Curve, Episode Normalized"
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel("Reward per Episode")
    plt.plot([*range(len(average_r_per_episode))],average_r_per_episode)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('Results/actor_learning_2.png', dpi = 500)
    plt.close()
    
    # Plot value learning curve
    plt.clf()
    title_str = "Critic Learning Curve"
    plt.title(title_str)
    plt.xlabel("Optimization Step")
    plt.ylabel("Value Target to Estimated Value % Error")
    plt.plot([*range(len(average_value_learning))],average_value_learning)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('Results/critic_learning.png', dpi = 500)
    plt.close()
   
    # Plot stdev curve
    plt.clf()
    title_str = "Laser Velocity Stdev"
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel("Laser Velocity Stdev [m/s]")
    plt.plot([*range(len(average_pos_rate_stdev))],average_pos_rate_stdev)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('Results/pos_rate_stdev.png', dpi = 500)
    plt.close()
    
    # Plot stdev curve
    plt.clf()
    title_str = "Laser Mag Rate Stdev"
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel('Laser Magnitude Rate Stdev [' + '$d^2$' + 'T/' + '$dt^2$' + ']')
    plt.plot([*range(len(average_mag_rate_stdev))],average_mag_rate_stdev)
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig('Results/mag_rate_stdev.png', dpi = 500)
    plt.close()
    
    print("Done!")