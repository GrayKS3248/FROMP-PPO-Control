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

def main(env, agent, total_sets):

    # Create dict to store data from simulation
    data = {
        'r_per_step' : [],
        'r_per_epoch' : [],
        'total_reward' : [],
        'value_error' : [],
        'pos_stdev': [],
        'mag_stdev': [],
        'input_location': [],
        'input_magnitude':[],
        'temperature_field': [],
        'cure_field': [],
    }
    r_per_step_history = deque()
    r_per_set_history = deque()
    
    # Create variables used to keep track of learning
    r_total = 0
    epoch_reward = 0
    curr_step = 1
    percent_complete = 0.0
    best_set = 0.0
    
    # Run a set of episodes
    for curr_set in range(total_sets):
        
        # User readout parameters
        percent_complete = curr_set / total_sets
        r_per_step_history.append(epoch_reward/agent.steps_per_trajectory)
        r_per_set_history.append(epoch_reward)
        if len(r_per_step_history) >= 20:
            r_per_step_history.popleft()
            r_per_set_history.popleft()
 
        # User readout
        print_str = (('{:03.2f}'.format(100.0 * percent_complete) + "% Complete").ljust(17) + 
            ("| Set: " + str(curr_set) + "/" + str(total_sets-1)).ljust(19) + 
            ("| Reward: " + '{:.0f}'.format(r_total)).ljust(22) + 
            ("| R/Step: " + '{:.2f}'.format(sum(r_per_step_history) / len(r_per_step_history))).ljust(17) + 
            ("| R/Set: " + '{:.0f}'.format(sum(r_per_set_history) / len(r_per_set_history))).ljust(16) + 
            ("| Best: " + '{:.0f}'.format(best_set)).ljust(14) + 
            "|")
        print(print_str, end="\r", flush=True)
        
        # Initialize simulation
        s = env.reset()
        epoch_reward = r_total
                
        # Simulate until episode is done
        done = False
        while not done:
            
            # Get action, do action, learn
            pos_action, pos_stdev, mag_action, mag_stdev = agent.get_action(s)
            (s2, r, done) = env.step(np.array([pos_action, mag_action]))
            agent.update_agent(s, np.array([pos_action, mag_action]), r)
            
            # Update logs
            data['pos_stdev'].append(pos_stdev)
            data['mag_stdev'].append(mag_stdev)
            r_total = r_total + r
            data['total_reward'].append(r_total)
            data['r_per_step'].append(r_total / curr_step)
            data['input_location'].append(env.input_location)
            data['temperature_field'].append(env.temperature_grid)
            data['cure_field'].append(env.cure_grid)
            data['input_magnitude'].append(env.input_magnitude)
            
            # Update state and step
            s = s2
            curr_step = curr_step + 1
    
        # Calculate reward per epoch
        epoch_reward = r_total - epoch_reward
        if epoch_reward > best_set or curr_set == 0:
            best_set = epoch_reward
        data['r_per_epoch'].append(epoch_reward)
   
    # Store the training data
    data['value_error'].append(agent.value_estimation_error)
        
    # User readout
    percent_complete = 1.0
    print_str = (("100.00% Complete").ljust(17) + 
        ("| Set: " + str(curr_set) + "/" + str(total_sets-1)).ljust(19) + 
        ("| Reward: " + '{:.0f}'.format(r_total)).ljust(22) + 
        ("| R/Step: " + '{:.2f}'.format(sum(r_per_step_history) / len(r_per_step_history))).ljust(17) + 
        ("| R/Set: " + '{:.0f}'.format(sum(r_per_set_history) / len(r_per_set_history))).ljust(16) + 
        ("| Best: " + '{:.0f}'.format(best_set)).ljust(14) + 
        "|")
    print(print_str, end="\n", flush=True)
    
    return data, agent, env


if __name__ == '__main__':
    
    # Create environment
    env = fes.FES()
    num_states = int(env.spacial_precision/10 + 4)
    
    # Set agent parameters
    trajectories_per_batch = 1
    num_epochs = 120
    total_simulation_step = 6000
    gamma = 0.99
    lamb = 0.95
    epsilon = 0.20
    alpha = 5e-4
    
    # Calculated agent parameters
    steps_per_trajecotry = int(env.simulation_time / env.temporal_precision)
    num_batches = int(total_simulation_step // (trajectories_per_batch * steps_per_trajecotry))
    minibatch_size = (trajectories_per_batch * steps_per_trajecotry) // num_epochs

    # Simulation parameters
    num_agents = 1
    logbook = {
        'data': [],
        'agents': [],
        'envs': [],
    }

    # Create agents, run simulations, save results
    for curr_agent in range(num_agents):
        agent = ppo.PPO_Agent(num_states, steps_per_trajecotry, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, alpha)
        data, agent, env = main(env, agent, trajectories_per_batch*num_batches)
        logbook['data'].append(data)
        logbook['agents'].append(agent)
        logbook['envs'].append(env)
    
    # Average results from all agents
    print('Processing...')
    average_r_per_step_curve = np.array([0.0]*len(logbook['data'][curr_agent]['r_per_step']))
    average_r_per_epoch_curve = np.array([0.0]*len(logbook['data'][curr_agent]['r_per_epoch']))
    average_value_learning_curve = np.array([0.0]*len(logbook['data'][curr_agent]['value_error'][0]))
    average_pos_stdev_curve = np.array([0.0]*len(logbook['data'][curr_agent]['pos_stdev']))
    average_mag_stdev_curve = np.array([0.0]*len(logbook['data'][curr_agent]['mag_stdev']))
    average_total_reward_curve = np.array([0.0]*len(logbook['data'][curr_agent]['total_reward']))
    highest_reward = -1e+20
    best_agent = 0.0
    for curr_agent in range(num_agents):
        average_r_per_step_curve = average_r_per_step_curve + np.array(logbook['data'][curr_agent]['r_per_step'])
        average_r_per_epoch_curve = average_r_per_epoch_curve + np.array(logbook['data'][curr_agent]['r_per_epoch'])
        average_value_learning_curve = average_value_learning_curve + np.array(logbook['data'][curr_agent]['value_error'][0])
        average_pos_stdev_curve = average_pos_stdev_curve + np.array(logbook['data'][curr_agent]['pos_stdev'])
        average_mag_stdev_curve = average_mag_stdev_curve + np.array(logbook['data'][curr_agent]['mag_stdev'])
        average_total_reward_curve = average_total_reward_curve + np.array(logbook['data'][curr_agent]['total_reward'])
        if logbook['data'][curr_agent]['total_reward'][-1] > highest_reward:
            best_agent = logbook['agents'][curr_agent]
            highest_reward = logbook['data'][curr_agent]['total_reward'][-1]
    average_r_per_step_curve = average_r_per_step_curve / float(num_agents)
    average_r_per_epoch_curve = average_r_per_epoch_curve / float(num_agents)
    average_value_learning_curve = average_value_learning_curve / float(num_agents)
    average_pos_stdev_curve = average_pos_stdev_curve / float(num_agents)
    average_mag_stdev_curve = average_mag_stdev_curve / float(num_agents)
    average_total_reward_curve = average_total_reward_curve / float(num_agents)

    print("Plotting...")
    
    # Make video of the best temperature field trajecotry as function of time
    trajectory_start = np.argmax(data['r_per_epoch']) * steps_per_trajecotry
    for curr_step in range(steps_per_trajecotry-1):
        if curr_step % 20 == 0 or curr_step == steps_per_trajecotry - 1:
            plt.clf()
            fig, ax1 = plt.subplots()
            title_str = "Fields: t = "+'{:2.4}'.format(curr_step*env.temporal_precision)+'s'
            plt.title(title_str)
            color = 'tab:red'
            ax1.set_xlabel('Position Step [m]')
            ax1.set_ylabel('Temperature [K]', color=color)
            ax1.plot(env.spacial_grid, data['temperature_field'][curr_step + trajectory_start], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xlim(0.0, env.field_length)
            ax1.set_ylim(0.99*np.min(data['temperature_field'][trajectory_start:trajectory_start+steps_per_trajecotry]), 1.01*np.max(data['temperature_field'][trajectory_start:trajectory_start+steps_per_trajecotry]))
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Degree Cure [-]', color=color)
            ax2.plot(env.spacial_grid, data['cure_field'][curr_step + trajectory_start], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0.0, 1.01)
            plt.axvline(x=data['input_location'][curr_step + trajectory_start],c='r',alpha=data['input_magnitude'][curr_step + trajectory_start]/env.peak_thermal_rate)
            plt.axvline(x=data['input_location'][curr_step + trajectory_start]+env.radius_of_input,c='r',ls=':',alpha=data['input_magnitude'][curr_step + trajectory_start]/env.peak_thermal_rate)
            plt.axvline(x=data['input_location'][curr_step + trajectory_start]-env.radius_of_input,c='r',ls=':',alpha=data['input_magnitude'][curr_step + trajectory_start]/env.peak_thermal_rate)
            plt.savefig('Results/fields/fields_'+'{:2.4}'.format(curr_step*env.temporal_precision)+'.png', dpi = 200)
            plt.close()
        
    # Plot learning curve 1
    plt.clf()
    title_str = "Reward per Step: α = " + str(alpha) + ", γ = " + str(gamma) + ", λ = " + str(lamb) + ", ε = " + str(epsilon)
    plt.title(title_str)
    plt.xlabel("Simulation Step")
    plt.ylabel("Reward")
    plt.plot([*range(len(average_r_per_step_curve))],average_r_per_step_curve)
    plt.savefig('Results/learning_curve_1.png', dpi = 200)
    plt.close()
    
    # Plot learning curve 2
    plt.clf()
    title_str = "Reward per Episode: α = " + str(alpha) + ", γ = " + str(gamma) + ", λ = " + str(lamb) + ", ε = " + str(epsilon)
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot([*range(len(average_r_per_epoch_curve))],average_r_per_epoch_curve)
    plt.savefig('Results/learning_curve_2.png', dpi = 200)
    plt.close()
    
    # Plot value learning curve
    plt.clf()
    title_str = "Value Learning: α = " + str(alpha) + ", γ = " + str(gamma) + ", λ = " + str(lamb) + ", ε = " + str(epsilon)
    plt.title(title_str)
    plt.xlabel("Episode")
    plt.ylabel("Value Target - Estimate Percent Error [%]")
    plt.plot([*range(len(average_value_learning_curve))],average_value_learning_curve)
    plt.savefig('Results/value_learning.png', dpi = 200)
    plt.close()
   
    # Plot stdev curve
    plt.clf()
    title_str = "Pos Stdev: α = " + str(alpha) + ", γ = " + str(gamma) + ", λ = " + str(lamb) + ", ε = " + str(epsilon)
    plt.title(title_str)
    plt.xlabel("Simulation Step")
    plt.ylabel("Stdev")
    plt.plot([*range(len(average_pos_stdev_curve))],average_pos_stdev_curve)
    plt.savefig('Results/pos_stdev.png', dpi = 200)
    plt.close()
    
    # Plot stdev curve
    plt.clf()
    title_str = "Mag Stdev: α = " + str(alpha) + ", γ = " + str(gamma) + ", λ = " + str(lamb) + ", ε = " + str(epsilon)
    plt.title(title_str)
    plt.xlabel("Simulation Step")
    plt.ylabel("Stdev")
    plt.plot([*range(len(average_mag_stdev_curve))],average_mag_stdev_curve)
    plt.savefig('Results/mag_stdev.png', dpi = 200)
    plt.close()
    
    # Plot total reward
    plt.clf()
    title_str = "Total Reward: α = " + str(alpha) + ", γ = " + str(gamma) + ", λ = " + str(lamb) + ", ε = " + str(epsilon)
    plt.title(title_str)
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Reward")
    plt.plot([*range(len(average_total_reward_curve))],average_total_reward_curve)
    plt.savefig('Results/reward.png', dpi = 200)
    plt.close()
    
    print("Done!")