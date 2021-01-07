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

def main(env, agent, total_trajectories, execution_rate):

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
                ("| Traj: " + str(curr_episode+1) + "/" + str(total_trajectories)).ljust(21) + 
                ("| R/Step: " + '{:.2f}'.format(data['r_per_episode'][-1])).ljust(20) + 
                ("| Avg_R/Step: " + '{:.2f}'.format(data['r_per_step'][-1])).ljust(24) + 
                ("| Best R: " + '{:.1f}'.format(best_episode)).ljust(17) + 
                ("| Avg R: " + '{:.1f}'.format(r_total / curr_episode)).ljust(16) + 
                "|")
            print(print_str, end="\r", flush=True)
        else:
            print_str = (('{:03.1f}'.format(100.0 * percent_complete) + "% Complete").ljust(16) + 
                ("| Traj: " + str(curr_episode+1) + "/" + str(total_trajectories)).ljust(21) + 
                ("| R/Step: " + '{:.2f}'.format(0.0)).ljust(20) + 
                ("| Avg_R/Step: " + '{:.2f}'.format(0.0)).ljust(24) + 
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
            trajectory['input_location'].append(env.input_location)
            trajectory['temperature_field'].append(env.temp_panels)
            trajectory['input_magnitude'].append(env.input_magnitude)
            trajectory['cure_field'].append(env.cure_panels)
            trajectory['front_location'].append(env.front_loc)
            trajectory['front_velocity'].append(env.front_vel)
            trajectory['target_velocity'].append(env.current_target_front_vel)
            trajectory['time'].append(env.current_time)
            
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
        data['x_loc_rate_stdev'].append(x_loc_rate_stdev)
        data['y_loc_rate_stdev'].append(y_loc_rate_stdev)
        data['mag_stdev'].append(mag_stdev)
        data['r_per_step'].append(r_total / (curr_step+1))
        trajectory['input_location'] = []
        trajectory['temperature_field'] = []
        trajectory['input_magnitude'] = []
        trajectory['cure_field'] = []
        trajectory['front_location'] = []
        trajectory['front_velocity'] = []
        trajectory['target_velocity'] = []
        trajectory['time'] = []
   
    # Store the training data
    data['value_error'].append(agent.value_estimation_error)
        
    # User readout
    percent_complete = 1.0
    if curr_episode > 0:
        print_str = (("100.0% Complete").ljust(16) + 
            ("| Traj: " + str(total_trajectories) + "/" + str(total_trajectories)).ljust(21) + 
            ("| R/Step: " + '{:.2f}'.format(data['r_per_episode'][-1])).ljust(20) + 
            ("| Avg_R/Step: " + '{:.2f}'.format(data['r_per_step'][-1])).ljust(24) + 
            ("| Best R: " + '{:.1f}'.format(best_episode)).ljust(17) + 
            ("| Avg R: " + '{:.1f}'.format(r_total / curr_episode)).ljust(16) + 
            "|")
    else:
        print_str = (("100.0% Complete").ljust(16) + 
            ("| Traj: " + str(total_trajectories) + "/" + str(total_trajectories)).ljust(21) + 
            ("| R/Step: " + '{:.2f}'.format(data['r_per_episode'][-1])).ljust(20) + 
            ("| Avg_R/Step: " + '{:.2f}'.format(data['r_per_step'][-1])).ljust(24) + 
            ("| Best R: " + '{:.1f}'.format(best_episode)).ljust(17) + 
            ("| Avg R: " + '{:.1f}'.format(r_total)).ljust(16) + 
            "|")     
    print(print_str, end="\n", flush=True)
    
    return data, agent, env


if __name__ == '__main__':
    
    # Create environment
    random_target = False
    target_switch = False
    control = True
    for_pd = False
    env = fes.FES(random_target=random_target, target_switch=target_switch, control=control, for_pd=for_pd)
    num_states = int((env.num_panels_length//20) * (env.num_panels_width//20) + 2*env.num_panels_width//10 + 28)
        
    # Set agent parameters
    total_trajectories = 1
    steps_per_trajecotry = 240
    trajectories_per_batch = 10
    num_epochs = 10
    gamma = 0.99
    lamb = 0.95
    epsilon = 0.20
    start_alpha = 1.0e-3
    end_alpha = 1.0e-4
    
    # Calculated agent parameters
    decay_rate = (end_alpha/start_alpha)**(trajectories_per_batch/total_trajectories)
    agent_temporal_precision = (env.sim_duration / float(steps_per_trajecotry))
    execution_rate = int(agent_temporal_precision / env.time_step)
    minibatch_size = (trajectories_per_batch * steps_per_trajecotry) // num_epochs

    # Check inputs
    if ((int(agent_temporal_precision / env.time_step) - (agent_temporal_precision / env.time_step))!=0):
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
        agent = ppo.PPO_Agent(num_states, steps_per_trajecotry, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, start_alpha, decay_rate)
        data, agent, env = main(env, agent, total_trajectories, execution_rate)
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
    plt.savefig('results/PPO-Controller/front_velocity.png', dpi = 500)
    plt.close()
        
    # Plot learning curve 1
    if not control:
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
    max_temp = max(1.05*np.max(logbook['data'][best_overall_agent]['temperature_field']), 1.05*env.temperature_limit)
    for curr_step in range(len(logbook['data'][best_overall_agent]['time'])):
        if curr_step % 720 == 0 or curr_step == len(logbook['data'][best_overall_agent]['time']) - 1:
            plt.clf()
            c = plt.pcolor(env.panels_x, env.panels_y, logbook['data'][best_overall_agent]['temperature_field'][curr_step], shading='auto', cmap='jet', vmin=min_temp, vmax=max_temp)
            cbar = plt.gcf().colorbar(c)
            cbar.set_label('Temperature K', labelpad=20)
            title_str = "Temperature: t = "+'{:.2f}'.format(curr_step*env.time_step)+'s'
            plt.title(title_str)
            plt.xlabel('X Position [m]')
            plt.ylabel('Y Position [m]')
            plt.savefig('results/PPO-Controller/videos/temp_'+'{:.2f}'.format(curr_step*env.time_step)+'.png', dpi = 100)
            plt.close()
            
            plt.clf()
            c = plt.pcolor(env.panels_x, env.panels_y, logbook['data'][best_overall_agent]['cure_field'][curr_step], shading='auto', cmap='jet', vmin=0.0, vmax=1.0)
            cbar = plt.gcf().colorbar(c)
            cbar.set_label('Degree Cure', labelpad=20)
            title_str = "Degree Cure: t = "+'{:.2f}'.format(curr_step*env.time_step)+'s'
            plt.title(title_str)
            plt.xlabel('X Position [m]')
            plt.ylabel('Y Position [m]')
            plt.savefig('results/PPO-Controller/videos/cure_'+'{:.2f}'.format(curr_step*env.time_step)+'.png', dpi = 100)
            plt.close()
    
    print("Done!")