# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:50:34 2020

@author: Grayson Schaer
"""
import Finite_Element_Solver_2D as fes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import pickle
import os
import shutil

def run(env, total_trajectories, execution_rate, frame_multiplier, denom_const, steps_per_trajectory):

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
                ("| Best R: " + '{:.1f}'.format(best_episode)).ljust(18) +
                ("| Avg R: " + '{:.1f}'.format(np.mean(data['r_per_episode'][-100:])*steps_per_trajectory)).ljust(17) +
                "|")
            print(print_str, end="\r", flush=True)
        else:
            print_str = (('{:03.1f}'.format(100.0 * percent_complete) + "% Complete").ljust(16) +
                ("| Traj: " + str(curr_episode+1) + "/" + str(total_trajectories)).ljust(20) +
                ("| R/Step: " + '{:.2f}'.format(0.0)).ljust(16) +
                ("| Avg_R/Step: " + '{:.2f}'.format(0.0)).ljust(20) +
                ("| Best R: " + '{:.1f}'.format(best_episode)).ljust(18) +
                ("| Avg R: " + '{:.1f}'.format(0.0)).ljust(17) +
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
                mag_action = (((1.0 - s[0]) / denom_const))
            (s2, r, done) = env.step(np.array([mag_action]))
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
        data['r_per_episode'].append(episode_reward / steps_per_trajectory)

        # Reset trajectory memory
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
        ("| Best R: " + '{:.1f}'.format(best_episode)).ljust(18) +
        ("| Avg R: " + '{:.1f}'.format(np.mean(data['r_per_episode'][-100:])*steps_per_trajectory)).ljust(17) +
        "|")
    print(print_str, end="\n", flush=True)

    return data, env

if __name__ == '__main__':

    # Simulation set parameters
    denom_const_set = np.linspace(0.012, 0.012, 1)
    loc_multiplier_set = np.linspace(0.16, 0.16, 1)
    render = True
    plot = True
    combine = False
    
    # Simulation parameters
    total_trajectories = 10
    uniform_target = False
    split_target = False
    random_target = True

    # Agent hyperparameters
    steps_per_trajectory = 3600

    # Rendering parameters
    frame_multiplier = 2.0
    dpi = 100

    # Initialize simulation set
    for i in range(len(denom_const_set)):
        denom_const = denom_const_set[i]
        for j in range(len(loc_multiplier_set)):
            loc_multiplier = loc_multiplier_set[j]
            
            # Calculated env and agent parameters
            env = fes.FES(loc_multiplier=loc_multiplier, uniform_target=uniform_target, split_target=split_target, random_target=random_target)
            agent_temporal_precision = (env.sim_duration / float(steps_per_trajectory))
            execution_rate = int(agent_temporal_precision / env.time_step)
        
            # Check inputs
            if ((int(agent_temporal_precision / env.time_step) - (agent_temporal_precision / env.time_step))!=0):
                raise RuntimeError("Agent execution rate is not multiple of simulation rate")
        
            # Make save paths
            path = "results/P_"+'{:03.3f}'.format(denom_const)+"_"+'{:03.2f}'.format(loc_multiplier)
            video_path = "results/P_"+'{:03.3f}'.format(denom_const)+"_"+'{:03.2f}'.format(loc_multiplier)+"/video"
            if not os.path.isdir(path):
                os.mkdir(path)
                os.mkdir(video_path)
            else:
                shutil.rmtree(path)
                os.mkdir(path)
                os.mkdir(video_path)
        
            # Create agents, run simulation, save results
            print("Denominator Constant = "+'{:03.3f}'.format(denom_const)+", Location Multiplier = "+'{:03.2f}'.format(loc_multiplier)+"...")
            data, env = run(env, total_trajectories, execution_rate, frame_multiplier, denom_const, steps_per_trajectory)
        
            # Pickle all important outputs
            print("Saving...")
            output = { 'data':data, 'env':env }
            save_file = path + "/output"
            with open(save_file, 'wb') as file:
                pickle.dump(output, file)
        
            if (plot):
                # Plot the trajectory
                print("Plotting...")
                plt.clf()
                plt.title("Relative Difference Trajectory",fontsize='xx-large')
                plt.xlabel("Time [s]",fontsize='large')
                plt.ylabel("Relative Difference from Target Temperature [%]",fontsize='large')
                plt.plot(data['time'],100.0*np.array(data['temperature_rel_error']),c='k',lw=2.0)
                plt.plot(data['time'],100.0*np.array(data['temperature_max_error']),c='r',ls='--',lw=2.0)
                plt.plot(data['time'],100.0*np.array(data['temperature_min_error']),c='b',ls='--',lw=2.0)
                plt.legend(('Average', 'Maximum', 'Minimum'),loc='lower right',fontsize='large')
                plt.grid(which='major',axis='y')
                plt.xticks(fontsize='large')
                plt.yticks(fontsize='large')
                plt.gcf().set_size_inches(8.5, 5.5)
                save_file = path + "/trajectory.png"
                plt.savefig(save_file, dpi = 500)
                plt.close()
            
                # Plot performance
                plt.clf()
                title_str = "Performance: Average = " + '{:03.3f}'.format(np.mean(data['r_per_episode'])) + ", Stdev = " + '{:03.3f}'.format(np.std(data['r_per_episode']))
                plt.title(title_str,fontsize='xx-large')
                plt.xlabel("Episode",fontsize='large')
                plt.ylabel("Average Reward per Simulation Step",fontsize='large')
                plt.plot([*range(len(data['r_per_episode']))],data['r_per_episode'],lw=2.0,c='r')
                plt.xticks(fontsize='large')
                plt.yticks(fontsize='large')
                plt.gcf().set_size_inches(8.5, 5.5)
                save_file = path + "/performance.png"
                plt.savefig(save_file, dpi = 500)
                plt.close()
        
            # Make videos of the best temperature field trajecotry as function of time
            if render: 
                print("Rendering...")
                min_temp = 0.99*np.min(data['temperature_field'])
                max_temp = max(1.01*np.max(data['temperature_field']), 1.05*np.max(env.target_temp_mesh))
                normalized_temperature = 100.0*np.array(data['temperature_field'])/np.array(data['temperature_target'])
                min_normalized_temp = np.min(0.99*normalized_temperature)
                max_normalized_temp = np.max(1.01*normalized_temperature)
            
                # Make custom color map for normalized data
                lower = min(5.0*round(min_normalized_temp//5.0), 90.0)
                mid_lower = 99.0
                mid_upper = 101.0
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
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
                    fig.set_size_inches(11,8.5)
            
                    # Plot temperature
                    c0 = ax0.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, normalized_temperature[curr_step], shading='auto', cmap=cmap, norm=norm)
                    cbar0 = fig.colorbar(c0, ax=ax0)
                    cbar0.set_label('Percent of Target',labelpad=20,fontsize='large')
                    cbar0.set_ticks(ticks)
                    cbar0.ax.tick_params(labelsize=12)
                    ax0.set_xlabel('X Position [cm]',fontsize='large')
                    ax0.set_ylabel('Y Position [cm]',fontsize='large')
                    ax0.tick_params(axis='x',labelsize=12)
                    ax0.tick_params(axis='y',labelsize=12)
                    ax0.set_aspect('equal', adjustable='box')
            
                    # Plot temperature
                    c1 = ax1.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, data['temperature_field'][curr_step], shading='auto', cmap='jet', vmin=min_temp, vmax=max_temp)
                    cbar1 = fig.colorbar(c1, ax=ax1)
                    cbar1.set_label('Temperature [K]',labelpad=20,fontsize='large')
                    cbar1.ax.tick_params(labelsize=12)
                    ax1.set_xlabel('X Position [cm]',fontsize='large')
                    ax1.set_ylabel('Y Position [cm]',fontsize='large')
                    ax1.tick_params(axis='x',labelsize=12)
                    ax1.tick_params(axis='y',labelsize=12)
                    ax1.set_aspect('equal', adjustable='box')
            
                    # Plot input
                    c2 = ax2.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, 1.0e-6*input_mesh, shading='auto', cmap='coolwarm', vmin=0.0, vmax=1.0e-6*env.max_input_mag)
                    cbar2 = fig.colorbar(c2, ax=ax2)
                    cbar2.set_label('Input Heat [MW/m^3]', labelpad=20,fontsize='large')
                    cbar2.ax.tick_params(labelsize=12)
                    ax2.set_xlabel('X Position [cm]',fontsize='large')
                    ax2.set_ylabel('Y Position [cm]',fontsize='large')
                    ax2.tick_params(axis='x',labelsize=12)
                    ax2.tick_params(axis='y',labelsize=12)
                    ax2.set_aspect('equal', adjustable='box')
            
                    # Set title and save
                    title_str = "Simulation Time: "+'{:.2f}'.format(data['time'][curr_step])+'s'
                    fig.suptitle(title_str,fontsize='xx-large')
                    plt.savefig(video_path+'/time_'+'{:.2f}'.format(data['time'][curr_step])+'.png', dpi=dpi)
                    plt.close()
            
            print(" ")
    
    # Make save paths
    if combine:
        print("Combining Results...")
        path = "results/P_Results"
        video_path = "results/P_Results/video"
        if not os.path.isdir(path):
            os.mkdir(path)
            os.mkdir(video_path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)
            os.mkdir(video_path)
        
        # Load previous results
        best = -1.0e20
        best_index = -1
        best_denom_const = 0.0
        best_loc_multiplier = 0.0
        denom_const_mesh, loc_multiplier_mesh = np.meshgrid(denom_const_set, loc_multiplier_set)
        load_data = {
            'data':[],
            'avg':np.zeros((len(denom_const_set), len(loc_multiplier_set))),
            'std':np.zeros((len(denom_const_set), len(loc_multiplier_set))),
        }
        for i in range(len(denom_const_set)):
            denom_const = denom_const_set[i]
            for j in range(len(loc_multiplier_set)):
                loc_multiplier = loc_multiplier_set[j]
                load_path = "results/P_"+'{:03.3f}'.format(denom_const)+"_"+'{:03.2f}'.format(loc_multiplier) + "/output"
                with open(load_path, 'rb') as file:
                    load = pickle.load(file)
                    load_data['data'].append(load['data'])
                    mean = np.mean(load['data']['r_per_episode'])
                    if mean > best:
                        best = mean
                        best_index = len(load_data['data'])-1
                        best_denom_const = denom_const
                        best_loc_multiplier = loc_multiplier
                    load_data['avg'][j, i] = mean
                    load_data['std'][j, i] = np.std(load['data']['r_per_episode'])
                    
        # Plot mean results
        fig = plt.figure()
        im = plt.gca().pcolormesh(denom_const_mesh, loc_multiplier_mesh, load_data['avg'], shading='auto')
        cbar = fig.colorbar(im, ax=plt.gca())
        cbar.set_label('Average Performance',labelpad=20,fontsize='large')
        cbar.ax.tick_params(labelsize=12)
        plt.xlabel("Denominator Constant",fontsize='large')
        plt.ylabel("Location Multiplier",fontsize='large')
        plt.xticks(denom_const_set, fontsize='large')
        plt.yticks(loc_multiplier_set, fontsize='large')
        plt.title("Average Reward Per Step",fontsize='xx-large')
        plt.gcf().set_size_inches(8.5, 5.5)
        plt.savefig(path+'/avg_perf.png', dpi=500)
        plt.close()
        
        # Plot std results
        fig = plt.figure()
        im = plt.gca().pcolormesh(denom_const_mesh, loc_multiplier_mesh, load_data['std'], shading='auto')
        cbar = fig.colorbar(im, ax=plt.gca())
        cbar.set_label('STD of Performance',labelpad=20,fontsize='large')
        cbar.ax.tick_params(labelsize=12)
        plt.xlabel("Denominator Constant",fontsize='large')
        plt.ylabel("Location Multiplier",fontsize='large')
        plt.xticks(denom_const_set, fontsize='large')
        plt.yticks(loc_multiplier_set, fontsize='large')
        plt.title("STD of Reward Per Step",fontsize='xx-large')
        plt.gcf().set_size_inches(8.5, 5.5)
        plt.savefig(path+'/std_perf.png', dpi=500)
        plt.close()
        
        # Plot best trajectory
        plt.clf()
        plt.title("Relative Difference Trajectory",fontsize='xx-large')
        plt.xlabel("Time [s]",fontsize='large')
        plt.ylabel("Relative Difference from Target Temperature [%]",fontsize='large')
        plt.plot(load_data['data'][best_index]['time'],100.0*np.array(load_data['data'][best_index]['temperature_rel_error']),c='k',lw=2.0)
        plt.plot(load_data['data'][best_index]['time'],100.0*np.array(load_data['data'][best_index]['temperature_max_error']),c='r',ls='--',lw=2.0)
        plt.plot(load_data['data'][best_index]['time'],100.0*np.array(load_data['data'][best_index]['temperature_min_error']),c='b',ls='--',lw=2.0)
        plt.legend(('Average', 'Maximum', 'Minimum'),loc='lower right',fontsize='large')
        plt.grid(which='major',axis='y')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/trajectory_"+'{:03.3f}'.format(best_denom_const)+"_"+'{:03.2f}'.format(best_loc_multiplier)+".png"
        plt.savefig(save_file, dpi = 500)
        plt.close()
        
        # Render best trajectory
        print("Rendering...")
        min_temp = 0.99*np.min(load_data['data'][best_index]['temperature_field'])
        max_temp = max(1.01*np.max(load_data['data'][best_index]['temperature_field']), 1.05*np.max(env.target_temp_mesh))
        normalized_temperature = 100.0*np.array(load_data['data'][best_index]['temperature_field'])/np.array(load_data['data'][best_index]['temperature_target'])
        min_normalized_temp = np.min(0.99*normalized_temperature)
        max_normalized_temp = np.max(1.01*normalized_temperature)
    
        # Make custom color map for normalized data
        lower = min(5.0*round(min_normalized_temp//5.0), 90.0)
        mid_lower = 99.0
        mid_upper = 101.0
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
        for curr_step in range(len(load_data['data'][best_index]['time'])):
    
            # Calculate input field
            input_magnitude = load_data['data'][best_index]['input_magnitude'][curr_step]
            input_location = load_data['data'][best_index]['input_location'][curr_step]
            input_mesh = input_magnitude * env.max_input_mag * np.exp(((env.mesh_cens_x_cords - input_location[0])**2 * env.exp_const) +
                                                                          (env.mesh_cens_y_cords - input_location[1])**2 * env.exp_const)
            input_mesh[input_mesh<0.01*env.max_input_mag] = 0.0
    
            # Make fig for temperature, cure, and input
            plt.cla()
            plt.clf()
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
            fig.set_size_inches(11,8.5)
    
            # Plot temperature percent
            c0 = ax0.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, normalized_temperature[curr_step], shading='auto', cmap=cmap, norm=norm)
            cbar0 = fig.colorbar(c0, ax=ax0)
            cbar0.set_label('Percent of Target',labelpad=20,fontsize='large')
            cbar0.set_ticks(ticks)
            cbar0.ax.tick_params(labelsize=12)
            ax0.set_xlabel('X Position [cm]',fontsize='large')
            ax0.set_ylabel('Y Position [cm]',fontsize='large')
            ax0.tick_params(axis='x',labelsize=12)
            ax0.tick_params(axis='y',labelsize=12)
            ax0.set_aspect('equal', adjustable='box')
    
            # Plot temperature
            c1 = ax1.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, load_data['data'][best_index]['temperature_field'][curr_step], shading='auto', cmap='jet', vmin=min_temp, vmax=max_temp)
            cbar1 = fig.colorbar(c1, ax=ax1)
            cbar1.set_label('Temperature [K]',labelpad=20,fontsize='large')
            cbar1.ax.tick_params(labelsize=12)
            ax1.set_xlabel('X Position [cm]',fontsize='large')
            ax1.set_ylabel('Y Position [cm]',fontsize='large')
            ax1.tick_params(axis='x',labelsize=12)
            ax1.tick_params(axis='y',labelsize=12)
            ax1.set_aspect('equal', adjustable='box')
    
            # Plot input
            c2 = ax2.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, 1.0e-6*input_mesh, shading='auto', cmap='coolwarm', vmin=0.0, vmax=1.0e-6*env.max_input_mag)
            cbar2 = fig.colorbar(c2, ax=ax2)
            cbar2.set_label('Input Heat [MW/m^3]', labelpad=20,fontsize='large')
            cbar2.ax.tick_params(labelsize=12)
            ax2.set_xlabel('X Position [cm]',fontsize='large')
            ax2.set_ylabel('Y Position [cm]',fontsize='large')
            ax2.tick_params(axis='x',labelsize=12)
            ax2.tick_params(axis='y',labelsize=12)
            ax2.set_aspect('equal', adjustable='box')
    
            # Set title and save
            title_str = "Simulation Time: "+'{:.2f}'.format(load_data['data'][best_index]['time'][curr_step])+'s'
            fig.suptitle(title_str,fontsize='xx-large')
            plt.savefig(video_path+'/time_'+'{:.2f}'.format(load_data['data'][best_index]['time'][curr_step])+'.png', dpi=dpi)
            plt.close()
        
    print("Done!")