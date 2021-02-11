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
    denom_const_set = np.linspace(0.001, 0.1, 10)
    loc_multiplier_set = np.linspace(0.01, 1.0, 10)
    render = False
    plot = False
    combine = True
    
    # Simulation parameters
    total_trajectories = 20
    control = False
    uniform_target = True
    split_target = False
    random_target = False

    # Agent hyperparameters
    steps_per_trajectory = 3000

    # Rendering parameters
    frame_multiplier = 3.0
    dpi = 100

    # Temperature to color fit data (degrees celcius to RGB values [0.0,255.0])
    ## RED_1 a0 + a1*cos(x*w) + b1*sin(x*w)
    ## RED_2 a0 + a1*cos(x*w) + b1*sin(x*w) + a2*cos(2*x*w) + b2*sin(2*x*w)
    red_1_max = 30.96
    red_2_min = 32.89
    red_1_a0 =  75.7205687988528
    red_1_a1 = -83.3811961554596
    red_1_b1 =  2.16921777958013
    red_1_w =   5.66519794764997
    red_2_a0 =  13.4423061030970
    red_2_a1 =  11.4310759961892
    red_2_b1 = -19.9304438902058
    red_2_a2 =  1.03633786169181
    red_2_b2 = -6.91674577098558
    red_2_w =   1.41780395653457
    
    ## GREEN a0 + a1*cos(x*w) + b1*sin(x*w) + a2*cos(2*x*w) + b2*sin(2*x*w) + a3*cos(3*x*w) + b3*sin(3*x*w)

    green_a0 =  41.8497458579400
    green_a1 =  50.5337801853276
    green_b1 =  61.2631767821492
    green_a2 =  72.4315809903627
    green_b2 = -14.1293352372328
    green_a3 =   0.8477446584536
    green_b3 = -21.8074000100672
    green_w =    0.8050372098715
    
    ## BLUE a0 + a1*cos(x*w) + b1*sin(x*w) + a2*cos(2*x*w) + b2*sin(2*x*w) + a3*cos(3*x*w) + b3*sin(3*x*w) + a4*cos(4*x*w) + b4*sin(4*x*w)
    blue_a0 = -52917511.6179542
    blue_a1 =  60193891.2132188
    blue_b1 = -60277928.6140266
    blue_a2 =  62362.8148054820
    blue_b2 =  43381416.1414114
    blue_a3 = -9057653.45279618
    blue_b3 = -9016664.70251709
    blue_a4 =  1667857.69952913
    blue_b4 = -5415.39415551451
    blue_w =   0.16819988522403

    # Make the colormap for the fit data
    red_1_fit = lambda x : (red_1_a0 + red_1_a1*np.cos(x*red_1_w) + red_1_b1*np.sin(x*red_1_w)) if (x>=30.0 and x<= red_1_max) else 0.0
    red_2_fit = lambda x : (red_2_a0 + red_2_a1*np.cos(x*red_2_w) + red_2_b1*np.sin(x*red_2_w) + red_2_a2*np.cos(2*x*red_2_w) + red_2_b2*np.sin(2*x*red_2_w)) if (x>=red_2_min and x<=36.0) else 0.0
    green_fit = lambda x : green_a0 + green_a1*np.cos(x*green_w) + green_b1*np.sin(x*green_w) + green_a2*np.cos(2*x*green_w) + green_b2*np.sin(2*x*green_w) + green_a3*np.cos(2*x*green_w) + green_b3*np.sin(2*x*green_w)
    blue_fit = lambda x : blue_a0 + blue_a1*np.cos(x*blue_w) + blue_b1*np.sin(x*blue_w) + blue_a2*np.cos(2*x*blue_w) + blue_b2*np.sin(2*x*blue_w) + blue_a3*np.cos(3*x*blue_w) + blue_b3*np.sin(3*x*blue_w) + blue_a4*np.cos(4*x*blue_w) + blue_b4*np.sin(4*x*blue_w)
    rgb = lambda x : (max((red_1_fit(x-273.15) + red_2_fit(x-273.15)), 0.0)/255.0, max(green_fit(x-273.15),0.0)/255.0, max(blue_fit(x-273.15),0.0)/255.0)  if (x-273.15>=29.0 and x-273.15<=36.0) else (0.0, 0.0, 0.0)

    # Initialize simulation set
    for i in range(len(denom_const_set)):
        denom_const = denom_const_set[i]
        for j in range(len(loc_multiplier_set)):
            loc_multiplier = loc_multiplier_set[j]
            
            # Calculated env and agent parameters
            env = fes.FES(control=control, loc_multiplier=loc_multiplier, uniform_target=uniform_target, split_target=split_target, random_target=random_target)
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
                
                # Make a custom colormap for the temperature data
                vals = np.ones((2000, 4))
                temps = np.linspace(min_temp, max_temp, 2000)
                for i in range(2000):
                    vals[i, 0] = rgb(temps[i])[0]
                    vals[i, 1] = rgb(temps[i])[1]
                    vals[i, 2] = rgb(temps[i])[2]
                cmap_2 = clr.ListedColormap(vals)
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
                    fig.set_size_inches(6.0,12.0)
            
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
                    c1 = ax1.pcolormesh(100.0*env.mesh_cens_x_cords, 100.0*env.mesh_cens_y_cords, data['temperature_field'][curr_step], shading='gouraud', cmap=cmap_2, vmin=min_temp, vmax=max_temp)
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
        
        # Make a custom colormap for the temperature data
        vals = np.ones((2000, 4))
        temps = np.linspace(min_temp, max_temp, 2000)
        for i in range(2000):
            vals[i, 0] = rgb(temps[i])[0]
            vals[i, 1] = rgb(temps[i])[1]
            vals[i, 2] = rgb(temps[i])[2]
        cmap_2 = clr.ListedColormap(vals)
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
            fig.set_size_inches(6.0,12.0)
    
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
            c1 = ax1.pcolor(100.0*env.mesh_verts_x_coords, 100.0*env.mesh_verts_y_coords, load_data['data'][best_index]['temperature_field'][curr_step], shading='auto', cmap=cmap_2, vmin=min_temp, vmax=max_temp)
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