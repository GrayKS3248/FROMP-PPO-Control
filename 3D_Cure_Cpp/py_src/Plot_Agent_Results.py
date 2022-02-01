# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:53:17 2021

@author: GKSch
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy import integrate
import matplotlib.cm as cm
import re
from string import digits

if __name__ == "__main__":
    
    
    ## INPUTS ##
    ## ====================================================================================================================================================================================================== ##
    # Define load paths
    controlled_path = "../results/PPO_5" 
    uncontrolled_path = "../results/SIM_2" 
    random_path = "../results/SIM_1" 
    
    # Define the temperature at which adiabatic, uncontrolled front speed would exactly equal target speed
    req_uncontrolled_temp = 306.15;
    
    # Define monomer properties
    initial_cure = 0.07
    initial_temp = 298.15
    Hr = 350000.0
    Cp = 1600.0
    rho = 980.0
    volume = 0.0000004
    area = 0.000916
    h = 20.0
    ambient_temp = 293.65
    
    # Options
    NN_vis = False
    trajectory_vis = True
    energy_vis = True
    actor_vis = True
    critic_vis = True
    
    
    ## LOADING ##
    ## ====================================================================================================================================================================================================== ##
    # Load data
    print("Loading...")
    with open(controlled_path+"/output", 'rb') as file:
        controlled = pickle.load(file)
    with open(uncontrolled_path+"/output", 'rb') as file:
        uncontrolled = pickle.load(file)
    with open(random_path+"/output", 'rb') as file:
        random = pickle.load(file)
        
        
    ## NN VISUALIZATION ##
    ## ====================================================================================================================================================================================================== ##
    # Load all previous agents, store their NN data
    if NN_vis:
        print("Loading NN visualization data...")
        fc1_nn_weights = []
        fc1_nn_biases = []
        fc2_nn_weights = []
        fc2_nn_biases = []
        fc3_nn_weights = []
        fc3_nn_biases = []
        remove_digits = str.maketrans('', '', digits)
        for agent in range(1,int(re.search(r'\d+$', controlled_path[-2:]).group())):
            controlled_path_no_digits = controlled_path.translate(remove_digits)
            with open(controlled_path_no_digits+str(agent)+"/output", 'rb') as file:
                curr_agent = pickle.load(file)
                fc1_nn_weights.append(curr_agent['actor'].fc1.weight.detach().cpu().numpy())
                fc1_nn_biases.append(curr_agent['actor'].fc1.bias.detach().cpu().numpy())
                fc2_nn_weights.append(curr_agent['actor'].fc2.weight.detach().cpu().numpy())
                fc2_nn_biases.append(curr_agent['actor'].fc2.bias.detach().cpu().numpy())
                fc3_nn_weights.append(curr_agent['actor'].fc3.weight.detach().cpu().numpy())
                fc3_nn_biases.append(curr_agent['actor'].fc3.bias.detach().cpu().numpy())
        fc1_nn_weights.append(controlled['actor'].fc1.weight.detach().cpu().numpy())
        fc1_nn_biases.append(controlled['actor'].fc1.bias.detach().cpu().numpy())
        fc2_nn_weights.append(controlled['actor'].fc2.weight.detach().cpu().numpy())
        fc2_nn_biases.append(controlled['actor'].fc2.bias.detach().cpu().numpy())
        fc3_nn_weights.append(controlled['actor'].fc3.weight.detach().cpu().numpy())
        fc3_nn_biases.append(controlled['actor'].fc3.bias.detach().cpu().numpy())
        fc1_nn_weights = np.array(fc1_nn_weights)
        fc1_nn_biases = np.array(fc1_nn_biases)
        fc2_nn_weights = np.array(fc2_nn_weights)
        fc2_nn_biases = np.array(fc2_nn_biases)
        fc3_nn_weights = np.array(fc3_nn_weights)
        fc3_nn_biases = np.array(fc3_nn_biases)
        
        # Calculate NN weight and bias ranges and normalize from -1 to 1
        max_weight = max((np.max((fc1_nn_weights)), np.max((fc2_nn_weights)), np.max((fc3_nn_weights))))
        min_weight = min((np.min((fc1_nn_weights)), np.min((fc2_nn_weights)), np.min((fc3_nn_weights))))
        max_bias = max((np.max((fc1_nn_biases)), np.max((fc2_nn_biases)), np.max((fc3_nn_biases))))
        min_bias = min((np.min((fc1_nn_biases)), np.min((fc2_nn_biases)), np.min((fc3_nn_biases))))
        fc1_nn_weights = np.round(255.0 * ((1.0 + (2.0*fc1_nn_weights - max_weight - min_weight) / (max_weight - min_weight)) / 2.0))
        fc1_nn_biases = np.round(255.0 * ((1.0 + (2.0*fc1_nn_biases - max_bias - min_bias) / (max_bias - min_bias)) / 2.0))
        fc2_nn_weights = np.round(255.0 * ((1.0 + (2.0*fc2_nn_weights - max_weight - min_weight) / (max_weight - min_weight)) / 2.0))
        fc2_nn_biases = np.round(255.0 * ((1.0 + (2.0*fc2_nn_biases - max_bias - min_bias) / (max_bias - min_bias)) / 2.0))
        fc3_nn_weights = np.round(255.0 * ((1.0 + (2.0*fc3_nn_weights - max_weight - min_weight) / (max_weight - min_weight)) / 2.0))
        fc3_nn_biases = np.round(255.0 * ((1.0 + (2.0*fc3_nn_biases - max_bias - min_bias) / (max_bias - min_bias)) / 2.0))
        
        # Create array of NN node locations
        num_inputs = len(fc1_nn_weights[0,0,:])
        num_nodes_1 = len(fc1_nn_biases[0,:])
        num_nodes_2 = len(fc2_nn_biases[0,:])
        num_outputs = len(fc3_nn_biases[0,:])
        width = max(num_inputs, num_nodes_1, num_nodes_2, num_outputs)
        
        input_y = np.linspace(0+((width-1.0)*(1.0-num_inputs/width))/2.0 ,(width-1.0)-((width-1.0)*(1.0-num_inputs/width))/2.0, num_inputs)
        input_bias_y = np.linspace(0+((width-1.0)*(1.0-num_nodes_1/width))/2.0 ,(width-1.0)-((width-1.0)*(1.0-num_nodes_1/width))/2.0, num_nodes_1)
        layer_1_y = np.linspace(0+((width-1.0)*(1.0-num_nodes_1/width))/2.0 ,(width-1.0)-((width-1.0)*(1.0-num_nodes_1/width))/2.0, num_nodes_1)
        layer_1_bias_y = np.linspace(0+((width-1.0)*(1.0-num_nodes_2/width))/2.0 ,(width-1.0)-((width-1.0)*(1.0-num_nodes_2/width))/2.0, num_nodes_2)
        layer_2_y = np.linspace(0+((width-1.0)*(1.0-num_nodes_2/width))/2.0 ,(width-1.0)-((width-1.0)*(1.0-num_nodes_2/width))/2.0, num_nodes_2)
        layer_2_bias_y = np.linspace(0+((width-1.0)*(1.0-num_outputs/width))/2.0 ,(width-1.0)-((width-1.0)*(1.0-num_outputs/width))/2.0, num_outputs)
        output_y = np.linspace(0+((width-1.0)*(1.0-num_outputs/width))/2.0 ,(width-1.0)-((width-1.0)*(1.0-num_outputs/width))/2.0, num_outputs)
        input_x = 0.0 * np.ones(num_inputs)
        input_bias_x = 9.0 * np.ones(num_nodes_1)
        layer_1_x = 10.0 * np.ones(num_nodes_1)
        layer_1_bias_x = 19.0 * np.ones(num_nodes_2)
        layer_2_x = 20.0 * np.ones(num_nodes_2)
        layer_2_bias_x = 29.0 * np.ones(num_outputs)
        output_x = 30.0 * np.ones(num_outputs)
        
        # Plot each iteration of the NN
        for network in range(len(fc1_nn_weights)):
            
            # Plot nodes and biases
            print("Rendering NN " + str(network+1) + " / " + str(len(fc1_nn_weights)) + "...")
            plt.clf()
            plt.scatter(input_x, input_y,c='k',s=2.)
            plt.scatter(input_bias_x, input_bias_y,s=2.,marker="s",c=cm.seismic(np.int32(fc1_nn_biases[network,:])))
            plt.scatter(layer_1_x, layer_1_y,c='k',s=2.)
            plt.scatter(layer_1_bias_x, layer_1_bias_y,s=2.,marker="s",c=cm.seismic(np.int32(fc2_nn_biases[network,:])))
            plt.scatter(layer_2_x, layer_2_y,c='k',s=2.)
            plt.scatter(layer_2_bias_x, layer_2_bias_y,s=2.,marker="s",c=cm.seismic(np.int32(fc3_nn_biases[network,:])))
            plt.scatter(output_x, output_y,c='k',s=2.)
                        
            # Plot Layer 1
            std_connection_weights = []
            for node in range(num_inputs):
                std_connection_weights.append(np.std(fc1_nn_weights[network,:,node]))
            node_plot_order = np.argsort(std_connection_weights)
            for node in range(num_inputs):
                for connection in range(num_nodes_1):
                    start_x = input_x[node_plot_order[node]]
                    end_x = input_bias_x[connection]
                    start_y = input_y[node_plot_order[node]]
                    end_y = input_bias_y[connection]
                    plt.plot([start_x, end_x], [start_y, end_y], lw=0.25,c=cm.seismic(np.int32(fc1_nn_weights[network,connection,node_plot_order[node]])),alpha=0.25)
                    if node==0:
                        start_x = input_bias_x[connection]
                        end_x = layer_1_x[connection]
                        start_y = input_bias_y[connection]
                        end_y = layer_1_y[connection]
                        plt.plot([start_x, end_x], [start_y, end_y], lw=1.0,c=cm.seismic(np.int32(fc1_nn_biases[network,connection])))
                       
            # Plot Layer 2
            std_connection_weights = []
            for node in range(num_nodes_1):
                std_connection_weights.append(np.std(fc2_nn_weights[network,:,node]))
            node_plot_order = np.argsort(std_connection_weights)
            for node in range(num_nodes_1):
                for connection in range(num_nodes_2):
                    start_x = layer_1_x[node_plot_order[node]]
                    end_x = layer_1_bias_x[connection]
                    start_y = layer_1_y[node_plot_order[node]]
                    end_y = layer_1_bias_y[connection]
                    plt.plot([start_x, end_x], [start_y, end_y], lw=0.25,c=cm.seismic(np.int32(fc2_nn_weights[network,connection,node_plot_order[node]])),alpha=0.25)
                    if node==0:
                        start_x = layer_1_bias_x[connection]
                        end_x = layer_2_x[connection]
                        start_y = layer_1_bias_y[connection]
                        end_y = layer_2_y[connection]
                        plt.plot([start_x, end_x], [start_y, end_y], lw=1.0,c=cm.seismic(np.int32(fc2_nn_biases[network,connection]))) 
                       
            # Plot Layer 3
            std_connection_weights = []
            for node in range(num_nodes_2):
                std_connection_weights.append(np.std(fc1_nn_weights[network,:,node]))
            node_plot_order = np.argsort(std_connection_weights)
            for node in range(num_nodes_2):
                for connection in range(num_outputs):
                    start_x = layer_2_x[node_plot_order[node]]
                    end_x = layer_2_bias_x[connection]
                    start_y = layer_2_y[node_plot_order[node]]
                    end_y = layer_2_bias_y[connection]
                    plt.plot([start_x, end_x], [start_y, end_y], lw=0.25,c=cm.seismic(np.int32(fc3_nn_weights[network,connection,node_plot_order[node]])),alpha=0.25)
                    if node==0:
                        start_x = layer_2_bias_x[connection]
                        end_x = output_x[connection]
                        start_y = layer_2_bias_y[connection]
                        end_y = output_y[connection]
                        plt.plot([start_x, end_x], [start_y, end_y], lw=1.0,c=cm.seismic(np.int32(fc3_nn_biases[network,connection])))
            
            # Format and save figure
            print("Saving NN " + str(network+1) + " / " + str(len(fc1_nn_weights)) + "...")
            plt.xticks([])
            plt.yticks([])
            plt.ylim([-0.05*(width-1.0), (width-1.0)+0.05*(width-1.0)])
            plt.xlim([-1.5, 31.5])
            fig = plt.gcf()
            fig.set_size_inches(10,10)
            plt.savefig("../results/Actor_NN_"+str(network+1)+".svg", format='svg', dpi=1000)
            plt.close()
        
        
    ## TRAJECTORY VISUALIZATION ##
    ## ====================================================================================================================================================================================================== ##    
    # Plot target trajectory
    if trajectory_vis:
        print("Rendering trajectory visualization...")
        plt.clf()
        if controlled['control_speed']:
            # Plot speed trajectory
            plt.clf()
            plt.title("Front Speed",fontsize='xx-large')
            plt.xlabel("Simulation Time [s]",fontsize='large')
            plt.ylabel("Front Speed [mm/s]",fontsize='large')
            plt.plot(uncontrolled['time'], 1000.0*uncontrolled['front_velocity'],c='g',lw=2.5,label='Uncontrolled')
            plt.plot(random['time'], 1000.0*random['front_velocity'],c='b',lw=2.5,label='Random')
            plt.plot(controlled['time'], 1000.0*controlled['front_velocity'],c='r',lw=2.5,label='Controlled')
            plt.plot(controlled['time'], 1000.0*controlled['target'],c='k',ls='--',lw=2.5,label='Target')
            plt.ylim(0.0, 1500.0*np.max(controlled['target']))
            plt.xlim(0.0, np.round(controlled['time'][-1]))
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("../results/speed.svg", format='svg', dpi = 1000)
            plt.close()
            
            # Plot front temperature trajectory
            plt.clf()
            uncontrolled_sorted_mean_front_x_locations = 1000.0*np.array(sorted(uncontrolled['mean_front_x_locations']))
            uncontrolled_sorted_front_temperature = np.array([x for _, x in sorted(zip(uncontrolled['mean_front_x_locations'], uncontrolled['front_temperature']))])-273.15
            random_sorted_mean_front_x_locations = 1000.0*np.array(sorted(random['mean_front_x_locations']))
            random_sorted_front_temperature = np.array([x for _, x in sorted(zip(random['mean_front_x_locations'], random['front_temperature']))])-273.15
            controlled_sorted_mean_front_x_locations = 1000.0*np.array(sorted(controlled['mean_front_x_locations']))
            controlled_sorted_front_temperature = np.array([x for _, x in sorted(zip(controlled['mean_front_x_locations'], controlled['front_temperature']))])-273.15
            plt.title("Front Temperature",fontsize='xx-large')
            plt.xlabel("Location [mm]",fontsize='large')
            plt.ylabel("Front Temperature [C]",fontsize='large')
            plt.plot(uncontrolled_sorted_mean_front_x_locations, uncontrolled_sorted_front_temperature, c='g', lw=2.5, label='Uncontrolled')
            plt.plot(random_sorted_mean_front_x_locations, random_sorted_front_temperature, c='b', lw=2.5, label='Random')
            plt.plot(controlled_sorted_mean_front_x_locations, controlled_sorted_front_temperature, c='r', lw=2.5, label='Controlled')
            plt.ylim(0.0, 1.025*(max(max(controlled['front_temperature']), max(uncontrolled['front_temperature']), max(random['front_temperature']))-273.15))
            plt.xlim(0.0, 1000.0*controlled['mesh_x_z0'][-1,0])
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large') 
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("../results/temperature.svg", format='svg', dpi = 1000)
            plt.close()
            
        else:
            # Plot speed trajectory
            plt.clf()
            plt.title("Front Speed",fontsize='xx-large')
            plt.xlabel("Simulation Time [s]",fontsize='large')
            plt.ylabel("Front Speed [mm/s]",fontsize='large')
            plt.plot(uncontrolled['time'], 1000.0*uncontrolled['front_velocity'],c='g',lw=2.5,label='Uncontrolled')
            plt.plot(random['time'], 1000.0*random['front_velocity'],c='b',lw=2.5,label='Random')
            plt.plot(controlled['time'], 1000.0*controlled['front_velocity'],c='r',lw=2.5,label='Controlled')
            plt.ylim(0.0, 1500.0*max(max(controlled['front_velocity']), max(uncontrolled['front_velocity']), max(random['front_velocity'])))
            plt.xlim(0.0, np.round(controlled['time'][-1]))
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("../results/speed.svg", format='svg', dpi = 1000)
            plt.close()
            
            # Plot front temperature trajectory
            plt.clf()
            uncontrolled_sorted_mean_front_x_locations = 1000.0*np.array(sorted(uncontrolled['mean_front_x_locations']))
            uncontrolled_sorted_front_temperature = np.array([x for _, x in sorted(zip(uncontrolled['mean_front_x_locations'], uncontrolled['front_temperature']))])-273.15
            random_sorted_mean_front_x_locations = 1000.0*np.array(sorted(random['mean_front_x_locations']))
            random_sorted_front_temperature = np.array([x for _, x in sorted(zip(random['mean_front_x_locations'], random['front_temperature']))])-273.15
            controlled_sorted_mean_front_x_locations = 1000.0*np.array(sorted(controlled['mean_front_x_locations']))
            controlled_sorted_front_temperature = np.array([x for _, x in sorted(zip(controlled['mean_front_x_locations'], controlled['front_temperature']))])-273.15
            plt.title("Front Temperature",fontsize='xx-large')
            plt.xlabel("Location [mm]",fontsize='large')
            plt.ylabel("Front Temperature [C]",fontsize='large')
            plt.plot(uncontrolled_sorted_mean_front_x_locations, uncontrolled_sorted_front_temperature, c='g', lw=2.5, label='Uncontrolled')
            plt.plot(random_sorted_mean_front_x_locations, random_sorted_front_temperature, c='b', lw=2.5, label='Random')
            plt.plot(controlled_sorted_mean_front_x_locations, controlled_sorted_front_temperature, c='r', lw=2.5, label='Controlled')
            plt.plot(controlled_sorted_mean_front_x_locations, controlled['target']-273.15, c='k', ls='--', lw=2.5, label='Target')
            plt.ylim(0.0, 1.025*(controlled['target']-273.15))
            plt.xlim(0.0, 1000.0*controlled['mesh_x_z0'][-1,0])
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large') 
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig("../results/temperature.svg", format='svg', dpi = 1000)
            plt.close()
    
    ## ENERGY CONSUMPTION VISUALIZATION ##
    ## ====================================================================================================================================================================================================== ##    
    # Calculate energy addition required for uncontrolled speed to match target speed
    if energy_vis:
        print("Rendering energy visualization...")
        plt.clf()
        T_max = (Hr * (1.0 - initial_cure))/Cp + initial_temp;
        uncontrolled_initial_temperature = np.mean(uncontrolled['temperature_field'][0,:,:] * (T_max - initial_temp) + initial_temp)
        required_delta_T = req_uncontrolled_temp - uncontrolled_initial_temperature
        required_energy = Cp*required_delta_T*rho*volume + uncontrolled['time'][-1]*h*area*(req_uncontrolled_temp-ambient_temp)
        
        # Calculate energy usage
        controlled_energy = integrate.cumtrapz(controlled['power'], x=controlled['time'])
        controlled_energy = np.insert(controlled_energy, 0, 0.0)
        uncontrolled_energy = integrate.cumtrapz(uncontrolled['power'], x=uncontrolled['time'])
        uncontrolled_energy = np.insert(uncontrolled_energy, 0, 0.0)
        random_energy = integrate.cumtrapz(random['power'], x=random['time'])
        random_energy = np.insert(random_energy, 0, 0.0)
        
        # Plot energy trajectory
        plt.plot(uncontrolled['time'], uncontrolled_energy,c='g',lw=2.5,label='Uncontrolled')
        plt.plot(random['time'], random_energy,c='b',lw=2.5,label='Random')
        plt.plot(controlled['time'], controlled_energy,c='r',lw=2.5,label='Controlled')
        plt.plot(random['time'], required_energy*np.ones(len(random['time'])),c='k',lw=2.5,ls="--",label='Required')
        plt.xlim(0.0, np.round(controlled['time'][-1]))
        plt.gcf().set_size_inches(8.5, 5.5)
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.title("External Energy Input",fontsize='xx-large')
        plt.xlabel("Simulation Time [s]",fontsize='large')
        plt.ylabel("Cumulative Energy Consumed [J]",fontsize='large')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig("../results/energy.svg", format='svg', dpi = 1000)
        plt.close()
    
    
    ## ACTOR LEARNING CURVE ##
    ## ====================================================================================================================================================================================================== ##    
    # Get the moving average and stdev of the learning curve
    if actor_vis:
        print("Rendering actor training data...")
        plt.clf()
        window = len(controlled['r_per_episode']) // 50
        if window > 1:
            rolling_std = np.array(pd.Series(controlled['r_per_episode']).rolling(window).std())
            rolling_avg = np.array(pd.Series(controlled['r_per_episode']).rolling(window).mean())
            rolling_std = rolling_std[~np.isnan(rolling_std)]
            rolling_avg = rolling_avg[~np.isnan(rolling_avg)]
            
        # Plot actor training curve
        plt.title('Rolling Actor Learning Curve, Window='+str(window),fontsize='xx-large')
        plt.xlabel("Trajectory",fontsize='large', labelpad=15)
        plt.ylabel("Mean Reward per Trajectory Step",fontsize='large', labelpad=15)
        plt.fill_between(np.arange(len(rolling_avg)), rolling_avg+rolling_std, rolling_avg-rolling_std, color='r', alpha=0.25, lw=0.0)
        plt.plot(np.arange(len(rolling_avg)), rolling_avg, lw=2.0, color='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.ylim([0.0,1.0])
        plt.gcf().set_size_inches(8.5, 5.5)
        plt.savefig("../results/actor_training.svg", format='svg', dpi = 1000)
        plt.close()
        
        
    ## CRITIC LEARNING CURVE ##
    ## ====================================================================================================================================================================================================== ##    
    # Get the moving average and stdev of the learning curve
    if critic_vis:
        print("Rendering critic training data...")
        plt.clf()
        window = len(controlled['value_error']) // 100
        if window > 1:
            rolling_std = np.array(pd.Series(controlled['value_error']).rolling(window).std())
            rolling_avg = np.array(pd.Series(controlled['value_error']).rolling(window).mean())
            rolling_std = rolling_std[~np.isnan(rolling_std)]
            rolling_avg = rolling_avg[~np.isnan(rolling_avg)]
            
        # Plot actor training curve
        plt.title('Rolling Critic Learning Curve, Window='+str(window),fontsize='xx-large')
        plt.xlabel("Optimization Step",fontsize='large', labelpad=15)
        plt.ylabel("Value Function Estimation MSE",fontsize='large', labelpad=15)
        plt.fill_between(np.arange(len(rolling_avg)), rolling_avg+rolling_std, rolling_avg-rolling_std, color='r', alpha=0.25, lw=0.0)
        plt.plot(np.arange(len(rolling_avg)), rolling_avg, lw=2.0, color='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.ylim([10.0**((np.log10(np.min(rolling_avg))//0.5)*0.5), 10.0**((np.log10(np.max(rolling_avg))//0.5 + 1.0)*0.5)])
        plt.yscale("log")
        plt.gcf().set_size_inches(8.5, 5.5)
        plt.savefig("../results/critic_training.svg", format='svg', dpi = 1000)
        plt.close()
    
    print("Done!")