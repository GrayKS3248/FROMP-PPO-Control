# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:53:17 2021

@author: GKSch
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import matplotlib.cm as cm
import re
from string import digits
from os.path import exists

if __name__ == "__main__":
    
    
    ## INPUTS ##
    ## ====================================================================================================================================================================================================== ##
    # Define load paths
    path = "../results/Quench/Quench_PPO_13" 
    performance_path = "../results/Quench"
    perf_1_file = "Quench_Semi-Controlled_Sim"
    perf_2_file = "Quench_Controlled_Sim"
    perf_combined_path = "../results/Quench"
    save_path = "../results/Quench"
    
    # Options
    NN_vis = True
    actor_vis = True
    critic_vis = True
        

    ## LOAD PERFORAMNCE 1 SIMULATION DATA ##
    ## ====================================================================================================================================================================================================== ##
    # Load all previous simulations, store their trajectory, front shape, and energy consumption
    if perf_combined_path == "":
        print("Loading simulation 1 data...")
        
        perf_1_speed = []
        perf_1_target = []
        
        temp_path = performance_path + "/" + perf_1_file + "_"
        sim_num = 1
        while exists(temp_path+str(sim_num)):
            curr_path = temp_path+str(sim_num)
            print(curr_path+"/output")
            with open(curr_path+"/output", 'rb') as file:
                dat = pickle.load(file)
                
                # Simulation data
                perf_1_target.append(dat['target'])
                perf_1_speed.append(dat['front_velocity'])
                
                # Iterate
                sim_num = sim_num + 1
                
                
    ## LOAD PERFORAMNCE 2 SIMULATION DATA ##
    ## ====================================================================================================================================================================================================== ##
    # Load all previous simulations, store their trajectory, front shape, and energy consumption
    if perf_combined_path == "":
        print("Loading simulation 2 data...")
        
        perf_2_target = []
        perf_2_speed = []
        
        temp_path = performance_path + "/" + perf_2_file + "_"
        sim_num = 1
        while exists(temp_path+str(sim_num)):
            curr_path = temp_path+str(sim_num)
            print(curr_path+"/output")
            with open(curr_path+"/output", 'rb') as file:
                dat = pickle.load(file)
                
                # Simulation data
                perf_2_target.append(dat['target'])
                perf_2_speed.append(dat['front_velocity'])
                
                # Iterate
                sim_num = sim_num + 1
       
            
    ## LOAD PRE-COMBINED DATA ##
    ## ====================================================================================================================================================================================================== ##
    # Load previously generated combined data
    print("Loading performance data...")
    
    if perf_combined_path != "":
        with open(perf_combined_path+"/perf_combined_data", 'rb') as file:
            data = pickle.load(file)
            
        for entry in data:
            globals()[entry] = data[entry]
        
        
    ## LOADING ##
    ## ====================================================================================================================================================================================================== ##
    # Load data
    print("Loading...")
    with open(path+"/output", 'rb') as file:
        dat = pickle.load(file)
        
        
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
        for agent in range(1,int(re.search(r'\d+$', path[-2:]).group())):
            path_no_digits = path.translate(remove_digits)
            with open(path_no_digits+str(agent)+"/output", 'rb') as file:
                curr_agent = pickle.load(file)
                fc1_nn_weights.append(curr_agent['actor'].fc1.weight.detach().cpu().numpy())
                fc1_nn_biases.append(curr_agent['actor'].fc1.bias.detach().cpu().numpy())
                fc2_nn_weights.append(curr_agent['actor'].fc2.weight.detach().cpu().numpy())
                fc2_nn_biases.append(curr_agent['actor'].fc2.bias.detach().cpu().numpy())
                fc3_nn_weights.append(curr_agent['actor'].fc3.weight.detach().cpu().numpy())
                fc3_nn_biases.append(curr_agent['actor'].fc3.bias.detach().cpu().numpy())
        fc1_nn_weights.append(dat['actor'].fc1.weight.detach().cpu().numpy())
        fc1_nn_biases.append(dat['actor'].fc1.bias.detach().cpu().numpy())
        fc2_nn_weights.append(dat['actor'].fc2.weight.detach().cpu().numpy())
        fc2_nn_biases.append(dat['actor'].fc2.bias.detach().cpu().numpy())
        fc3_nn_weights.append(dat['actor'].fc3.weight.detach().cpu().numpy())
        fc3_nn_biases.append(dat['actor'].fc3.bias.detach().cpu().numpy())
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
            plt.savefig(save_path+"/Actor_NN_"+str(network+1)+".svg", format='svg', dpi=1000)
            plt.close()
    
    
    ## ACTOR LEARNING CURVE ##
    ## ====================================================================================================================================================================================================== ##    
    # Calculate and plot relative performance data
    target_1 = np.mean(perf_1_target)
    target_2 = np.mean(perf_2_target)
    perf_1_speed_err = np.array(perf_1_speed)[:,int(np.ceil(0.33*len(np.array(perf_1_speed)[0,:]))):int(np.floor(0.66*len(np.array(perf_1_speed)[0,:])))]
    perf_1_speed_err = 100.0 * abs((target_1 - perf_1_speed_err) / target_1)
    perf_1_speed_err = perf_1_speed_err.flatten()
    perf_2_speed_err = np.array(perf_2_speed)[:,int(np.ceil(0.33*len(np.array(perf_2_speed)[0,:]))):int(np.floor(0.66*len(np.array(perf_2_speed)[0,:])))]
    perf_2_speed_err = 100.0 * abs((target_2 - perf_2_speed_err) / target_2)
    perf_2_speed_err = perf_2_speed_err.flatten()
    
    plt.clf()
    plt.gcf().set_size_inches(2.5,3.0)
    plt.ylim([-1, 12])
    plt.boxplot([perf_1_speed_err, perf_2_speed_err], showfliers=False, widths=(0.5,0.5))
    plt.ylabel('Front Speed Error [%]', fontsize=16)
    plt.xticks([1,2], labels=["Before\nReset", "After\nReset"], fontsize=16)
    plt.yticks([0, 4, 8, 12], fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path+"/actor_performance.svg", dpi = 500)
    plt.close()
    
    # Get the moving average and stdev of the learning curve
    if actor_vis:
        print("Rendering actor training data...")
        plt.clf()
        plt.gcf().set_size_inches(8.5,5.5)
        plt.xlabel("Trajectory",fontsize=16)
        plt.ylabel("Mean Reward per Trajectory Step",fontsize=16)
        window = ((len(dat['r_per_episode']) // 100) // 50) * 50
        if window > 1:
            rolling_std = np.array(pd.Series(dat['r_per_episode']).rolling(window).std())
            rolling_avg = np.array(pd.Series(dat['r_per_episode']).rolling(window).mean())
            rolling_std = rolling_std[~np.isnan(rolling_std)]
            rolling_avg = rolling_avg[~np.isnan(rolling_avg)]
            plt.fill_between(np.arange(len(rolling_avg)), rolling_avg+rolling_std, rolling_avg-rolling_std, color='r', alpha=0.25, lw=0.0)
            plt.plot(np.arange(len(rolling_avg)), rolling_avg, lw=2.0, color='r')
        
        else:
            plt.plot(np.arange(len(dat['r_per_episode'])), dat['r_per_episode'], lw=2.0, color='r')
            
        # Plot actor training curve
        plt.ylim(0.0,.9)
        plt.xlim(-1000,42500)
        plt.xticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000], fontsize=16)
        plt.yticks(fontsize=16)
        plt.title("(a) Control policy learning curve",fontsize=20, y=-0.25)
        plt.tight_layout()
        plt.savefig(save_path+"/actor_training.svg", dpi = 500)
        plt.close()
        
        
    ## CRITIC LEARNING CURVE ##
    ## ====================================================================================================================================================================================================== ##    
    # Get the moving average and stdev of the learning curve
    if critic_vis:
        print("Rendering critic training data...")
        plt.clf()
        plt.gcf().set_size_inches(8.5,5.5)
        plt.xlabel("Optimization Step",fontsize=16)
        plt.ylabel("Value Function Estimation MSE",fontsize=16)
        window = ((len(dat['r_per_episode']) // 100) // 50) * 50
        if window > 1:
            rolling_std = np.array(pd.Series(dat['value_error']).rolling(window).std())
            rolling_avg = np.array(pd.Series(dat['value_error']).rolling(window).mean())
            rolling_std = rolling_std[~np.isnan(rolling_std)]
            rolling_avg = rolling_avg[~np.isnan(rolling_avg)]
            plt.fill_between(np.arange(len(rolling_avg)), rolling_avg+rolling_std, rolling_avg-rolling_std, color='r', alpha=0.25, lw=0.0)
            plt.plot(np.arange(len(rolling_avg)), rolling_avg, lw=2.0, color='r')
            plt.ylim(10.0**((np.log10(np.min(rolling_avg))//0.5)*0.5), 10.0**((np.log10(np.max(rolling_avg))//0.5 + 1.0)*0.5)) 
        
        else:
            plt.plot(np.arange(len(dat['value_error'])), dat['value_error'], lw=2.0, color='r')
            plt.ylim(10.0**((np.log10(np.min(dat['value_error']))//0.5)*0.5), 10.0**((np.log10(np.max(dat['value_error']))//0.5 + 1.0)*0.5))
            
        # Plot actor training curve
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title("Critic learning curve",fontsize=20, y=-0.25)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(save_path+"/critic_training.svg", dpi = 500)
        plt.close()
        
        
    ## SAVE DATA USED TO GENERATE PERFORMANCE GRAPHS ##
    ## ====================================================================================================================================================================================================== ##
    print("Saving...")
    data = {
        'perf_1_target' : perf_1_target,
        'perf_2_target' : perf_2_target,
        'perf_1_speed' : perf_1_speed,
        'perf_2_speed' : perf_2_speed
    }
    with open(save_path + "/perf_combined_data", 'wb') as file:
            pickle.dump(data, file)
    
    print("Done!")