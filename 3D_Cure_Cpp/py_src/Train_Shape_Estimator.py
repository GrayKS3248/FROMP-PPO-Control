# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:18:04 2021

@author: GKSch
"""

from Shape_Estimator import Estimator
import numpy as np
import random

def get_print_string(batch_num, batch_count, num_batches, estimates_list):
    
    if batch_count == -1:
        batch_count_str = "Batch " + str(0) + "/" + str(num_batches)
        batch_str = " | Set ----"
        means_str = " | Mean = " + '{:.3f}'.format(0.000)
        stds_str = " | Std = " + '{:.3f}'.format(0.000)
    else:
        batch_count_str = "Batch " + str(batch_count+1) + "/" + str(num_batches)
        batch_str = " | Set " + str(batch_num)
        if batch_num < 10:
            batch_str = batch_str + "   "
        elif batch_num < 100:
            batch_str = batch_str + "  "
        elif batch_num < 1000:
            batch_str = batch_str + " "
        
        means = np.mean(np.array(estimates_list),axis=0)
        means_str = ""
        for i in range(len(means)): 
            means_str = means_str + " | Mean " + str(i) + " = " + '{:.3f}'.format(means[i])
            
        stds = np.std(np.array(estimates_list),axis=0)
        stds_str = ""
        for i in range(len(stds)): 
            stds_str = stds_str + " | Std " + str(i) + " = " + '{:.3f}'.format(stds[i])
    
    return batch_count_str+batch_str+means_str+stds_str+" |                            "

if __name__ == "__main__":
    
    # Parameters
    x_dim = 256
    y_dim = 32
    x_max = 0.05
    y_max = 0.008
    order = 3
    load_path = ""
    
    # Training data parameters
    num_traj = 5000
    samples_per_traj = 20
    samples_per_batch = 100
    path = '../training_data/DCPD_GC2/Location_Estimator'
    
    # Calculted parameters
    num_batches = int(num_traj//(samples_per_batch//samples_per_traj))
    
    print(get_print_string(0.0, -1, num_batches, 0.0), end='\r')
        
    # Initialize the estimator
    estimator = Estimator(x_dim, y_dim, x_max, y_max, order, load_path=load_path)
    estimates_list = []
    
    # Generate random batch order
    access_order = list(range(num_batches))
    random.shuffle(access_order)

    # Run the batches
    for curr_batch in range(num_batches):
            
        # Load and format current batches's training data 
        curr_states_file = path+'/states_data_' + str(access_order[curr_batch]) + '.csv'
        states = np.genfromtxt(curr_states_file, delimiter=',')
        states = states.reshape(samples_per_batch, estimator.x_dim, estimator.y_dim)
        curr_target_file = path+'/target_data_' + str(access_order[curr_batch]) + '.csv'
        targets = np.genfromtxt(curr_target_file, delimiter=',')
        
        # Update the weight and bias of the estimator
        for i in range(len(states)):
            estimates_list.append(estimator.get_estimate(states[i]))

        print(get_print_string(access_order[curr_batch], curr_batch, num_batches, estimates_list), end='\r')
                
    # Draw training data
    estimator.save(estimates_list)
    print("Done!")