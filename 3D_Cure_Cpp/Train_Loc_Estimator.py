# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:18:04 2021

@author: GKSch
"""

from Loc_Estimator import Estimator
import numpy as np
import random

def get_print_string(curr_epochs, num_epochs, batch_num, batch_count, num_batches, curr_loss, curr_weight, curr_bias):
    
    if batch_count == -1:
        epoch_str = "Epoch " + str(0) + "/" + str(num_epochs)
        batch_count_str = " | Batch " + str(0) + "/" + str(num_batches)
        batch_str = " | Set ----"
        loss_str = " | Loss = " + '{:.3f}'.format(0.000)
        weight_str = " | Weight = " + '{:.3f}'.format(1.000)
        bias_str = " | Bias = " + '{:.3f}'.format(0.000)
    else:
        epoch_str = "Epoch " + str(curr_epochs+1) + "/" + str(num_epochs)
        batch_count_str = " | Batch " + str(batch_count+1) + "/" + str(num_batches)
        batch_str = " | Set " + str(batch_num)
        if batch_num < 10:
            batch_str = batch_str + "   "
        elif batch_num < 100:
            batch_str = batch_str + "  "
        elif batch_num < 1000:
            batch_str = batch_str + " "
        loss_str = " | Loss = " + '{:.3f}'.format(round(curr_loss,3))
        weight_str = " | Weight = " + '{:.3f}'.format(round(curr_weight,3))
        bias_str = " | Bias = " + '{:.3f}'.format(round(curr_bias,3))
    
    return epoch_str+batch_count_str+batch_str+loss_str+weight_str+bias_str+" |                            "

if __name__ == "__main__":
    
    # Training hyperparameters
    x_dim = 256
    y_dim = 32
    alpha_zero = 20.0;
    alpha_last = 0.01;
    load_path = ""
    
    # Training data parameters
    num_traj = 5000
    samples_per_traj = 20
    samples_per_batch = 100
    num_epochs = 5;
    path = 'training_data/DCPD_GC2/Location_Estimator'
    
    # Buffers
    loss_buffer = []
    weight_buffer = []
    bias_buffer = []
    alpha_buffer = []
    
    # Calculted parameters
    num_batches = int(num_traj//(samples_per_batch//samples_per_traj))
    decay_rate = (alpha_last/alpha_zero) ** (1.0/(num_batches*num_epochs))
    
    print(get_print_string(0.0, num_epochs, 0.0, -1, num_batches, 0.0, 0.0, 0.0), end='\r')
    
    # Calculate the target front speed properties
    target_max = -10000000.0
    target_min = 10000000.0
    for i in range(num_batches):
        curr_target_file = path+'/target_data_' + str(i) + '.csv'
        target = np.genfromtxt(curr_target_file, delimiter=',')
        if target.max() > target_max:
            target_max = target.max()
        if target.min() < target_min:
            target_min = target.min()
        
    # Initialize the estimator
    estimator = Estimator(x_dim, y_dim, target_min, target_max, alpha_zero, decay_rate, load_path)
    
    # Run the epochs
    for curr_epoch in range(num_epochs):
    
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
            loss, weight, bias, alpha = estimator.learn(states, target)
            loss_buffer.append(loss)
            weight_buffer.append(weight)
            bias_buffer.append(bias)
            alpha_buffer.append(alpha)
        
            print(get_print_string(curr_epoch, num_epochs, access_order[curr_batch], curr_batch, num_batches, loss_buffer[-1], weight_buffer[-1], bias_buffer[-1]), end='\r')
        
    # Draw training data
    save_path = estimator.save(loss_buffer, weight_buffer, bias_buffer, alpha_buffer)
    estimator.draw_training_curve(max(2, num_batches * num_epochs // 100), save_path)
    print("Done!")