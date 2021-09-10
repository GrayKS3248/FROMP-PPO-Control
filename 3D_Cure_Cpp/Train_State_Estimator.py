# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:18:04 2021

@author: GKSch
"""

from State_Estimator import State_Estimator
import numpy as np
import random

def get_print_string(batch_num, batch_count, num_batches, curr_loss):
    
    if batch_count == -1:
        batch_count_str = "Batch " + str(0) + "/" + str(num_batches)
        batch_str = " | Set ----"
        loss_str = " | Loss = " + '{:.3f}'.format(0.000)
    else:
        batch_count_str = "Batch " + str(batch_count+1) + "/" + str(num_batches)
        batch_str = " | Set " + str(batch_num)
        if batch_num < 10:
            batch_str = batch_str + "   "
        elif batch_num < 100:
            batch_str = batch_str + "  "
        elif batch_num < 1000:
            batch_str = batch_str + " "
        loss_str = " | Loss = " + '{:.3f}'.format(round(curr_loss,3))
    
    return batch_count_str+batch_str+loss_str+" |"

if __name__ == "__main__":
    
    # Training hyperparameters
    x_dim = 256
    y_dim = 32
    bottleneck = 128
    kernal_size = 5
    num_states = 4
    time_between_state_frames = 0.2
    alpha_zero = 1.0e-3;
    alpha_last = 1.0e-4;
    
    # Training data parameters
    num_traj = 2000
    samples_per_traj = 20
    samples_per_batch = 100
    num_epochs = 1;
    path = 'training_data/DCPD_GC2/Speed_Estimator_4_0.20'
    
    # Buffers
    loss_buffer = []
    lr_buffer = []
    
    # Calculted parameters
    num_batches = int(num_traj//(samples_per_batch//samples_per_traj))
    decay_rate = (alpha_last/alpha_zero) ** (1.0/(num_batches*num_epochs))
    
    # Initialize the estimator
    estimator = State_Estimator(x_dim, y_dim, bottleneck, kernal_size, num_states, time_between_state_frames, alpha_zero, decay_rate, num_epochs, samples_per_batch)
    
    # Generate random batch order
    access_order = list(range(num_batches))
    random.shuffle(access_order)
    
    print(get_print_string(0.0, -1, num_batches, 0.0), end='\r')
    
    # Calculate the target front speed properties
    target_mean = 0.0
    target_std = 0.0
    for i in range(num_batches):
        curr_target_file = path+'/target_data_' + str(access_order[i]) + '.csv'
        target = np.genfromtxt(curr_target_file, delimiter=',')
        target_mean = target_mean + target.mean()
        target_std = target_std + target.std()
    target_mean = (target_mean / num_batches)
    target_std = (target_std / num_batches)
        
    # Run the batches
    for i in range(num_batches):
            
        # Load and format current batches's training data 
        curr_states_file = path+'/states_data_' + str(access_order[i]) + '.csv'
        states = np.genfromtxt(curr_states_file, delimiter=',')
        states = states.reshape(samples_per_batch, num_states, estimator.x_dim, estimator.y_dim)
        curr_target_file = path+'/target_data_' + str(access_order[i]) + '.csv'
        target = np.genfromtxt(curr_target_file, delimiter=',')
        
        # Normalize the target data
        target = (target - target_mean) / target_std
        
        # Get the current learning rate
        lr_buffer.append(estimator.optimizer.param_groups[0]['lr'])
        
        # Load batch into autoencoder training buffer
        for j in range(samples_per_batch):
            loss_buffer.append(estimator.learn(states[j], target[j]))
        
        # Remove junk data from training buffer
        try:
            while True:
                loss_buffer.remove(-1)
        except ValueError:
            pass
        
        print(get_print_string(access_order[i], i, num_batches, loss_buffer[-1]), end='\r')
        
    # Draw training data
    save_path = estimator.save(loss_buffer, lr_buffer)
    estimator.draw_training_curve(loss_buffer, lr_buffer, save_path)
    print("Done!")