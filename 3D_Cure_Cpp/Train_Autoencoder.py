# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:18:04 2021

@author: GKSch
"""

import time
from Autoencoder import Autoencoder
import numpy as np
import random

def get_print_string(batch_num, batch_count, num_batches, loss_buffer):
    
    if batch_count == -1:
        batch_count_str = "Batch " + str(0) + "/" + str(num_batches)
        batch_str = " | Set ----"
        loss_str = ""
        for i in range(len(loss_buffer)):
            loss_str = loss_str + " | Loss " + str(i+1) + " = " + '{:.3f}'.format(0.000)
    else:
        batch_count_str = "Batch " + str(batch_count+1) + "/" + str(num_batches)
        batch_str = " | Set " + str(batch_num)
        if batch_num < 10:
            batch_str = batch_str + "   "
        elif batch_num < 100:
            batch_str = batch_str + "  "
        elif batch_num < 1000:
            batch_str = batch_str + " "
        loss_str = ""
        for i in range(len(loss_buffer)):
            loss_str = loss_str + " | Loss " + str(i+1) + " = " + '{:.3f}'.format(round(loss_buffer[i][-1],3))
    
    return batch_count_str+batch_str+loss_str+" |"

if __name__ == "__main__":
    
    # Training data parameters
    num_traj = 5000
    samples_per_traj = 20
    samples_per_batch = 100
    x_dim = 256
    y_dim = 32
    initial_training_criterion = 0.60
    path = 'training_data/DCPD_GC2'
    
    # Hyperparameters May train up to three autoencoders at once
    kernal_size  = [5]
    objct_func   = [1]
    len_output   = [1]
    bottleneck   = [64]
    weighted_arr = [0]
    load_path = [""]
    alpha_zero = 1.0e-3;
    alpha_last = 1.0e-4;
    
    # Calculated parameters
    decay_rate = (alpha_last/alpha_zero) ** (1.0/(num_traj*samples_per_traj))
    num_batches = int(num_traj//(samples_per_batch//samples_per_traj))
    if not (len(bottleneck) == len(len_output) and len(bottleneck) == len(objct_func) and len(bottleneck) == len(weighted_arr) and len(bottleneck) == len(kernal_size) and len(bottleneck) == len(load_path)):
        raise RuntimeError('Unable to parse number of autoencoders to train.')
    if num_batches <= 10:
        raise RuntimeError('Training requires at least 11 batches.')
    
    # Ensure training is occuring
    loss_at_0 = np.zeros(len(bottleneck))
    loss_at_10 = 100.0 * np.ones(len(bottleneck))
    initial_loss_buffer_len = []
    while True:
        
        # Reset loss buffers and autoencoders
        autoencoders = []
        loss_buffer = []
        temp_loss_buffer = []
        for i in range(len(bottleneck)):
            autoencoders.append(Autoencoder(alpha_zero, decay_rate, x_dim, y_dim, bottleneck[i], samples_per_batch, len_output[i], objct_func[i], kernal_size[i], weighted_arr[i]))
            if load_path[i] != "":
                loss_buffer.append(list(autoencoders[i].load(load_path[i])))
                initial_loss_buffer_len.append(len(loss_buffer[i]))
            else:
                loss_buffer.append([])
                initial_loss_buffer_len.append(0)
            
        # Generate random batch order
        access_order = list(range(num_batches))
        random.shuffle(access_order)
        
        # UI and start time
        print('\n')
        print(get_print_string(access_order[0], -1, num_batches, loss_buffer), end='\r')
        start_time = time.time()
        
        # Run first 10 batches
        for i in range(10):
                
            # Load and format current epoch's training data 
            curr_temp_file = path+'/temp_data_' + str(access_order[i]) + '.csv'
            curr_cure_file = path+'/cure_data_' + str(access_order[i]) + '.csv'
            temp = np.genfromtxt(curr_temp_file, delimiter=',')
            cure = np.genfromtxt(curr_cure_file, delimiter=',')
            temp = temp.reshape(samples_per_batch, x_dim, y_dim)
            cure = cure.reshape(samples_per_batch, x_dim, y_dim)
    
            # Load epoch training data into autoencoder training buffer
            for j in range(samples_per_batch):
                for k in range(len(autoencoders)):
                    temp_loss_buffer.append(autoencoders[k].learn(temp[j,:,:], cure[j,:,:]))
            
                # After optimization occured, record training loss
                if temp_loss_buffer[-1] != -1:
                    for l in range(len(autoencoders)):
                        loss_buffer[l].append(temp_loss_buffer[l-len(autoencoders)])
                        if len(loss_buffer[l])==initial_loss_buffer_len[l]+1:
                            loss_at_0[l] = temp_loss_buffer[l-len(autoencoders)]
                        if len(loss_buffer[l])==initial_loss_buffer_len[l]+10:
                            loss_at_10[l] = temp_loss_buffer[l-len(autoencoders)]
                    print(get_print_string(access_order[i], i, num_batches, loss_buffer), end='\r')
                
                # Reset loss buffer
                temp_loss_buffer = []
                
        # Exit first 10 loop if training condition met
        if (loss_at_10 <= initial_training_criterion * loss_at_0).all():
            break
        
        # Reset if training condition is not met
        if len(loss_buffer[0])==initial_loss_buffer_len[0]+10:
            print('\nInitial training condition not met. Resetting...\n')
            
    # Run the rest of the batches
    for i in range(10, num_batches):
            
        # Load and format current epoch's training data 
        curr_temp_file = path+'/temp_data_' + str(access_order[i]) + '.csv'
        curr_cure_file = path+'/cure_data_' + str(access_order[i]) + '.csv'
        temp = np.genfromtxt(curr_temp_file, delimiter=',')
        cure = np.genfromtxt(curr_cure_file, delimiter=',')
        temp = temp.reshape(samples_per_batch, x_dim, y_dim)
        cure = cure.reshape(samples_per_batch, x_dim, y_dim)
    
        # Load epoch training data into autoencoder training buffer
        for j in range(samples_per_batch):
            for k in range(len(autoencoders)):
                temp_loss_buffer.append(autoencoders[k].learn(temp[j,:,:], cure[j,:,:]))
        
            # After optimization occured, record training loss
            if temp_loss_buffer[-1] != -1:
                for l in range(len(autoencoders)):
                    loss_buffer[l].append(temp_loss_buffer[l-len(autoencoders)])
                print(get_print_string(access_order[i], i, num_batches, loss_buffer), end='\r')
            
            # Reset loss buffer
            temp_loss_buffer = []
        
    # Tell user how long training took
    elapsed = time.time()-start_time
    print("\nTraining took: " + str(round(elapsed,1)) + ' seconds')
        
    # Draw training data
    for i in range(len(autoencoders)):
        save_path = autoencoders[i].save(loss_buffer[i])
        autoencoders[i].draw_training_curve(loss_buffer[i], save_path)
         
        # Store rendering data to autoencoder save buffers and render
        print("Loading rendering data...")
        render_temp = np.genfromtxt(path+'/rendering_data/temp_data_0.csv', delimiter=',')
        render_cure = np.genfromtxt(path+'/rendering_data/cure_data_0.csv', delimiter=',')
        render_temp = render_temp.reshape(len(render_temp)//x_dim, x_dim, y_dim)
        render_cure = render_cure.reshape(len(render_cure)//x_dim, x_dim, y_dim)
        
        # Save feature maps
        autoencoders[i].plot_feature_maps(render_temp[int(round(2.0*len(render_temp)/3.0))], save_path)
        
        # Render
        for j in range(len(render_temp)):
            autoencoders[i].save_frame(render_temp[j], render_cure[j])
        autoencoders[i].render(save_path)