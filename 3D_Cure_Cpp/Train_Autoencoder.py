# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:18:04 2021

@author: GKSch
"""

import time
from Autoencoder import Autoencoder
import numpy as np

def get_print_string(curr_epoch, num_epochs, loss_buffer):
    
    if curr_epoch == -1:
        epoch_str = "Epoch " + str(0) + "/" + str(num_epochs)
        loss_str = ""
        for i in range(len(loss_buffer)):
            loss_str = loss_str + " | Loss " + str(i+1) + " = " + '{:.3f}'.format(0.000)
    else:
        epoch_str = "Epoch " + str(curr_epoch+1) + "/" + str(num_epochs)
        loss_str = ""
        for i in range(len(loss_buffer)):
            loss_str = loss_str + " | Loss " + str(i+1) + " = " + '{:.3f}'.format(round(loss_buffer[i][-1],3))
    
    return epoch_str+loss_str+" |"

if __name__ == "__main__":
    
    # Training data parameters
    num_traj = 5000
    samples_per_traj = 20
    samples_per_batch = 100
    x_dim = 360
    y_dim = 40
    path = 'training_data/DCPD_GC2'
    
    # Hyperparameters
    n_filter_1 = [8]
    n_filter_2 = [16]
    bottleneck = [32]
    len_output = [2]
    objct_func = [2]
    alpha_zero = 1.0e-3;
    alpha_last = 1.0e-5;
    
    # Calculated parameters
    decay_rate = (alpha_last/alpha_zero) ** (1.0/(num_traj*samples_per_traj))
    num_epochs = int(num_traj//(samples_per_batch//samples_per_traj))
    if not (len(n_filter_1) == len(n_filter_2) and len(n_filter_1) == len(bottleneck) and len(n_filter_1) == len(len_output) and len(n_filter_1) == len(objct_func)):
        raise RuntimeError('Unable to parse number of autoencoders to train.')
    
    # Create autoencoders
    autoencoders = []
    loss_buffer = []
    temp_buffer = []
    for i in range(len(n_filter_1)):
        autoencoders.append(Autoencoder(alpha_zero, decay_rate, x_dim, y_dim, n_filter_1[i], n_filter_2[i], bottleneck[i], samples_per_batch, len_output[i], objct_func[i]))
        loss_buffer.append([])
    
    # Train over set of epochs
    print('\n')
    print(get_print_string(-1, num_epochs, loss_buffer), end='\r')
    start_time = time.time()
    for i in range(num_epochs):
            
        # Load and format current epoch's training data 
        curr_temp_file = path+'/temp_data_' + str(i) + '.csv'
        curr_cure_file = path+'/cure_data_' + str(i) + '.csv'
        temp = np.genfromtxt(curr_temp_file, delimiter=',')
        cure = np.genfromtxt(curr_cure_file, delimiter=',')
        temp = temp.reshape(samples_per_batch, x_dim, y_dim)
        cure = cure.reshape(samples_per_batch, x_dim, y_dim)

        # Load epoch training data into autoencoder trainging buffer
        for j in range(samples_per_batch):
            for k in range(len(autoencoders)):
                temp_buffer.append(autoencoders[k].learn(temp[j,:,:], cure[j,:,:]))
        
            # After optimization occured, record training loss
            if temp_buffer[-1] != -1:
                for l in range(len(autoencoders)):
                    loss_buffer[l].append(temp_buffer[l-len(autoencoders)])
                print(get_print_string(i, num_epochs, loss_buffer), end='\r')
            temp_buffer = []
    
    # Draw training data
    for i in range(len(autoencoders)):
        print("\n")
        save_path = autoencoders[i].save(loss_buffer[i])
        autoencoders[i].draw_training_curve(loss_buffer[i], save_path)
         
        # Store rendering data to autoencoder save buffers and render
        print("Loading rendering data...")
        render_temp = np.genfromtxt(path+'/rendering_data/temp_data.csv', delimiter=',')
        render_cure = np.genfromtxt(path+'/rendering_data/cure_data.csv', delimiter=',')
        render_temp = render_temp.reshape(len(render_temp)//x_dim, x_dim, y_dim)
        render_cure = render_cure.reshape(len(render_cure)//x_dim, x_dim, y_dim)
        for j in range(len(render_temp)):
            autoencoders[i].save_frame(render_temp[j], render_cure[j])
        autoencoders[i].render(save_path)
        
    # Tell user how long training took
    elapsed = time.time()-start_time
    print("\nTraining took: " + str(round(elapsed,1)) + ' seconds')