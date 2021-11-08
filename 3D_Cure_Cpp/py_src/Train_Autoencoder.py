# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:18:04 2021

@author: GKSch
"""

from Autoencoder import Autoencoder
import numpy as np
import random

def get_print_string(curr_epochs, num_epochs, batch_num, batch_count, num_batches, curr_loss, curr_lr, curr_sparsity):
    
    if batch_count == -1:
        epoch_str = "Epoch " + str(0) + "/" + str(num_epochs)
        batch_count_str = " | Batch " + str(0) + "/" + str(num_batches)
        batch_str = " | Set ----"
        loss_str = " | Loss = " + '{:.3f}'.format(0.000)
        lr_str = " | Lr = " + '{:.5f}'.format(0.00000)
        sparsity_str = " | Sparsity = " + '{:.2f}'.format(0.00)
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
        lr_str = " | Lr = " + '{:.5f}'.format(round(curr_lr,5))
        sparsity_str = " | Sparsity = " + '{:.2f}'.format(round(curr_sparsity,2))
        
    return epoch_str+batch_count_str+batch_str+loss_str+lr_str+sparsity_str+" |                            "

if __name__ == "__main__":
    
    # Training data parameters
    num_epochs = 5
    num_batches = 1000
    samples_per_batch = 100
    total_snapshots = 100
    training_data_path = '../training_data/DCPD_GC2/Autoencoder_2'
    
    # Image parameters
    x_dim = 256
    y_dim = 32
    norm_min = 0.0
    norm_max = 1.0
    
    # Hyperparameters
    load_path = ""
    targets = ['temp']
    latent_targets = ['ftemp','fshape','fspeed']
    sparsity_parameter = 0.50
    sparsity_const = 1e-3
    image_const = 1.0
    latent_const = 0.1
    noise_prob = 0.50
    noise_stdev = 0.025
    occ_prob = 0.50
    occ_max = 3
    occ_min_area = 0.01
    occ_max_area = 0.25
    alpha_zero = 1.0e-3
    alpha_last = 1.0e-4
    
    # Calculated parameters
    alpha_decay = (alpha_last/alpha_zero) ** (1.0/(num_batches*num_epochs))
    
    # Initialize the estimator
    autoencoder = Autoencoder(alpha_zero, alpha_decay, load_path=load_path, dim_1=x_dim, dim_2=y_dim, norm_min=norm_min, norm_max=norm_max, num_targets=len(targets), num_latent=len(latent_targets), sparsity_parameter=sparsity_parameter, sparsity_const=sparsity_const, image_const=image_const, latent_const=latent_const, noise_prob=noise_prob, noise_stdev=noise_stdev, occ_prob=occ_prob, occ_max=occ_max, occ_min_area=occ_min_area, occ_max_area=occ_max_area, verbose=True)
    
    # Run the epochs
    optimization_count = 0
    snapshot_optimizations = np.arange(0,num_batches*num_epochs,np.ceil(num_batches * num_epochs / total_snapshots))
    print(get_print_string(0.0, num_epochs, 0.0, -1, num_batches, 0.0, 0.0, 0.0), end='\r')
    for epoch in range(num_epochs):
    
        # Generate random batch order
        access_order = list(range(num_batches))
        random.shuffle(access_order)
    
        # Run the batches
        for batch in range(num_batches):
                
            # Load and format current batches's temperature field data 
            print()
            temp_file = training_data_path+'/temp_data_' + str(access_order[batch]) + '.csv'
            temp_batch = np.genfromtxt(temp_file, delimiter=',')
            temp_batch = temp_batch.reshape(samples_per_batch, x_dim, y_dim)
            
            # Load and format current batches's cure field data 
            cure_file = training_data_path+'/cure_data_' + str(access_order[batch]) + '.csv'
            cure_batch = np.genfromtxt(cure_file, delimiter=',')
            cure_batch = cure_batch.reshape(samples_per_batch, x_dim, y_dim)
            
            # Load and format current batches's cure field data 
            floc_file = training_data_path+'/floc_data_' + str(access_order[batch]) + '.csv'
            floc_batch = np.genfromtxt(floc_file, delimiter=',')
            
            # Load and format current batches's cure field data 
            ftemp_file = training_data_path+'/ftemp_data_' + str(access_order[batch]) + '.csv'
            ftemp_batch = np.genfromtxt(ftemp_file, delimiter=',')
            
            # Load and format current batches's cure field data 
            fshape_file = training_data_path+'/fshape_data_' + str(access_order[batch]) + '.csv'
            fshape_batch = np.genfromtxt(fshape_file, delimiter=',')
            
            # Load and format current batches's cure field data 
            fspeed_file = training_data_path+'/fspeed_data_' + str(access_order[batch]) + '.csv'
            fspeed_batch = np.genfromtxt(fspeed_file, delimiter=',')
            
            # Designate target
            target_batch = []
            if "temp" in targets:
                target_batch.append(temp_batch)
            if "cure" in targets:
                target_batch.append(cure_batch)
            if "floc" in latent_targets:
                target_batch.append(floc_batch)
            if "ftemp" in latent_targets:
                target_batch.append(ftemp_batch)
            if "fshape" in latent_targets:
                target_batch.append(fshape_batch)
            if "fspeed" in latent_targets:
                target_batch.append(fspeed_batch)
            
            # Update the weight and bias of the estimator
            loss, lr, sparsity = autoencoder.learn(temp_batch, target_batch, take_snapshot=(optimization_count in snapshot_optimizations))
            optimization_count = optimization_count + 1
            
            print(get_print_string(epoch, num_epochs, access_order[batch], batch, num_batches, loss, lr, sparsity), end='\r')
        
    # Draw training data
    autoencoder.save()
    print("Done!")