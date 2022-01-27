# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:18:04 2021

@author: GKSch
"""

from Autoencoder import Autoencoder
import numpy as np

def get_print_string(curr_epochs, num_epochs, batch_num, batch_count, num_batches, curr_loss, curr_lr, curr_sparsity):
    
    if batch_count == -1:
        epoch_str = "Epoch " + str(0) + "/" + str(num_epochs)
        batch_count_str = " | File " + str(0) + "/" + str(num_batches)
        batch_str = " | Batch ---"
        loss_str = " | Loss = " + '{:.3f}'.format(0.000)
        lr_str = " | Lr = " + '{:.5f}'.format(0.00000)
        sparsity_str = " | Sparsity = " + '{:.2f}'.format(0.00)
    else:
        epoch_str = "Epoch " + str(curr_epochs+1) + "/" + str(num_epochs)
        batch_count_str = " | File " + str(batch_count+1) + "/" + str(num_batches)
        batch_str = " | Batch " + str(batch_num)
        if batch_num < 10:
            batch_str = batch_str + "  "
        elif batch_num < 100:
            batch_str = batch_str + " "
        loss_str = " | Loss = " + '{:.3f}'.format(round(curr_loss,3))
        lr_str = " | Lr = " + '{:.5f}'.format(round(curr_lr,5))
        sparsity_str = " | Sparsity = " + '{:.2f}'.format(round(curr_sparsity,2))
        
    return epoch_str+batch_count_str+batch_str+loss_str+lr_str+sparsity_str+" |                            "

if __name__ == "__main__":
    
    # Training data parameters
    num_epochs = 1
    num_files = 2500
    samples_per_file = 40
    samples_per_batch = 1
    total_snapshots = 100
    training_data_path = '../training_data/DCPD_GC2/Autoencoder_Shuffled'
    
    # Image parameters
    x_dim = 256
    y_dim = 32
    norm_min = 0.0
    norm_max = 1.0
    
    # Hyperparameters
    load_path = ""
    targets = ['temp']
    latent_targets = ['ftemp', 'fspeed']
    sparsity_parameter = 0.20
    sparsity_const = 0.0
    image_const = 1.0
    latent_const = 1.0
    noise_prob = 0.25
    noise_stdev = 0.025
    occ_prob = 0.25
    occ_max = 3
    occ_min_area = 0.01
    occ_max_area = 0.25
    alpha_zero = 1.0e-4
    alpha_last = 1.0e-4
    
    # Calculated parameters
    batch_per_file = int(samples_per_file / samples_per_batch)
    num_batches = (samples_per_file / samples_per_batch) * num_files
    alpha_decay = (alpha_last/alpha_zero) ** (1.0/(num_batches*num_epochs))
    
    # Initialize the estimator
    autoencoder = Autoencoder(alpha_zero, alpha_decay, load_path=load_path, dim_1=x_dim, dim_2=y_dim, norm_min=norm_min, norm_max=norm_max, num_targets=len(targets), num_latent=len(latent_targets), sparsity_parameter=sparsity_parameter, sparsity_const=sparsity_const, image_const=image_const, latent_const=latent_const, noise_prob=noise_prob, noise_stdev=noise_stdev, occ_prob=occ_prob, occ_max=occ_max, occ_min_area=occ_min_area, occ_max_area=occ_max_area, verbose=True)
    
    # Run the epochs
    optimization_count = 0
    snapshot_optimizations = np.arange(0,num_batches*num_epochs,np.ceil(num_batches * num_epochs / total_snapshots))
    print(get_print_string(0, num_epochs, 0, -1, num_files, 0.0, 0.0, 0.0), end='\r')
    for epoch in range(num_epochs):
        for file in range(num_files):
                
            # Load and format current file's temperature field data 
            temp_file = training_data_path+'/temp_data_' + str(file) + '.csv'
            temp_file = np.genfromtxt(temp_file, delimiter=',')
            temp_file = temp_file.reshape(samples_per_file, x_dim, y_dim)
            
            # Load and format current file's cure field data 
            cure_file = training_data_path+'/cure_data_' + str(file) + '.csv'
            cure_file = np.genfromtxt(cure_file, delimiter=',')
            cure_file = cure_file.reshape(samples_per_file, x_dim, y_dim)
            
            # Load and format current file's cure field data 
            floc_file = training_data_path+'/floc_data_' + str(file) + '.csv'
            floc_file = np.genfromtxt(floc_file, delimiter=',')
            
            # Load and format current file's cure field data 
            ftemp_file = training_data_path+'/ftemp_data_' + str(file) + '.csv'
            ftemp_file = np.genfromtxt(ftemp_file, delimiter=',')
            
            # Load and format current file's cure field data 
            fshape_file = training_data_path+'/fshape_data_' + str(file) + '.csv'
            fshape_file = np.genfromtxt(fshape_file, delimiter=',')
            
            # Load and format current file's cure field data 
            fspeed_file = training_data_path+'/fspeed_data_' + str(file) + '.csv'
            fspeed_file = np.genfromtxt(fspeed_file, delimiter=',')
            
            # Designate target
            target_file = []
            if "temp" in targets:
                target_file.append(temp_file)
            if "cure" in targets:
                target_file.append(cure_file)
            if "floc" in latent_targets:
                target_file.append(floc_file)
            if "ftemp" in latent_targets:
                target_file.append(ftemp_file)
            if "fshape" in latent_targets:
                target_file.append(fshape_file)
            if "fspeed" in latent_targets:
                target_file.append(fspeed_file)
            
            # Split file into batches
            for batch in range(batch_per_file):
                start_index = batch*samples_per_batch
                end_index = (batch+1)*samples_per_batch-1
                temp_batch = temp_file[start_index:end_index+1]
                target_batch = []
                for target in range(len(target_file)):
                    target_batch.append(target_file[target][start_index:end_index+1])
            
                # Update the weight and bias of the estimator
                loss, lr, sparsity = autoencoder.learn(temp_batch, target_batch, take_snapshot=(optimization_count in snapshot_optimizations))
                optimization_count = optimization_count + 1
                
                print(get_print_string(epoch, num_epochs, batch, file, num_files, loss, lr, sparsity), end='\r')
        
    # Draw training data
    autoencoder.save()
    print("Done!")