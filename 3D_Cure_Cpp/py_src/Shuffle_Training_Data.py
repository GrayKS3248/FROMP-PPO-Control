# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:09:17 2021

@author: GKSch
"""

import numpy as np
import csv
import os
import shutil

if __name__ == "__main__":
    
    # Data parameters
    num_batches = 2500
    samples_per_batch = 40
    training_data_path = '../training_data/DCPD_GC2/Quenched_AE_controlled_inputs'
    x_dim = 256
    y_dim = 32
    
    # Path manip
    if not os.path.isdir(training_data_path+"_shuffled"):
        os.mkdir(training_data_path+"_shuffled")
    else:
        shutil.rmtree(training_data_path+"_shuffled")
        os.mkdir(training_data_path+"_shuffled")
    
    # Generate random batch and sample order to mix batches
    access_order = np.zeros((num_batches, samples_per_batch))
    for batch in range(num_batches):
        access_order[batch] = np.random.permutation(samples_per_batch)
    mixed_batch_indices = np.zeros((num_batches, samples_per_batch))
    access_index = 0
    for batch in range(num_batches):
        for sample in range(samples_per_batch):
            access_batch = access_index % num_batches
            access_sample = int(access_index // num_batches)
            mixed_batch_indices[access_batch][access_sample] = batch
            access_index = access_index + 1
    
    # Load and compile all the latent batch variables
    floc_batches = np.zeros((num_batches, samples_per_batch))
    ftemp_batches = np.zeros((num_batches, samples_per_batch))
    fshape_batches = np.zeros((num_batches, samples_per_batch))
    fspeed_batches = np.zeros((num_batches, samples_per_batch))
    for batch in range(num_batches):
            
        # Load and format current batches's cure field data 
        floc_file = training_data_path+'/floc_data_' + str(batch) + '.csv'
        floc_batch = np.genfromtxt(floc_file, delimiter=',')
        floc_batches[batch] = floc_batch
        
        # Load and format current batches's cure field data 
        ftemp_file = training_data_path+'/ftemp_data_' + str(batch) + '.csv'
        ftemp_batch = np.genfromtxt(ftemp_file, delimiter=',')
        ftemp_batches[batch] = ftemp_batch
        
        # Load and format current batches's cure field data 
        fshape_file = training_data_path+'/fshape_data_' + str(batch) + '.csv'
        fshape_batch = np.genfromtxt(fshape_file, delimiter=',')
        fshape_batches[batch] = fshape_batch
        
        # Load and format current batches's cure field data 
        fspeed_file = training_data_path+'/fspeed_data_' + str(batch) + '.csv'
        fspeed_batch = np.genfromtxt(fspeed_file, delimiter=',')
        fspeed_batches[batch] = fspeed_batch   
        
    # Mix together batches and run SGD
    access_index = 0
    for batch in range(num_batches):
        
        # Load and format current batches's temperature field data 
        temp_file = training_data_path+'/temp_data_' + str(batch) + '.csv'
        temp_batch = np.genfromtxt(temp_file, delimiter=',')
        temp_batch = temp_batch.reshape(samples_per_batch, x_dim, y_dim)
        
        # Load and format current batches's cure field data 
        cure_file = training_data_path+'/cure_data_' + str(batch) + '.csv'
        cure_batch = np.genfromtxt(cure_file, delimiter=',')
        cure_batch = cure_batch.reshape(samples_per_batch, x_dim, y_dim)
        
        # Initialize mixed latent variable batches
        mixed_floc_batch = np.zeros(samples_per_batch)
        mixed_ftemp_batch = np.zeros(samples_per_batch)
        mixed_fshape_batch = np.zeros(samples_per_batch)
        mixed_fspeed_batch = np.zeros(samples_per_batch)
        
        for sample in range(samples_per_batch):
                
            # Gather current temperature and cure samples
            temp = temp_batch[int(access_order[batch][sample])].tolist()
            cure = cure_batch[int(access_order[batch][sample])].tolist()
        
            # Save a single element of the temperature and cure data
            with open(training_data_path+"_Shuffled"+"/temp_data_"+str(int(mixed_batch_indices[batch][sample]))+".csv", 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                for row in range(x_dim):
                    writer.writerow(temp[row])
            with open(training_data_path+"_Shuffled"+"/cure_data_"+str(int(mixed_batch_indices[batch][sample]))+".csv", 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')
                for row in range(x_dim):
                    writer.writerow(cure[row])
        
            # Gather access data
            access_batch = access_index % num_batches
            access_sample = int(access_order[access_batch][access_index // num_batches])
            access_index = access_index + 1
        
            # Get the latent samples
            floc = floc_batches[access_batch][access_sample]
            ftemp = ftemp_batches[access_batch][access_sample]
            fshape = fshape_batches[access_batch][access_sample]
            fspeed = fspeed_batches[access_batch][access_sample]
            
            # Update mixed batches
            mixed_floc_batch[sample] = floc
            mixed_ftemp_batch[sample] = ftemp
            mixed_fshape_batch[sample] =fshape
            mixed_fspeed_batch[sample] = fspeed
            
        # Save the shuffled latent data
        with open(training_data_path+"_Shuffled"+"/floc_data_"+str(batch)+".csv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for row in range(samples_per_batch):
                writer.writerow([mixed_floc_batch[row]])
        with open(training_data_path+"_Shuffled"+"/ftemp_data_"+str(batch)+".csv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for row in range(samples_per_batch):
                writer.writerow([mixed_ftemp_batch[row]])
        with open(training_data_path+"_Shuffled"+"/fshape_data_"+str(batch)+".csv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for row in range(samples_per_batch):
                writer.writerow([mixed_fshape_batch[row]])
        with open(training_data_path+"_Shuffled"+"/fspeed_data_"+str(batch)+".csv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for row in range(samples_per_batch):
                writer.writerow([mixed_fspeed_batch[row]])