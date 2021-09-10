# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
from CNN_MLP_Estimator import Model as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class State_Estimator:

    def __init__(self, x_dim, y_dim, bottleneck, kernal_size, num_states, time_between_state_frames, alpha, decay, num_epochs, samples_per_batch):
        
        # Hyperparameters
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.bottleneck = bottleneck
        self.kernal_size = kernal_size
        self.num_states = num_states
        self.time_between_state_frames = time_between_state_frames
        self.alpha = alpha
        self.decay = decay
        
        # Initialize model
        self.model = nn(self.x_dim, self.y_dim, self.bottleneck, self.kernal_size, self.num_states)
        
        # Training parameters
        self.samples_per_batch = samples_per_batch
        self.states_batch = []
        self.target_batch = []
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.decay)
        
        # Load model onto GPU
        self.device = self.get_device()
        self.model.to(self.device)
        
        # Initialize batches
        self.num_epochs = num_epochs
        self.samples_per_batch = samples_per_batch
        self.temp_batch = []
        self.cure_batch = []
        
        # Save frames
        self.temp_save_buffer = []
        self.cure_save_buffer = []
        
    # Gets the cpu or gpu on which to run NN
    # @return device code
    def get_device(self):
        
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device
 
    # Forward propagates the input states
    # @param the input states
    # @return the estimated singular state
    def forward(self, states):
        
        with torch.no_grad():
            # Format input states
            states = torch.tensor(states)
            states = states.reshape(states.shape[0],1,states.shape[1],states.shape[2]).float().to(self.device)
        
            # Forward propogate formatted state
            estimated_state = self.model.forward(states).to('cpu')

        # Return the estimated state
        return estimated_state
    
    # Updates the state estimator
    # @param the input states
    # @param the target singular state given the input states
    # @return average epoch training loss or -1 if no optimization epoch occured
    def learn(self, states, target):
        
        # Store the current temperature and cure frames in the batch (add noise if noisy)
        self.states_batch.append(np.array(states))
        self.target_batch.append(target)
         
        # If the batch is full, tochastic gradient descent
        if len(self.states_batch) >= self.samples_per_batch:
            
            # Perform num_epochs epochs of optimization
            total_MSE = 0.0
            for epoch in range(self.num_epochs):
            
                # Step through batch
                rand_indcies = np.random.permutation(self.samples_per_batch)
                for i in range(self.samples_per_batch):
                    
                    # Get current states and target
                    curr_states = self.states_batch[rand_indcies[i]]
                    curr_target = self.target_batch[rand_indcies[i]]
                            
                    # Format and forward propagate temperature data
                    curr_states = torch.tensor(curr_states)
                    curr_states = curr_states.reshape(1,curr_states.shape[0],curr_states.shape[1],curr_states.shape[2]).float().to(self.device)
                    estimated_state = self.model.forward(curr_states).squeeze()
            
                    # Get the MSE
                    curr_MSE = ((estimated_state - curr_target)**2)
            
                    # Take optimization step
                    self.optimizer.zero_grad()
                    curr_MSE.backward()
                    self.optimizer.step()
                    
                    # Sum the batch's total loss
                    total_MSE = total_MSE + curr_MSE.item()
                    
                # Update LR after each epoch
                self.lr_scheduler.step()
            
            # Empty the batches
            self.states_batch = []
            self.target_batch = []

            # Return the average RMS reconstruction error
            return np.sqrt(total_MSE / (self.num_epochs * self.samples_per_batch))
        
        # return -1 if no optimization epoch occured
        return -1
    
    # Saves the training data and trained model
    # @param training loss curve
    # @param learning rate curve
    # @return path at which data has been saved
    def save(self, training_curve, lr_curve):
        print("\nSaving state estimator results...")

        # Store data to dictionary
        data = {
            'num_states' : self.num_states,
            'time_between_state_frames' : self.time_between_state_frames,
            'alpha' : self.alpha,
            'decay' : self.decay,
            'samples_per_batch' : self.samples_per_batch,
            'training_curve' : np.array(training_curve),
            'lr_curve' : np.array(lr_curve),
            'model' : self.model.to('cpu'),
        }
        self.model.to(self.device)

        # Find save paths
        initial_path = "results/" + str(self.num_states) + '_{0:.3f}'.format(self.time_between_state_frames)
        path = initial_path
        done = False
        curr_dir_num = 1
        while not done:
            if not os.path.isdir(path):
                os.mkdir(path)
                done = True
            else:
                curr_dir_num = curr_dir_num + 1
                path = initial_path + "(" + str(curr_dir_num) + ")"

        # Pickle all important outputs
        save_file = path + "/output"
        with open(save_file, 'wb') as file:
            pickle.dump(data, file)

        return path

    # Draw and save the training curve
    # @param training curve to be drawn
    # @param path at which training curve is saved
    def draw_training_curve(self, training_curve, lr_curve, path):
        print("Plotting estimator training curve...")
        
        plt.clf()
        plt.title("State Estimator Learning Curve",fontsize='xx-large')
        plt.xlabel("Batch",fontsize='large')
        plt.ylabel("Mean Square Error",fontsize='large')
        plt.plot([*range(len(training_curve))],training_curve,lw=2.5,c='r')
        plt.yscale("log")
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/estimator_training.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()
        
        plt.clf()
        plt.title("State Estimator Learning Rate Curve",fontsize='xx-large')
        plt.xlabel("Batch",fontsize='large')
        plt.ylabel("Learning Rate",fontsize='large')
        plt.plot([*range(len(lr_curve))],lr_curve,lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/estimator_lr.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()