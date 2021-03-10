# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
import torch.nn as nn
import Autoencoder_NN as auto
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class Autoencoder:
    
    def __init__(self, alpha, decay, x_dim, y_dim, out_size, frame_buffer_size, load_previous):
        
        # Initialize model, loss criterion, and optimizer
        self.model = auto.NN(x_dim, y_dim, out_size);
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decay)
    
        # Load model onto GPU
        self.device = self.get_device()
        self.model.to(self.device)
        print("device(")
        print("  " + self.device)
        print(")\n")
        print(self.model) 
    
        # Store NN shape parameters
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.out_size = out_size
        
        # Memory for MSE loss
        self.tot_MSE_loss = 0.0
        
        # Initialize frame buffer
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = []
        
    def get_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device
        
    def update(self, frame):
        
        # Store the current frame
        self.frame_buffer.append(np.array(frame))

        # If the frame buffer is full, perform one epoch of stochastic gradient descent
        if len(self.frame_buffer) >= self.frame_buffer_size:
            
            # Step through frame buffer
            self.tot_MSE_loss = 0.0
            rand_indcies = np.random.permutation(self.frame_buffer_size)
            
            for i in range(self.frame_buffer_size):
                
                # Convert frame to proper data type
                with torch.no_grad():
                    frame = torch.tensor(self.frame_buffer[rand_indcies[i]])
                    frame = frame.reshape(1,1,frame.shape[0],frame.shape[1]).float()
                    frame = frame.to(self.device)
                
                # Forward propogate the frame through the autoencoder
                rebuilt_frame = self.model.forward(frame)
            
                # Calculate loss and take optimization step and learning rate step
                self.optimizer.zero_grad()
                curr_MSE_loss = self.criterion(rebuilt_frame, frame)
                curr_MSE_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                
                # Sum the epoch's total loss
                self.tot_MSE_loss = self.tot_MSE_loss + curr_MSE_loss.item()
                
            
            # Empty frame buffer
            self.frame_buffer = []
        
        
        return self.tot_MSE_loss
    
    def display_and_save(self, MSE_loss):
        print("Saving results...")

        data = {
            'MSE_loss' : np.array(MSE_loss),
        }

        # Find save paths
        done = False
        curr_folder = 1
        while not done:
            path = "results/Auto_"+str(curr_folder)
            if not os.path.isdir(path):
                os.mkdir(path)
                done = True
            else:
                curr_folder = curr_folder + 1

        # Pickle all important outputs
        output = { 'data':data, 'autoencoder':self.model}
        save_file = path + "/output"
        with open(save_file, 'wb') as file:
            pickle.dump(output, file)

        # Plot the BCE loss training data
        print("Plotting...")
        # Plot value learning curve
        plt.clf()
        title_str = "Autoencoder Learning Curve"
        plt.title(title_str,fontsize='xx-large')
        plt.xlabel("Optimization Frame",fontsize='large')
        plt.ylabel("MSE Loss",fontsize='large')
        plt.plot([*range(len(data['MSE_loss']))],data['MSE_loss'],lw=2.5,c='r')
        plt.yscale("log")
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/autoencoder_learning.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()