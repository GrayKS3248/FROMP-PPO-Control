# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
import torch.nn as nn
import Autoencoder_NN
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class Autoencoder:
    
    def __init__(self, alpha, decay, x_dim, y_dim, out_size, frame_buffer_size, load_previous):
        
        # Initialize model
        self.model = Autoencoder_NN.NN(x_dim, y_dim, out_size);
        if load_previous:
            self.load_previous()
            
        # Initialize loss criterion, and optimizer
        self.criterion_1 = nn.MSELoss()
        self.criterion_2 = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decay)
    
        # Load model onto GPU
        self.device = self.get_device()
        self.model.to(self.device)
        print("Device(")
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
        
        # Test frames
        self.test_frame_buffer = []
        
    def load_previous(self):
        # Find load paths
        done = False
        curr_folder = 1
        while not done:
            path = "results/Auto_"+str(curr_folder)
            if not os.path.isdir(path):
                done = True
            else:
                curr_folder = curr_folder + 1
        path = "results/Auto_"+str(curr_folder-1)+"/output"
        
        if not os.path.isdir("results/Auto_"+str(curr_folder-1)):
            raise RuntimeError("Could not find previous autoencoder to load")
        
        # Load the previous autoencoder
        with open(path, 'rb') as file:
            load_data = pickle.load(file)
            
        # Copy previous NN to current module
        print("Loading previous autoencoder...\n")
        load_autoencoder = load_data['autoencoder']
        self.model.load_state_dict(load_autoencoder.state_dict())
        
    def get_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device
 
    def forward(self, frame):
        
        # Convert frame to proper data type
        with torch.no_grad():
            frame = torch.tensor(frame)
            frame = frame.reshape(1,1,frame.shape[0],frame.shape[1]).float()
            frame = frame.to(self.device)
            rebuilt_frame = self.model.forward(frame).to('cpu')
            
        # convert encoded frame to proper data type
        rebuilt_frame = rebuilt_frame.squeeze().numpy()
        
        # Return the encoded frame of the proper data type
        return rebuilt_frame
       
    def encode(self, frame):
        # Convert frame to proper data type
        with torch.no_grad():
            frame = torch.tensor(frame)
            frame = frame.reshape(1,1,frame.shape[0],frame.shape[1]).float()
            frame = frame.to(self.device)
            encoded_frame = self.model.encode(frame).to('cpu')
            
        # convert encoded frame to proper data type
        encoded_frame = encoded_frame.squeeze().numpy().tolist()
        
        # Return the encoded frame of the proper data type
        return encoded_frame
    
    def update(self, frame):
        
        # Store the current frame
        self.frame_buffer.append(np.array(frame))
        if len(self.test_frame_buffer) < self.frame_buffer_size:
            self.test_frame_buffer.append(np.array(frame))

        # If the frame buffer is full, perform one epoch of stochastic gradient descent
        if len(self.frame_buffer) >= self.frame_buffer_size:
            
            # Step through frame buffer
            self.tot_MSE_loss = 0.0
            rand_indcies = np.random.permutation(self.frame_buffer_size)
            
            for i in range(self.frame_buffer_size):
                
                # Convert frame to proper data type
                with torch.no_grad():
                    # Format frame data
                    frame = torch.tensor(self.frame_buffer[rand_indcies[i]])
                    frame = frame.reshape(1,1,frame.shape[0],frame.shape[1]).float()
                    
                    # Calculate front location
                    frame_front_loc = (torch.roll(frame, 1, 2) - frame)
                    front_exists = frame_front_loc[0,0,0,:]<0.0
                    front_param = torch.quantile(frame_front_loc[:,:,1:,:], 0.995)
                    frame_front_loc = (frame_front_loc-front_param).clamp(0.0, 1.0)
                    frame_front_loc[0,0,0,:]=0.0
                    frame_front_loc[frame_front_loc>0.0]=1.0
                    
                    # Calculate distance from each mesh vertex to nearest front index in column
                    row_indices = torch.linspace(0,len(frame[0,0,:,0])-1,len(frame[0,0,:,0]))
                    frame_front_loc_indices = frame_front_loc.argmax(2)[0,0,:]
                    frame_front_loc_indices[frame_front_loc_indices==0]=(len(frame[0,0,:,0])-1)
                    front_dist = torch.zeros((1,1,len(frame[0,0,:,0]),len(frame[0,0,0,:])))
                    for j in range(len(frame[0,0,0,:])):
                        if front_exists[j].item():
                            temp = abs(1.0 - row_indices / frame_front_loc_indices[j])
                        else:
                            temp = abs(1.0 - row_indices / (len(frame[0,0,:,0])-1))
                        front_dist[0,0,:,j] = temp / temp.max()
                    front_dist = (1.0 - front_dist)**10
                    front_dist[front_dist<0.01] = 0.0
                    
                    # Combine and format temperature and front location data
                    target = torch.cat((frame, front_dist), 1)
                    target = target.to(self.device)
                
                # Forward propogate the frame through the autoencoder
                frame = frame.to(self.device)
                rebuilt_frame = self.model.forward(frame)
            
                # Calculate loss and take optimization step and learning rate step
                self.optimizer.zero_grad()
                curr_MSE_loss = self.criterion_1(rebuilt_frame, target)
                curr_MSE_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                
                # Sum the epoch's total loss
                self.tot_MSE_loss = self.tot_MSE_loss + curr_MSE_loss.item()
            
            # Empty frame buffer
            self.frame_buffer = []
        
        
        return self.tot_MSE_loss
    
    def display_and_save(self, MSE_loss):
        print("Saving autoencoder results...")

        data = {
            'MSE_loss' : np.array(MSE_loss),
            'Test_frames' : self.test_frame_buffer,
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
        print("Plotting autoencoder results...")
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
        
    def render(self, frames):
        print("Rendering...")
        x_grid, y_grid = np.meshgrid(np.linspace(0,1,self.x_dim), np.linspace(0,1,self.y_dim))
        
        # Find save paths
        done = False
        curr_folder = 1
        while not done:
            path = "results/Auto_"+str(curr_folder)
            if not os.path.isdir(path):
                done = True
            else:
                curr_folder = curr_folder + 1
        path = "results/Auto_"+str(curr_folder-1)+"/video"
        if not os.path.isdir(path):
            os.mkdir(path)
        
        for curr_step in range(len(frames)):

            # Make fig for temperature, cure, and input
            plt.cla()
            plt.clf()
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
            fig.set_size_inches(8.5,11)
            
            # Plot frame
            c0 = ax0.pcolormesh(x_grid, y_grid, np.transpose(frames[curr_step]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
            cbar0 = fig.colorbar(c0, ax=ax0)
            cbar0.set_label('Nondimensional Temperature [-]',labelpad=20,fontsize='large')
            cbar0.ax.tick_params(labelsize=12)
            ax0.set_xlabel('X Position [-]',fontsize='large')
            ax0.set_ylabel('Y Position [-]',fontsize='large')
            ax0.tick_params(axis='x',labelsize=12)
            ax0.tick_params(axis='y',labelsize=12)
            ax0.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax0.set_title('True Frame')
            
            # Rebuilt temperature
            c1 = ax1.pcolormesh(x_grid, y_grid, np.transpose(self.forward(frames[curr_step])[0,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
            cbar1 = fig.colorbar(c1, ax=ax1)
            cbar1.set_label('Nondimensional Temperature [-]',labelpad=20,fontsize='large')
            cbar1.ax.tick_params(labelsize=12)
            ax1.set_xlabel('X Position [-]',fontsize='large')
            ax1.set_ylabel('Y Position [-]',fontsize='large')
            ax1.tick_params(axis='x',labelsize=12)
            ax1.tick_params(axis='y',labelsize=12)
            ax1.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax1.set_title('Rebuilt Temperature Field')
            
            # Rebuilt front location
            c2 = ax2.pcolormesh(x_grid, y_grid, np.transpose(self.forward(frames[curr_step])[1,:,:]), shading='gouraud', cmap='YlOrRd', vmin=0.0, vmax=1.0)
            cbar2 = fig.colorbar(c2, ax=ax2)
            cbar2.set_label('Front Probability [-]',labelpad=20,fontsize='large')
            cbar2.ax.tick_params(labelsize=12)
            ax2.set_xlabel('X Position [-]',fontsize='large')
            ax2.set_ylabel('Y Position [-]',fontsize='large')
            ax2.tick_params(axis='x',labelsize=12)
            ax2.tick_params(axis='y',labelsize=12)
            ax2.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax2.set_title('Rebuilt Front Location')
            
            # Set title and save
            plt.savefig(path+"/"+str(curr_step).zfill(4)+'.png', dpi=100)
            plt.close()
                
        
if __name__ == '__main__':
    autoencoder = Autoencoder(1.0e-3, 1.0, 360, 40, 64, 20, True)
    
    with open("results/Auto_1/test_frames", 'rb') as file:
        test_frames = pickle.load(file)
    test_frames = test_frames['data']['Test_frames']
    
    # for i in range (len(test_frames)):
    #     autoencoder.update(test_frames[1784+i])
    
    autoencoder.render(test_frames)