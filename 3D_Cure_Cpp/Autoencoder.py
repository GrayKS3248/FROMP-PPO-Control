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
    
    # OBJECTIVE FNC 1: Target temperature field
    # OBJECTIVE FNC 2: Target temperature field, and blurred front location
    # OBJECTIVE FNC 3: Target temperature field, blurred front location, and cure field
    # OBJECTIVE FNC 5: Target quantized temperature field
    def __init__(self, alpha, decay, x_dim_input, y_dim_input, num_filter_1, num_filter_2, bottleneck, buffer_size, num_output_layers, objective_fnc):
        
        # Initialize model
        self.model = Autoencoder_NN.NN(x_dim_input, y_dim_input, num_filter_1, num_filter_2, bottleneck, num_output_layers);
            
        # Initialize loss criterion, and optimizer
        self.criterion = nn.MSELoss()
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
        self.x_dim = x_dim_input
        self.y_dim = y_dim_input
        self.bottleneck = bottleneck
        self.num_output_layers = num_output_layers
        if num_output_layers > 8 or num_output_layers < 0:
            raise RuntimeError('Number of output layers must be greater than 0 and less than 9.')
        
        # Objective fnc type
        self.objective_fnc = objective_fnc
        if objective_fnc > num_output_layers or objective_fnc == 4:
            raise RuntimeError('Objective function must be greater than 0 and less than or equal to the number of output layers. (!= 4)')
        
        # Initialize frame buffer
        self.buffer_size = buffer_size
        self.temp_buffer = []
        self.cure_buffer = []
        
        # Save frames
        self.save_temp_buffer = []
        self.save_cure_buffer = []
        
    # Loads a given saved autoencoder
    # @param the path from which the autoencoder will be loaded
    def load(self, path):
        
        # Copy NN at path to current module
        print("\nLoading: " + path)
        if not os.path.isdir(path):
            raise RuntimeError("Could not find " + path)
        else:
            with open(path+"/output", 'rb') as file:
                loaded_data = pickle.load(file)
            loaded_model = loaded_data['autoencoder']
            self.model.load_state_dict(loaded_model.state_dict())
        
    # Gets the cpu or gpu on which to run NN
    # @return device code
    def get_device(self):
        
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device
 
    # Forward propagates the temperature through the autoencoder
    # @param the temperature field that informs data reconstruction
    # @return the reconstructed data
    def forward(self, temp):
        
        # Convert frame to proper data type
        with torch.no_grad():
            temp = torch.tensor(temp)
            temp = temp.reshape(1,1,temp.shape[0],temp.shape[1]).float()
            temp = temp.to(self.device)
            rebuilt_data = self.model.forward(temp).to('cpu')
            
            # convert encoded frame to proper data type
            rebuilt_data = rebuilt_data.squeeze().numpy().tolist()
        
        # Return the encoded frame of the proper data type
        return rebuilt_data
       
    # Encodes a given temperature field
    # @param the temperature field to encode
    # @return list of the encoded data
    def encode(self, temp):
        # Convert frame to proper data type
        with torch.no_grad():
            temp = torch.tensor(temp)
            temp = temp.reshape(1,1,temp.shape[0],temp.shape[1]).float()
            temp = temp.to(self.device)
            encoded_data = self.model.encode(temp).to('cpu')
            
            # convert encoded frame to proper data type
            encoded_data = encoded_data.squeeze().numpy().tolist()
        
        # Return the encoded frame of the proper data type
        return encoded_data
    
    # Adds temperature and cure frame to save buffers
    # @param the temperature frame
    # @param the cure frame
    def save_frame(self, temp, cure):
        
        self.save_temp_buffer.append(np.array(temp))
        self.save_cure_buffer.append(np.array(cure))
    
    # Calculates the blurred front location
    # @param cure field used to determine front location
    # @return blurred front location 
    def get_front_location(self, cure):
        
        # Determine blurring factor
        blur_half_range = 0.04
        
        # Solve for cure front
        front_location = np.concatenate(((abs(np.diff(cure,axis=0))) > 0.25, np.zeros((1,40))))
        distance_indices = np.arange(len(cure))
                
        # Apply blur
        for j in range(len(cure[0,:])):
            front_slice = distance_indices[(front_location[:,j] == 1.0)]
            for ind in range(len(front_slice)):
                i = front_slice[ind]
                start_blur = int(round(i - blur_half_range * len(cure)))
                if start_blur < 0:
                    start_blur = 0
                end_blur = int(round(i + blur_half_range * len(cure)))
                if end_blur > len(cure) - 1:
                    end_blur = len(cure) - 1
                for ii in range(start_blur, end_blur+1):
                    if ii < i:
                        front_location[ii,j] =  max((ii - int(round(i-blur_half_range*len(cure)))) / (i - int(round(i-blur_half_range*len(cure)))), front_location[ii,j])
                    elif ii == i:
                        front_location[ii,j] = 1.0
                    elif ii > i:
                        front_location[ii,j] = max(1.0 - (ii - i) / (int(round(i+blur_half_range*len(cure))) - i), front_location[ii,j])
                
        # Format data
        front_location = torch.tensor(front_location)
        front_location = front_location.reshape(1,1,front_location.shape[0],front_location.shape[1]).float()
                
        return front_location

    # Gets the quantized temperature field training target
    # @param temperature field to be quantized
    # @return the qunatized version of the parameter temperature field
    def get_quantized_temp(self, temp):
        
        with torch.no_grad():
            low_target =      ((1.0*(temp>=0.00) + 1.0*(temp<0.10))-1.0)
            low_mid_target =  ((1.0*(temp>=0.10) + 1.0*(temp<0.65))-1.0)
            mid_target =      ((1.0*(temp>=0.65) + 1.0*(temp<0.70))-1.0)
            mid_high_target = ((1.0*(temp>=0.70) + 1.0*(temp<0.75))-1.0)
            high_target =     ((1.0*(temp>=0.75) + 1.0*(temp<=1.0))-1.0)
    
        quantized_temp = torch.cat((low_target, low_mid_target, mid_target, mid_high_target, high_target), 1)
        return quantized_temp
    
    # Calculates the training target given the objective function ID
    # @param temperature field used to get target
    # @param cure field used to get target
    # @return target given input temperature, cure, and objective function ID
    def get_target(self, temp, cure):
        
        # Convert temperature frame to proper data form
        with torch.no_grad():
            temp = torch.tensor(temp)
            temp = temp.reshape(1,1,temp.shape[0],temp.shape[1]).float()
          
            if self.objective_fnc == 1:
                target = temp
            
            elif self.objective_fnc == 2:
                front_location = self.get_front_location(cure)
                target = torch.cat((temp, front_location), 1)
                
            elif self.objective_fnc == 3:
                front_location = self.get_front_location(cure)
                cure = torch.tensor(cure)
                cure = cure.reshape(1,1,cure.shape[0],cure.shape[1]).float()
                target = torch.cat((temp, front_location, cure), 1)
                
            elif self.objective_fnc == 5:
                target = self.get_quantized_temp(temp)
                
            elif self.objective_fnc == 6:
                quantized_temp = self.get_quantized_temp(temp)
                target = torch.cat((temp, quantized_temp), 1)
                
            elif self.objective_fnc == 7:
                quantized_temp = self.get_quantized_temp(temp)
                front_location = self.get_front_location(cure)
                target = torch.cat((temp, quantized_temp, front_location), 1)
                
            elif self.objective_fnc == 8:
                quantized_temp = self.get_quantized_temp(temp)
                front_location = self.get_front_location(cure)
                cure = torch.tensor(cure)
                cure = cure.reshape(1,1,cure.shape[0],cure.shape[1]).float()
                target = torch.cat((temp, quantized_temp, front_location, cure), 1)
            
        target = target.to(self.device)
        
        return target
    
    # Calcualtes the loss for autoencoder learning
    # @param Autoencoder rebuilt differentiable data
    # @param Target of objective function
    # @return Differentialable training loss
    def get_loss(self, rebuilt_data, target):
        
        # Get rebuilt loss
        if self.objective_fnc == 8:
            curr_loss = self.criterion(rebuilt_data, target)
        
        elif self.objective_fnc == 7:
            curr_loss = self.criterion(rebuilt_data[0,0:7,:,:], target[0,0:7,:,:])
            
        elif self.objective_fnc == 6:
            curr_loss = self.criterion(rebuilt_data[0,0:6,:,:], target[0,0:6,:,:])
        
        elif self.objective_fnc == 5:
            curr_loss = self.criterion(rebuilt_data[0,0:5,:,:], target[0,0:5,:,:])
            
        elif self.objective_fnc == 3:
            curr_loss = self.criterion(rebuilt_data[0,0:3,:,:], target[0,0:3,:,:])
            
        elif self.objective_fnc == 2:
            curr_loss = self.criterion(rebuilt_data[0,0:2,:,:], target[0,0:2,:,:])
            
        elif self.objective_fnc == 1:
            curr_loss = self.criterion(rebuilt_data[0,0,:,:], target[0,0,:,:])
            
        return curr_loss
        
    # Updates the autoencoder
    # @param temperature field to be added to training buffer
    # @param cure field to be added to training buffer
    # @return average epoch training loss or -1 if no optimization epoch occured
    def learn(self, temp, cure):
        
        # Store the current temperature and cure frames in the learning buffers
        self.temp_buffer.append(np.array(temp))
        self.cure_buffer.append(np.array(cure))
        
        # If the frame buffer is full, perform one epoch of stochastic gradient descent
        if len(self.temp_buffer) >= self.buffer_size:
            
            # Step through frame buffer
            RMS_loss = 0.0
            rand_indcies = np.random.permutation(self.buffer_size)
            for i in range(self.buffer_size):
                
                # Get temperature and cure at random location from buffer
                curr_temp = self.temp_buffer[rand_indcies[i]]
                curr_cure = self.cure_buffer[rand_indcies[i]]
                
                # Calculate target
                target = self.get_target(curr_temp, curr_cure)
                        
                # Format and forward propagate temperature data
                curr_temp = torch.tensor(curr_temp)
                curr_temp = curr_temp.reshape(1,1,curr_temp.shape[0],curr_temp.shape[1]).float()
                curr_temp = curr_temp.to(self.device)
                rebuilt_data = self.model.forward(curr_temp)
        
                # Get the loss
                curr_loss = self.get_loss(rebuilt_data, target)
        
                # Take optimization step and learning rate step
                self.optimizer.zero_grad()
                curr_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                
                # Sum the epoch's total loss
                RMS_loss = RMS_loss + np.sqrt(curr_loss.item())
            
            # Empty frame buffer
            self.temp_buffer = []
            self.cure_buffer = []

            # Return the average RMS reconstruction error
            return RMS_loss / (float(self.num_output_layers) * self.buffer_size)
        
        # return -1 if no optimization epoch occured
        return -1
    
    # Calculates the RMS error in the temperature field reconstruction over a set of saved temperature fields
    # @param the set of temperature fields over which the RMS error is computed
    # @return RMS error in temperature field reconstruction in percent points
    def get_temp_error(self, temp_array):
        print("Getting temperature reconstruction error...")
        
        RMS_error = 0.0
        with torch.no_grad():
            for i in range(len(temp_array)):
                
                # Format temperature field data
                temp = torch.tensor(temp_array[i])
                temp = temp.reshape(1,1,temp.shape[0],temp.shape[1]).float()
            
                # Forward propogate the frame through the autoencoder
                temp = temp.to(self.device)
                rebuilt_temp = self.model.forward(temp)[0,0,:,:].to('cpu').numpy().squeeze()
                
                # Calcualte temperature reconstruction error
                RMS_error = RMS_error + np.sqrt(np.mean((rebuilt_temp-temp_array[i])**2.0))
                
        # Get RMS reconstruction error
        return (100.0*RMS_error) / len(temp_array)
    
    # Calculates the RMS error in the front location reconstruction over a set of saved temperature fields
    # @param the set of temperature fields over which the RMS error is computed
    # @param the set of cure fields over which the RMS error is computed
    # @return RMS error in front location reconstruction in percent points
    def get_front_error(self, temp_array, cure_array):
        print("Getting front location reconstruction error...")
        
        RMS_error = 0.0
        with torch.no_grad():
            for i in range(len(temp_array)):
                
                # Format temperature field data
                temp = torch.tensor(temp_array[i])
                temp = temp.reshape(1,1,temp.shape[0],temp.shape[1]).float()
            
                # Forward propogate the frame through the autoencoder
                temp = temp.to(self.device)
                rebuilt_front = self.model.forward(temp)[0,1,:,:].to('cpu').numpy().squeeze()
                
                # get the current front location
                front = self.get_front_location(cure_array[i])[0,0,:,:].to('cpu').numpy().squeeze()
                
                # Calcualte temperature reconstruction error
                RMS_error = RMS_error + np.sqrt(np.mean((rebuilt_front-front)**2.0))
                
        # Get RMS reconstruction error
        return (100.0*RMS_error) / len(temp_array)
    
    # Calculates the RMS error in the cure field reconstruction over a set of saved temperature fields
    # @param the set of temperature fields over which the RMS error is computed
    # @param the set of cure fields over which the RMS error is computed
    # @return RMS error in cure field reconstruction in percent points
    def get_cure_error(self, temp_array, cure_array):
        print("Getting cure reconstruction error...")
        
        RMS_error = 0.0
        with torch.no_grad():
            for i in range(len(temp_array)):
                
                # Format temperature field data
                temp = torch.tensor(temp_array[i])
                temp = temp.reshape(1,1,temp.shape[0],temp.shape[1]).float()
            
                # Forward propogate the frame through the autoencoder
                temp = temp.to(self.device)
                rebuilt_cure = self.model.forward(temp)[0,2,:,:].to('cpu').numpy().squeeze()
                
                # Calcualte temperature reconstruction error
                RMS_error = RMS_error + np.sqrt(np.mean((rebuilt_cure-cure_array[i])**2.0))
                
        # Get RMS reconstruction error
        return (100.0*RMS_error) / len(temp_array)
    
    # Saves the training data and trained autoencoder model
    # @param training loss curve
    # @return path at which data has been saved
    def save(self, training_curve):
        print("Saving autoencoder results...")

        # Store data to dictionary
        data = {
            'x_dim' : self.x_dim, 
            'y_dim' : self.y_dim, 
            'bottleneck' : self.bottleneck, 
            'num_output_layers' : self.num_output_layers, 
            'objective_fnc' : self.objective_fnc, 
            'buffer_size' : self.buffer_size, 
            'training_curve' : np.array(training_curve),
            'temp_array' : self.save_temp_buffer,
            'cure_array' : self.save_cure_buffer,
            'autoencoder' : self.model,
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
        save_file = path + "/output"
        with open(save_file, 'wb') as file:
            pickle.dump(data, file)

        return path

    # Draw and save the training curve
    # @param training curve to be drawn
    # @param path at which training curve is saved
    def draw_training_curve(self, training_curve, path):
        print("Plotting autoencoder training curve...")
        
        plt.clf()
        plt.title("Autoencoder Learning Curve",fontsize='xx-large')
        plt.xlabel("Optimization Epoch",fontsize='large')
        plt.ylabel("RMS Reconstruction Error",fontsize='large')
        plt.plot([*range(len(training_curve))],training_curve,lw=2.5,c='r')
        plt.yscale("log")
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/autoencoder_learning.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()
    
    # Draws the current frame for autoencoder with objective function 1
    # @param x grid over which data are plotted
    # @param y grid over which data are plotted
    # @param temperature field
    # @param rebuilt temperature field
    # @param path to which rendered video is saved (in folder called 'video')
    # @param frame number
    def draw_obj_1(self, x_grid, y_grid, temp, rebuilt_temp, path, frame_number):
        
        # Make figure
        plt.cla()
        plt.clf()
        fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True)
        fig.set_size_inches(16,2.6667)
        
        # Plot temperature
        ax0.pcolormesh(x_grid, y_grid, np.transpose(temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax0.tick_params(axis='x',labelsize=12)
        ax0.tick_params(axis='y',labelsize=12)
        ax0.set_aspect(0.25, adjustable='box')
        ax0.set_title('True Temperature Field (Observed)',fontsize='x-large')
        
        # Plot rebuilt temperature
        c1 = ax1.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        cbar1 = fig.colorbar(c1, ax=ax1, pad=0.1)
        cbar1.set_label('T / '+r'$T_{ref}$'+'   [-]',labelpad=10,fontsize='large')
        cbar1.ax.tick_params(labelsize=12)
        ax1.tick_params(axis='x',labelsize=12)
        ax1.tick_params(axis='y',labelsize=12)
        ax1.set_aspect(0.25, adjustable='box')
        ax1.set_title('Rebuilt Temperature Field',fontsize='x-large')
        
        # Set title and save
        plt.savefig(path+"/"+str(frame_number).zfill(4)+'.png', dpi=100)
        plt.close()
    
    # Draws the current frame for autoencoder with objective function 2
    # @param x grid over which data are plotted
    # @param y grid over which data are plotted
    # @param temperature field
    # @param front location
    # @param rebuilt temperature field
    # @param rebuilt front location
    # @param path to which rendered video is saved (in folder called 'video')
    # @param frame number
    def draw_obj_2(self, x_grid, y_grid, temp, front, rebuilt_temp, rebuilt_front, path, frame_number):
        
        # Make figure
        plt.cla()
        plt.clf()
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, constrained_layout=True)
        fig.set_size_inches(16,5.333)
        
        # Plot temperature
        ax0.pcolormesh(x_grid, y_grid, np.transpose(temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax0.tick_params(axis='x',labelsize=12)
        ax0.tick_params(axis='y',labelsize=12)
        ax0.set_aspect(0.25, adjustable='box')
        ax0.set_title('True Temperature Field (Observed)',fontsize='x-large')
        
        # Plot rebuilt temperature
        c1 = ax1.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        cbar1 = fig.colorbar(c1, ax=ax1, pad=0.1)
        cbar1.set_label('T / '+r'$T_{ref}$'+'   [-]',labelpad=10,fontsize='large')
        cbar1.ax.tick_params(labelsize=12)
        ax1.tick_params(axis='x',labelsize=12)
        ax1.tick_params(axis='y',labelsize=12)
        ax1.set_aspect(0.25, adjustable='box')
        ax1.set_title('Rebuilt Temperature Field',fontsize='x-large')
        
        # Plot front location
        ax2.pcolormesh(x_grid, y_grid, np.transpose(front), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
        ax2.tick_params(axis='x',labelsize=12)
        ax2.tick_params(axis='y',labelsize=12)
        ax2.set_aspect(0.25, adjustable='box')
        ax2.set_title('True Front Mask (Unobserved)',fontsize='x-large')
        
        # Plot inferred front location
        ax3.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_front), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
        ax3.tick_params(axis='x',labelsize=12)
        ax3.tick_params(axis='y',labelsize=12)
        ax3.set_aspect(0.25, adjustable='box')
        ax3.set_title('Inferred Front Mask',fontsize='x-large')
        
        # Set title and save
        plt.savefig(path+"/"+str(frame_number).zfill(4)+'.png', dpi=100)
        plt.close()
    
    # Draws the current frame for autoencoder with objective function 3
    # @param x grid over which data are plotted
    # @param y grid over which data are plotted
    # @param temperature field
    # @param front location
    # @param cure field
    # @param rebuilt temperature field
    # @param rebuilt front location
    # @param rebuilt cure field
    # @param path to which rendered video is saved (in folder called 'video')
    # @param frame number
    def draw_obj_3(self, x_grid, y_grid, temp, front, cure, rebuilt_temp, rebuilt_front, rebuilt_cure, path, frame_number):
        
        # Make figure
        plt.cla()
        plt.clf()
        fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2, constrained_layout=True)
        fig.set_size_inches(16,8)
        
        # Plot temperature
        ax0.pcolormesh(x_grid, y_grid, np.transpose(temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax0.tick_params(axis='x',labelsize=12)
        ax0.tick_params(axis='y',labelsize=12)
        ax0.set_aspect(0.25, adjustable='box')
        ax0.set_title('True Temperature Field (Observed)',fontsize='x-large')
        
        # Plot rebuilt temperature
        c1 = ax1.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        cbar1 = fig.colorbar(c1, ax=ax1, pad=0.025)
        cbar1.set_label('T / '+r'$T_{ref}$'+'   [-]',labelpad=10,fontsize='large')
        cbar1.ax.tick_params(labelsize=12)
        ax1.tick_params(axis='x',labelsize=12)
        ax1.tick_params(axis='y',labelsize=12)
        ax1.set_aspect(0.25, adjustable='box')
        ax1.set_title('Rebuilt Temperature Field',fontsize='x-large')
        
        # Plot front location
        ax2.pcolormesh(x_grid, y_grid, np.transpose(front), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
        ax2.tick_params(axis='x',labelsize=12)
        ax2.tick_params(axis='y',labelsize=12)
        ax2.set_aspect(0.25, adjustable='box')
        ax2.set_title('True Front Mask (Unobserved)',fontsize='x-large')
        
        # Plot inferred front location
        ax3.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_front), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
        ax3.tick_params(axis='x',labelsize=12)
        ax3.tick_params(axis='y',labelsize=12)
        ax3.set_aspect(0.25, adjustable='box')
        ax3.set_title('Inferred Front Mask',fontsize='x-large')
        
        # Cure frame
        ax4.pcolormesh(x_grid, y_grid, np.transpose(cure), shading='gouraud', cmap='YlOrRd', vmin=0.0, vmax=1.0)
        ax4.tick_params(axis='x',labelsize=12)
        ax4.tick_params(axis='y',labelsize=12)
        ax4.set_aspect(0.25, adjustable='box')
        ax4.set_title('True Cure Field (Unobserved)',fontsize='x-large')
        
        # Inferred cure degree
        c5 = ax5.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_cure), shading='gouraud', cmap='YlOrRd', vmin=0.0, vmax=1.0)
        cbar5 = fig.colorbar(c5, ax=ax5, pad=0.025)
        cbar5.set_label('Degree Cure [-]',labelpad=10,fontsize='large')
        cbar5.ax.tick_params(labelsize=12)
        ax5.tick_params(axis='x',labelsize=12)
        ax5.tick_params(axis='y',labelsize=12)
        ax5.set_aspect(0.25, adjustable='box')
        ax5.set_title('Inferred Cure Field',fontsize='x-large')
        
        # Set title and save
        plt.savefig(path+"/"+str(frame_number).zfill(4)+'.png', dpi=100)
        plt.close()
    
    # Draws the current frame for autoencoder with objective function 5
    # @param x grid over which data are plotted
    # @param y grid over which data are plotted
    # @param temperature field
    # @param rebuilt quantized temperature field
    # @param path to which rendered video is saved (in folder called 'video')
    # @param frame number
    def draw_obj_5(self, x_grid, y_grid, temp, rebuilt_temp, path, frame_number):

        # Make figure
        plt.cla()
        plt.clf()
        fig, ((ax0, ax1), (ax2, ax3), (ax4,ax5)) = plt.subplots(3, 2, constrained_layout=True)
        fig.set_size_inches(16,8)
        
        # Plot temperature
        c0 = ax0.pcolormesh(x_grid, y_grid, np.transpose(temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        cbar0 = fig.colorbar(c0, ax=ax0, pad=0.025)
        cbar0.set_label('T / '+r'$T_{ref}$'+'   [-]',labelpad=10,fontsize='large')
        cbar0.ax.tick_params(labelsize=12)
        ax0.tick_params(axis='x',labelsize=12)
        ax0.tick_params(axis='y',labelsize=12)
        ax0.set_aspect(0.25, adjustable='box')
        ax0.set_title('True Temperature Field (Observed)',fontsize='x-large')
        
        # Plot rebuilt low temperature mask
        ax1.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_temp[0]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax1.tick_params(axis='x',labelsize=12)
        ax1.tick_params(axis='y',labelsize=12)
        ax1.set_aspect(0.25, adjustable='box')
        ax1.set_title('Rebuilt Low Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt low mid temperature mask
        ax2.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_temp[1]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax2.tick_params(axis='x',labelsize=12)
        ax2.tick_params(axis='y',labelsize=12)
        ax2.set_aspect(0.25, adjustable='box')
        ax2.set_title('Rebuilt Low-Mid Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt mid temperature mask
        ax3.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_temp[2]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax3.tick_params(axis='x',labelsize=12)
        ax3.tick_params(axis='y',labelsize=12)
        ax3.set_aspect(0.25, adjustable='box')
        ax3.set_title('Rebuilt Mid Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt mid high temperature mask
        ax4.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_temp[3]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax4.tick_params(axis='x',labelsize=12)
        ax4.tick_params(axis='y',labelsize=12)
        ax4.set_aspect(0.25, adjustable='box')
        ax4.set_title('Rebuilt Mid-High Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt high temperature mask
        ax5.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_temp[4]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax5.tick_params(axis='x',labelsize=12)
        ax5.tick_params(axis='y',labelsize=12)
        ax5.set_aspect(0.25, adjustable='box')
        ax5.set_title('Rebuilt High Temperature Mask',fontsize='x-large')
        
        # Set title and save
        plt.savefig(path+"/"+str(frame_number).zfill(4)+'.png', dpi=100)
        plt.close()
        
    # Draws the current frame for autoencoder with objective function 5
    # @param x grid over which data are plotted
    # @param y grid over which data are plotted
    # @param temperature field
    # @param rebuilt data from autoencoder
    # @param path to which rendered video is saved (in folder called 'video')
    # @param frame number
    def draw_obj_6(self, x_grid, y_grid, temp, rebuilt_data, path, frame_number):

        # Make figure
        plt.cla()
        plt.clf()
        fig, ((ax0, ax1), (ax2, ax3), (ax4,ax5), (ax6,ax7)) = plt.subplots(4, 2, constrained_layout=True)
        fig.set_size_inches(16,10.6667)
        ax7.set_axis_off()
        
        # Plot temperature
        ax0.pcolormesh(x_grid, y_grid, np.transpose(temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax0.tick_params(axis='x',labelsize=12)
        ax0.tick_params(axis='y',labelsize=12)
        ax0.set_aspect(0.25, adjustable='box')
        ax0.set_title('True Temperature Field (Observed)',fontsize='x-large')
        
        # Plot rebuilt temperature
        c1 = ax1.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[0]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        cbar1 = fig.colorbar(c1, ax=ax1, pad=0.025)
        cbar1.set_label('T / '+r'$T_{ref}$'+'   [-]',labelpad=10,fontsize='large')
        cbar1.ax.tick_params(labelsize=12)
        ax1.tick_params(axis='x',labelsize=12)
        ax1.tick_params(axis='y',labelsize=12)
        ax1.set_aspect(0.25, adjustable='box')
        ax1.set_title('Rebuilt Temperature Field',fontsize='x-large')
        
        # Plot rebuilt low temperature mask
        ax2.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[1]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax2.tick_params(axis='x',labelsize=12)
        ax2.tick_params(axis='y',labelsize=12)
        ax2.set_aspect(0.25, adjustable='box')
        ax2.set_title('Rebuilt Low Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt low mid temperature mask
        ax3.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[2]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax3.tick_params(axis='x',labelsize=12)
        ax3.tick_params(axis='y',labelsize=12)
        ax3.set_aspect(0.25, adjustable='box')
        ax3.set_title('Rebuilt Low-Mid Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt mid temperature mask
        ax4.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[3]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax4.tick_params(axis='x',labelsize=12)
        ax4.tick_params(axis='y',labelsize=12)
        ax4.set_aspect(0.25, adjustable='box')
        ax4.set_title('Rebuilt Mid Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt mid high temperature mask
        ax5.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[4]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax5.tick_params(axis='x',labelsize=12)
        ax5.tick_params(axis='y',labelsize=12)
        ax5.set_aspect(0.25, adjustable='box')
        ax5.set_title('Rebuilt Mid-High Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt high temperature mask
        ax6.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[5]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax6.tick_params(axis='x',labelsize=12)
        ax6.tick_params(axis='y',labelsize=12)
        ax6.set_aspect(0.25, adjustable='box')
        ax6.set_title('Rebuilt High Temperature Mask',fontsize='x-large')
        
        # Set title and save
        plt.savefig(path+"/"+str(frame_number).zfill(4)+'.png', dpi=100)
        plt.close()
        
    # Draws the current frame for autoencoder with objective function 5
    # @param x grid over which data are plotted
    # @param y grid over which data are plotted
    # @param temperature field
    # @param true front mask
    # @param rebuilt data from autoencoder
    # @param path to which rendered video is saved (in folder called 'video')
    # @param frame number
    def draw_obj_7(self, x_grid, y_grid, temp, front, rebuilt_data, path, frame_number):

        # Make figure
        plt.cla()
        plt.clf()
        fig, ((ax0, ax1), (ax2, ax3), (ax4,ax5), (ax6,ax7), (ax8,ax9)) = plt.subplots(5, 2, constrained_layout=True)
        fig.set_size_inches(16,13.3333)
        ax9.set_axis_off()
        
        # Plot temperature
        ax0.pcolormesh(x_grid, y_grid, np.transpose(temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax0.tick_params(axis='x',labelsize=12)
        ax0.tick_params(axis='y',labelsize=12)
        ax0.set_aspect(0.25, adjustable='box')
        ax0.set_title('True Temperature Field (Observed)',fontsize='x-large')
        
        # Plot rebuilt temperature
        c1 = ax1.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[0]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        cbar1 = fig.colorbar(c1, ax=ax1, pad=0.025)
        cbar1.set_label('T / '+r'$T_{ref}$'+'   [-]',labelpad=10,fontsize='large')
        cbar1.ax.tick_params(labelsize=12)
        ax1.tick_params(axis='x',labelsize=12)
        ax1.tick_params(axis='y',labelsize=12)
        ax1.set_aspect(0.25, adjustable='box')
        ax1.set_title('Rebuilt Temperature Field',fontsize='x-large')
        
        # Plot front location
        ax2.pcolormesh(x_grid, y_grid, np.transpose(front), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
        ax2.tick_params(axis='x',labelsize=12)
        ax2.tick_params(axis='y',labelsize=12)
        ax2.set_aspect(0.25, adjustable='box')
        ax2.set_title('True Front Mask (Unobserved)',fontsize='x-large')
        
        # Plot inferred front location
        ax3.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[6]), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
        ax3.tick_params(axis='x',labelsize=12)
        ax3.tick_params(axis='y',labelsize=12)
        ax3.set_aspect(0.25, adjustable='box')
        ax3.set_title('Inferred Front Mask',fontsize='x-large')
        
        # Plot rebuilt low temperature mask
        ax4.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[1]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax4.tick_params(axis='x',labelsize=12)
        ax4.tick_params(axis='y',labelsize=12)
        ax4.set_aspect(0.25, adjustable='box')
        ax4.set_title('Rebuilt Low Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt low mid temperature mask
        ax5.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[2]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax5.tick_params(axis='x',labelsize=12)
        ax5.tick_params(axis='y',labelsize=12)
        ax5.set_aspect(0.25, adjustable='box')
        ax5.set_title('Rebuilt Low-Mid Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt mid temperature mask
        ax6.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[3]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax6.tick_params(axis='x',labelsize=12)
        ax6.tick_params(axis='y',labelsize=12)
        ax6.set_aspect(0.25, adjustable='box')
        ax6.set_title('Rebuilt Mid Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt mid high temperature mask
        ax7.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[4]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax7.tick_params(axis='x',labelsize=12)
        ax7.tick_params(axis='y',labelsize=12)
        ax7.set_aspect(0.25, adjustable='box')
        ax7.set_title('Rebuilt Mid-High Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt high temperature mask
        ax8.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[5]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax8.tick_params(axis='x',labelsize=12)
        ax8.tick_params(axis='y',labelsize=12)
        ax8.set_aspect(0.25, adjustable='box')
        ax8.set_title('Rebuilt High Temperature Mask',fontsize='x-large')
        
        # Set title and save
        plt.savefig(path+"/"+str(frame_number).zfill(4)+'.png', dpi=100)
        plt.close()
        
    # Draws the current frame for autoencoder with objective function 5
    # @param x grid over which data are plotted
    # @param y grid over which data are plotted
    # @param temperature field
    # @param true front mask
    # @param true cure field
    # @param rebuilt data from autoencoder
    # @param path to which rendered video is saved (in folder called 'video')
    # @param frame number
    def draw_obj_8(self, x_grid, y_grid, temp, front, cure, rebuilt_data, path, frame_number):

        # Make figure
        plt.cla()
        plt.clf()
        fig, ((ax0, ax1), (ax2, ax3), (ax4,ax5), (ax6,ax7), (ax8,ax9), (ax10,ax11)) = plt.subplots(6, 2, constrained_layout=True)
        fig.set_size_inches(16,16)
        ax11.set_axis_off()
        
        # Plot temperature
        ax0.pcolormesh(x_grid, y_grid, np.transpose(temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax0.tick_params(axis='x',labelsize=12)
        ax0.tick_params(axis='y',labelsize=12)
        ax0.set_aspect(0.25, adjustable='box')
        ax0.set_title('True Temperature Field (Observed)',fontsize='x-large')
        
        # Plot rebuilt temperature
        c1 = ax1.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[0]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        cbar1 = fig.colorbar(c1, ax=ax1, pad=0.025)
        cbar1.set_label('T / '+r'$T_{ref}$'+'   [-]',labelpad=10,fontsize='large')
        cbar1.ax.tick_params(labelsize=12)
        ax1.tick_params(axis='x',labelsize=12)
        ax1.tick_params(axis='y',labelsize=12)
        ax1.set_aspect(0.25, adjustable='box')
        ax1.set_title('Rebuilt Temperature Field',fontsize='x-large')
        
        # Plot front location
        ax2.pcolormesh(x_grid, y_grid, np.transpose(front), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
        ax2.tick_params(axis='x',labelsize=12)
        ax2.tick_params(axis='y',labelsize=12)
        ax2.set_aspect(0.25, adjustable='box')
        ax2.set_title('True Front Mask (Unobserved)',fontsize='x-large')
        
        # Plot inferred front location
        ax3.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[6]), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
        ax3.tick_params(axis='x',labelsize=12)
        ax3.tick_params(axis='y',labelsize=12)
        ax3.set_aspect(0.25, adjustable='box')
        ax3.set_title('Inferred Front Mask',fontsize='x-large')
        
        # Cure frame
        ax4.pcolormesh(x_grid, y_grid, np.transpose(cure), shading='gouraud', cmap='YlOrRd', vmin=0.0, vmax=1.0)
        ax4.tick_params(axis='x',labelsize=12)
        ax4.tick_params(axis='y',labelsize=12)
        ax4.set_aspect(0.25, adjustable='box')
        ax4.set_title('True Cure Field (Unobserved)',fontsize='x-large')
        
        # Inferred cure degree
        c5 = ax5.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[7]), shading='gouraud', cmap='YlOrRd', vmin=0.0, vmax=1.0)
        cbar5 = fig.colorbar(c5, ax=ax5, pad=0.025)
        cbar5.set_label('Degree Cure [-]',labelpad=10,fontsize='large')
        cbar5.ax.tick_params(labelsize=12)
        ax5.tick_params(axis='x',labelsize=12)
        ax5.tick_params(axis='y',labelsize=12)
        ax5.set_aspect(0.25, adjustable='box')
        ax5.set_title('Inferred Cure Field',fontsize='x-large')
        
        # Plot rebuilt low temperature mask
        ax6.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[1]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax6.tick_params(axis='x',labelsize=12)
        ax6.tick_params(axis='y',labelsize=12)
        ax6.set_aspect(0.25, adjustable='box')
        ax6.set_title('Rebuilt Low Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt low mid temperature mask
        ax7.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[2]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax7.tick_params(axis='x',labelsize=12)
        ax7.tick_params(axis='y',labelsize=12)
        ax7.set_aspect(0.25, adjustable='box')
        ax7.set_title('Rebuilt Low-Mid Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt mid temperature mask
        ax8.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[3]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax8.tick_params(axis='x',labelsize=12)
        ax8.tick_params(axis='y',labelsize=12)
        ax8.set_aspect(0.25, adjustable='box')
        ax8.set_title('Rebuilt Mid Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt mid high temperature mask
        ax9.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[4]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax9.tick_params(axis='x',labelsize=12)
        ax9.tick_params(axis='y',labelsize=12)
        ax9.set_aspect(0.25, adjustable='box')
        ax9.set_title('Rebuilt Mid-High Temperature Mask',fontsize='x-large')
        
        # Plot rebuilt high temperature mask
        ax10.pcolormesh(x_grid, y_grid, np.transpose(rebuilt_data[5]), shading='nearest', cmap='binary', vmin=0.0, vmax=1.0)
        ax10.tick_params(axis='x',labelsize=12)
        ax10.tick_params(axis='y',labelsize=12)
        ax10.set_aspect(0.25, adjustable='box')
        ax10.set_title('Rebuilt High Temperature Mask',fontsize='x-large')
        
        # Set title and save
        plt.savefig(path+"/"+str(frame_number).zfill(4)+'.png', dpi=100)
        plt.close()
    
    # Renders video showing reconstruction of all objective functions based on save frame buffer
    # @param path to which rendered video is saved (in folder called 'video')
    def render(self, path):
        print("Rendering...")
        
        # Find save paths
        if not os.path.isdir(path):
            os.mkdir(path)
        path = path + "/video"
        if not os.path.isdir(path):
            os.mkdir(path)
        
        x_grid, y_grid = np.meshgrid(np.linspace(0,1,self.x_dim), np.linspace(0,1,self.y_dim))
        for i in range(len(self.save_temp_buffer)):
            with torch.no_grad():
                # Get rebuilt data
                temp = torch.tensor(self.save_temp_buffer[i])
                temp = temp.reshape(1,1,temp.shape[0],temp.shape[1]).float()
                temp = temp.to(self.device)
                rebuilt_data = self.model.forward(temp)
                
                # Get temperature field, front location, and cure field
                temp = self.save_temp_buffer[i]
                front = self.get_front_location(self.save_cure_buffer[i])[0,0,:,:].to('cpu').numpy().squeeze()
                cure = self.save_cure_buffer[i]
                
                if len(rebuilt_data[0,:,0,0]) == 1:
                    rebuilt_temp = rebuilt_data[0,0,:,:].to('cpu').numpy().squeeze()
                    
                elif len(rebuilt_data[0,:,0,0]) == 2:
                    rebuilt_temp = rebuilt_data[0,0,:,:].to('cpu').numpy().squeeze()
                    rebuilt_front = rebuilt_data[0,1,:,:].to('cpu').numpy().squeeze()
                    
                elif len(rebuilt_data[0,:,0,0]) == 3:
                    rebuilt_temp = rebuilt_data[0,0,:,:].to('cpu').numpy().squeeze()
                    rebuilt_front = rebuilt_data[0,1,:,:].to('cpu').numpy().squeeze()
                    rebuilt_cure = rebuilt_data[0,2,:,:].to('cpu').numpy().squeeze()
                    
                elif len(rebuilt_data[0,:,0,0]) == 5:
                    rebuilt_low_temp = rebuilt_data[0,0,:,:].to('cpu').numpy().squeeze()
                    rebuilt_low_mid_temp = rebuilt_data[0,1,:,:].to('cpu').numpy().squeeze()
                    rebuilt_mid_temp = rebuilt_data[0,2,:,:].to('cpu').numpy().squeeze()
                    rebuilt_mid_high_temp = rebuilt_data[0,3,:,:].to('cpu').numpy().squeeze()
                    rebuilt_high_temp = rebuilt_data[0,4,:,:].to('cpu').numpy().squeeze()
                    render_data = [rebuilt_low_temp, rebuilt_low_mid_temp, rebuilt_mid_temp, rebuilt_mid_high_temp, rebuilt_high_temp]
                    
                elif len(rebuilt_data[0,:,0,0]) == 6:
                    rebuilt_temp = rebuilt_data[0,0,:,:].to('cpu').numpy().squeeze()
                    rebuilt_low_temp = rebuilt_data[0,1,:,:].to('cpu').numpy().squeeze()
                    rebuilt_low_mid_temp = rebuilt_data[0,2,:,:].to('cpu').numpy().squeeze()
                    rebuilt_mid_temp = rebuilt_data[0,3,:,:].to('cpu').numpy().squeeze()
                    rebuilt_mid_high_temp = rebuilt_data[0,4,:,:].to('cpu').numpy().squeeze()
                    rebuilt_high_temp = rebuilt_data[0,5,:,:].to('cpu').numpy().squeeze()
                    render_data = [rebuilt_temp, rebuilt_low_temp, rebuilt_low_mid_temp, rebuilt_mid_temp, rebuilt_mid_high_temp, rebuilt_high_temp]
                    
                elif len(rebuilt_data[0,:,0,0]) == 7:
                    rebuilt_temp = rebuilt_data[0,0,:,:].to('cpu').numpy().squeeze()
                    rebuilt_low_temp = rebuilt_data[0,1,:,:].to('cpu').numpy().squeeze()
                    rebuilt_low_mid_temp = rebuilt_data[0,2,:,:].to('cpu').numpy().squeeze()
                    rebuilt_mid_temp = rebuilt_data[0,3,:,:].to('cpu').numpy().squeeze()
                    rebuilt_mid_high_temp = rebuilt_data[0,4,:,:].to('cpu').numpy().squeeze()
                    rebuilt_high_temp = rebuilt_data[0,5,:,:].to('cpu').numpy().squeeze()
                    rebuilt_front = rebuilt_data[0,6,:,:].to('cpu').numpy().squeeze()
                    render_data = [rebuilt_temp, rebuilt_low_temp, rebuilt_low_mid_temp, rebuilt_mid_temp, rebuilt_mid_high_temp, rebuilt_high_temp, rebuilt_front]
                    
                elif len(rebuilt_data[0,:,0,0]) == 8:
                    rebuilt_temp = rebuilt_data[0,0,:,:].to('cpu').numpy().squeeze()
                    rebuilt_low_temp = rebuilt_data[0,1,:,:].to('cpu').numpy().squeeze()
                    rebuilt_low_mid_temp = rebuilt_data[0,2,:,:].to('cpu').numpy().squeeze()
                    rebuilt_mid_temp = rebuilt_data[0,3,:,:].to('cpu').numpy().squeeze()
                    rebuilt_mid_high_temp = rebuilt_data[0,4,:,:].to('cpu').numpy().squeeze()
                    rebuilt_high_temp = rebuilt_data[0,5,:,:].to('cpu').numpy().squeeze()
                    rebuilt_front = rebuilt_data[0,6,:,:].to('cpu').numpy().squeeze()
                    rebuilt_cure = rebuilt_data[0,7,:,:].to('cpu').numpy().squeeze()
                    render_data = [rebuilt_temp, rebuilt_low_temp, rebuilt_low_mid_temp, rebuilt_mid_temp, rebuilt_mid_high_temp, rebuilt_high_temp, rebuilt_front, rebuilt_cure]
            
            # Draw and save the current frame
            if len(rebuilt_data[0,:,0,0]) == 1:
                self.draw_obj_1(x_grid, y_grid, temp, rebuilt_temp, path, i)
            
            elif len(rebuilt_data[0,:,0,0]) == 2:
                self.draw_obj_2(x_grid, y_grid, temp, front, rebuilt_temp, rebuilt_front, path, i)
            
            elif len(rebuilt_data[0,:,0,0]) == 3:
                self.draw_obj_3(x_grid, y_grid, temp, front, cure, rebuilt_temp, rebuilt_front, rebuilt_cure, path, i)
            
            elif len(rebuilt_data[0,:,0,0]) == 5:
                self.draw_obj_5(x_grid, y_grid, temp, render_data, path, i)
                
            elif len(rebuilt_data[0,:,0,0]) == 6:
                self.draw_obj_6(x_grid, y_grid, temp, render_data, path, i)
                
            elif len(rebuilt_data[0,:,0,0]) == 7:
                self.draw_obj_7(x_grid, y_grid, temp, front, render_data, path, i)
                
            elif len(rebuilt_data[0,:,0,0]) == 8:
                self.draw_obj_8(x_grid, y_grid, temp, front, cure, render_data, path, i)
        
if __name__ == '__main__':
    
    ##---------------------------------------------------------------------------------------------------------------------##
    # autoencoder_1 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 1, 20, False)
    # autoencoder_1.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64")
    # autoencoder_2 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 2, 20, False)
    # autoencoder_2.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-1")
    # autoencoder_3 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 3, 20, False)
    # autoencoder_3.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-2")
    # autoencoder_4 = Autoencoder(1.0e-3, 1.0, 360, 40, 12, 12, 64, 1, 20, False)
    # autoencoder_4.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-12-12_64")
    # autoencoder_5 = Autoencoder(1.0e-3, 1.0, 360, 40, 12, 12, 64, 2, 20, False)
    # autoencoder_5.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-12-12_64_aux-1")
    # autoencoder_6 = Autoencoder(1.0e-3, 1.0, 360, 40, 12, 12, 64, 3, 20, False)
    # autoencoder_6.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-12-12_64_aux-2")
    # autoencoder_7 = Autoencoder(1.0e-3, 1.0, 360, 40, 12, 16, 64, 1, 20, False)
    # autoencoder_7.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-12-16_64")
    # autoencoder_8 = Autoencoder(1.0e-3, 1.0, 360, 40, 12, 16, 64, 2, 20, False)
    # autoencoder_8.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-12-16_64_aux-1")
    # autoencoder_9 = Autoencoder(1.0e-3, 1.0, 360, 40, 12, 16, 64, 3, 20, False)
    # autoencoder_9.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-12-16_64_aux-2")
    
    # with open("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64/test_frames", 'rb') as file:
    #     test_frames = pickle.load(file)
    # test_frames = test_frames['data']['Test_frames']
    
    # temp_error_1 = autoencoder_1.get_temp_error(test_frames)
    # temp_error_2 = autoencoder_2.get_temp_error(test_frames)
    # temp_error_3 = autoencoder_3.get_temp_error(test_frames)
    # temp_error_4 = autoencoder_4.get_temp_error(test_frames)
    # temp_error_5 = autoencoder_5.get_temp_error(test_frames)
    # temp_error_6 = autoencoder_6.get_temp_error(test_frames)
    # temp_error_7 = autoencoder_7.get_temp_error(test_frames)
    # temp_error_8 = autoencoder_8.get_temp_error(test_frames)
    # temp_error_9 = autoencoder_9.get_temp_error(test_frames)
    
    # temp_error_1 = [temp_error_1, temp_error_4, temp_error_7]
    # temp_error_2 = [temp_error_2, temp_error_5, temp_error_8]
    # temp_error_3 = [temp_error_3, temp_error_6, temp_error_9]
    # fig,ax = plt.subplots()
    # index = np.arange(3)
    # bar_width=0.3
    # opacity=0.60
    # rects_1=plt.bar(index, temp_error_1, bar_width, alpha=opacity, color='r', label='Default', edgecolor='k')
    # rects_2=plt.bar(index + bar_width, temp_error_2, bar_width, alpha=opacity, color='g', label='Aux 1', edgecolor='k')
    # rects_2=plt.bar(index + 2.0*bar_width, temp_error_3, bar_width, alpha=opacity, color='b', label='Aux 2', edgecolor='k')
    # plt.xticks(index+bar_width, ('1-8-16','1-12-12','1-12-16'), fontsize='large')
    # plt.yticks(fontsize='large')
    # plt.ylabel('MSE [%]',fontsize='large')
    # plt.xlabel('Encoder Filter Count',fontsize='large')
    # plt.legend(fontsize='large',loc='upper left')
    # plt.title("Temperature Field Reconstruction",fontsize='xx-large')
    # plt.gcf().set_size_inches(8.5, 5.5)
    # save_file = "validation/DCPD_GC2_Autoencoder/0%_Cropped/temp_reconstruction.png"
    # plt.savefig(save_file, dpi = 500)
    # plt.close()
    
    ##---------------------------------------------------------------------------------------------------------------------##
    path = "results/Auto_2"
    
    autoencoder = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 128, 20, 3, 3)
    autoencoder.load('results/f8-16_bn128_ob3')
    
    with open(path+"/output", 'rb') as file:
        load_file = pickle.load(file)
    autoencoder.save_temp_buffer = load_file['temp_array']
    autoencoder.save_cure_buffer = load_file['cure_array']
    
    autoencoder.render('results/Auto_1')
    
    ##---------------------------------------------------------------------------------------------------------------------##
    # autoencoder_1 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 1, 20, False)
    # autoencoder_1.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_def")
    # autoencoder_2 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 2, 20, False)
    # autoencoder_2.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-1")
    # autoencoder_3 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 3, 20, False)
    # autoencoder_3.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-2")
    # autoencoder_4 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 2, 20, False)
    # autoencoder_4.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-1_retrain-def")
    # autoencoder_5 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 3, 20, False)
    # autoencoder_5.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-2_retrain-def")
    # autoencoder_6 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 3, 20, False)
    # autoencoder_6.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-2_retrain-aux-1")
    
    # with open("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-2/test_frames", 'rb') as file:
    #     test_frames = pickle.load(file)
    # test_frames = test_frames['data']['Test_frames']
    
    # temp_error_1 = autoencoder_1.get_temp_error(test_frames)
    # temp_error_2 = autoencoder_2.get_temp_error(test_frames)
    # temp_error_3 = autoencoder_3.get_temp_error(test_frames)
    # temp_error_4 = autoencoder_4.get_temp_error(test_frames)
    # temp_error_5 = autoencoder_5.get_temp_error(test_frames)
    # temp_error_6 = autoencoder_6.get_temp_error(test_frames)
    
    # temp_error_1 = [temp_error_1, temp_error_4, temp_error_5]
    # temp_error_2 = [temp_error_2, temp_error_6]
    # temp_error_3 = [temp_error_3]
    # fig,ax = plt.subplots()
    # bar_width=0.3
    # opacity=0.60
    # rects_1=plt.bar(np.array([0,1,2]), temp_error_1, bar_width, alpha=opacity, color='r', label='Default', edgecolor='k')
    # rects_2=plt.bar(np.array([1.3,2.3]), temp_error_2, bar_width, alpha=opacity, color='g', label='Aux-1', edgecolor='k')
    # rects_2=plt.bar(np.array([2.6]), temp_error_3, bar_width, alpha=opacity, color='b', label='Aux-2', edgecolor='k')
    # plt.xticks(np.array([0,1.15,2.3]), ('Default','Aux-1','Aux-2'), fontsize='large')
    # plt.yticks(fontsize='large')
    # plt.ylabel('MSE [%]',fontsize='large')
    # plt.xlabel('Encoder Type (1-8-16_64)',fontsize='large')
    # plt.legend(fontsize='large',loc='upper right',title='Objective Fnc')
    # plt.title("Retrained Temperature Field Reconstruction",fontsize='xx-large')
    # plt.gcf().set_size_inches(8.5, 5.5)
    # save_file = "validation/DCPD_GC2_Autoencoder/0%_Cropped/retrain.png"
    # plt.savefig(save_file, dpi = 500)
    # plt.close()