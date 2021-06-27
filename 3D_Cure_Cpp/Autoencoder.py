# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
from CNN import Model as cnn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class Autoencoder:
    
    # OBJECTIVE FNC 1: Target temperature field
    # OBJECTIVE FNC 2: Target temperature field, and blurred front location
    # OBJECTIVE FNC 3: Target temperature field, blurred front location, and cure field
    # OBJECTIVE FNC 5: Target quantized temperature field
    def __init__(self, alpha, decay, x_dim_input, y_dim_input, bottleneck, buffer_size, num_output_layers, objective_fnc, kernal_size):
        
        # Initialize model
        self.model = cnn(x_dim_input, y_dim_input, bottleneck, num_output_layers, kernal_size)
            
        # Initialize loss criterion, and optimizer
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
        self.kernal_size = kernal_size
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
    # @return the training curve of the loaded autoencoder
    def load(self, path):
        
        # Copy NN at path to current module
        print("\nLoading: " + path)
        if not os.path.isdir(path):
            raise RuntimeError("Could not find " + path)
        else:
            with open(path+"/output", 'rb') as file:
                loaded_data = pickle.load(file)

            # Load hyperparameters
            self.x_dim = loaded_data['x_dim']
            self.y_dim = loaded_data['y_dim']
            self.bottleneck = loaded_data['bottleneck']
            self.num_output_layers = loaded_data['num_output_layers']
            self.kernal_size = loaded_data['kernal_size']
            self.objective_fnc = loaded_data['objective_fnc']
            self.buffer_size = loaded_data['buffer_size']
            self.save_temp_buffer = loaded_data['temp_array']
            self.save_cure_buffer = loaded_data['cure_array']
            
            # Load parameters
            self.model = cnn.NN(self.x_dim, self.y_dim, self.bottleneck, self.num_output_layers, self.kernal_size)
            loaded_model = loaded_data['autoencoder']
            self.model.load_state_dict(loaded_model.state_dict())
            
            # Load model onto GPU
            self.device = self.get_device()
            self.model.to(self.device)
            print("\nDevice(")
            print("  " + self.device)
            print(")\n")
            print(self.model) 
            
        return loaded_data['training_curve']
        
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
    
    # Forward propagates the temperature through the autoencoder and collects the feature maps at each layer
    # @param the temperature field that informs data reconstruction
    # @return the reconstructed data as np array
    # @return The first feature map as np array
    # @return The second feature map as np array
    # @return The third feature map as np array
    def forward_features(self, temp):
        
        # Convert frame to proper data type
        with torch.no_grad():
            temp = torch.tensor(temp)
            temp = temp.reshape(1,1,temp.shape[0],temp.shape[1]).float()
            temp = temp.to(self.device)
            rebuilt_data, features_1, features_2, features_3 = self.model.forward_features(temp)
            
            # convert encoded frame to proper data type
            rebuilt_data = rebuilt_data.to('cpu').squeeze().numpy()
            features_1 = features_1.to('cpu').squeeze().numpy()
            features_2 = features_2.to('cpu').squeeze().numpy()
            features_3 = features_3.to('cpu').squeeze().numpy()
        
        # Return the encoded frame of the proper data type
        return rebuilt_data, features_1, features_2, features_3
       
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
    def get_front_location(self, cure, blur_half_range=0.04):
        
        # Solve for cure front
        front_location = np.concatenate(((abs(np.diff(cure,axis=0))) > 0.25, np.zeros((1,self.y_dim))))
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
    # @return weight tensor
    def get_target(self, temp, cure):
        
        # Convert temperature frame to proper data form
        with torch.no_grad():
            temp = torch.tensor(temp)
            temp = temp.reshape(1,1,temp.shape[0],temp.shape[1]).float()
            weights = self.get_front_location(cure,blur_half_range=0.10)+0.25
            weights = weights / torch.mean(weights)
            
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
        
        return target, weights.squeeze().to(self.device)
    
    # Calcualtes the loss for autoencoder learning
    # @param Autoencoder rebuilt differentiable data
    # @param Target of objective function
    # @return Differentialable training loss
    def get_loss(self, rebuilt_data, target, weights):
        
        # Get rebuilt loss
        if self.objective_fnc == 8:
            curr_loss = torch.mean(weights * (rebuilt_data - target)**2.0)
        
        elif self.objective_fnc == 7:
            curr_loss = torch.mean(weights * (rebuilt_data[0,0:7,:,:] - target[0,0:7,:,:])**2.0)
            
        elif self.objective_fnc == 6:
            curr_loss = torch.mean(weights * (rebuilt_data[0,0:6,:,:] - target[0,0:6,:,:])**2.0)
        
        elif self.objective_fnc == 5:
            curr_loss = torch.mean(weights * (rebuilt_data[0,0:5,:,:] - target[0,0:5,:,:])**2.0)
            
        elif self.objective_fnc == 3:
            curr_loss = torch.mean(weights * (rebuilt_data[0,0:3,:,:] - target[0,0:3,:,:])**2.0)
            
        elif self.objective_fnc == 2:
            curr_loss = torch.mean(weights * (rebuilt_data[0,0:2,:,:] - target[0,0:2,:,:])**2.0)
            
        elif self.objective_fnc == 1:
            curr_loss = torch.mean(weights * (rebuilt_data[0,0,:,:] - target[0,0,:,:])**2.0)
            
        return curr_loss
        
    # Updates the autoencoder
    # @param temperature field to be added to training buffer
    # @param cure field to be added to training buffer
    # @return average epoch training loss or -1 if no optimization epoch occured
    def learn(self, temp, cure, weighted=False):
        
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
                target, weights = self.get_target(curr_temp, curr_cure)
                        
                # Format and forward propagate temperature data
                curr_temp = torch.tensor(curr_temp)
                curr_temp = curr_temp.reshape(1,1,curr_temp.shape[0],curr_temp.shape[1]).float()
                curr_temp = curr_temp.to(self.device)
                rebuilt_data = self.model.forward(curr_temp)
        
                # Get the loss
                if weighted:
                    curr_loss = self.get_loss(rebuilt_data, target, weights)
                else:
                    curr_loss = self.get_loss(rebuilt_data, target, 1.0)
        
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
                temp = torch.tensor(temp_array[i,:,:])
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
        print("\nSaving autoencoder results...")

        # Store data to dictionary
        data = {
            'x_dim' : self.x_dim.to('cpu'), 
            'y_dim' : self.y_dim.to('cpu'), 
            'bottleneck' : self.bottleneck.to('cpu'), 
            'num_output_layers' : self.num_output_layers.to('cpu'), 
            'objective_fnc' : self.objective_fnc.to('cpu'), 
            'kernal_size' : self.kernal_size.to('cpu'),
            'buffer_size' : self.buffer_size.to('cpu'), 
            'training_curve' : np.array(training_curve).to('cpu'),
            'temp_array' : self.save_temp_buffer.to('cpu'),
            'cure_array' : self.save_cure_buffer.to('cpu'),
            'autoencoder' : self.model.to('cpu'),
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
               
    # Plots each layer's feature maps at a given input temperautre
    # @param temperature field over which to generate feature maps
    # @param path to which feature maps are saved
    def plot_feature_maps(self, temp, path):
        # Generate feature maps
        _, features_1, features_2, features_3 = self.forward_features(temp)
        
        # Create grid spaces over which to render
        temp_x_grid, temp_y_grid = np.meshgrid(np.linspace(0,1,len(temp)), np.linspace(0,1,len(temp[0])))
        features_1_x_grid, features_1_y_grid = np.meshgrid(np.linspace(0,1,len(features_1[0,:,0])), np.linspace(0,1,len(features_1[0,0,:])))
        features_2_x_grid, features_2_y_grid = np.meshgrid(np.linspace(0,1,len(features_2[0,:,0])), np.linspace(0,1,len(features_2[0,0,:])))
        features_3_x_grid, features_3_y_grid = np.meshgrid(np.linspace(0,1,len(features_3[0,:,0])), np.linspace(0,1,len(features_3[0,0,:])))
        
        # Make input layer map
        plt.cla()
        plt.clf()
        fig, ax0 = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(8,2.6667)
        
        # Plot input
        ax0.pcolormesh(temp_x_grid, temp_y_grid, np.transpose(temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax0.axes.xaxis.set_visible(False)
        ax0.axes.yaxis.set_visible(False)
        ax0.set_aspect(0.25, adjustable='box')
        
        # Set title and save
        plt.suptitle('Input Image',fontsize='xx-large')
        plt.savefig(path+'/input_image.png', dpi=100)
        plt.close()
        
        # Make first features map figure
        plt.cla()
        plt.clf()
        fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True)
        fig.set_size_inches(16,2.6667)
        
        # Plot fm1
        ax0.pcolormesh(features_1_x_grid, features_1_y_grid, np.transpose(features_1[0,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax0.axes.xaxis.set_visible(False)
        ax0.axes.yaxis.set_visible(False)
        ax0.set_aspect(0.25, adjustable='box')
        ax0.set_title('Feature Map 1',fontsize='x-large')
        
        # Plot fm2
        ax1.pcolormesh(features_1_x_grid, features_1_y_grid, np.transpose(features_1[1,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        ax1.set_aspect(0.25, adjustable='box')
        ax1.set_title('Feature Map 2',fontsize='x-large')
        
        # Set title and save
        plt.suptitle('First Convolutional Layer',fontsize='xx-large')
        plt.savefig(path+'/layer_1_feature_maps.png', dpi=100)
        plt.close()
        
        # Make second features map figure
        plt.cla()
        plt.clf()
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, constrained_layout=True)
        fig.set_size_inches(16,5.3333)
        
        # Plot fm1
        ax0.pcolormesh(features_2_x_grid, features_2_y_grid, np.transpose(features_2[0,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax0.axes.xaxis.set_visible(False)
        ax0.axes.yaxis.set_visible(False)
        ax0.set_aspect(0.25, adjustable='box')
        ax0.set_title('Feature Map 1',fontsize='x-large')
        
        # Plot fm2
        ax1.pcolormesh(features_2_x_grid, features_2_y_grid, np.transpose(features_2[1,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        ax1.set_aspect(0.25, adjustable='box')
        ax1.set_title('Feature Map 2',fontsize='x-large')
        
        # Plot fm3
        ax2.pcolormesh(features_2_x_grid, features_2_y_grid, np.transpose(features_2[2,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        ax2.set_aspect(0.25, adjustable='box')
        ax2.set_title('Feature Map 3',fontsize='x-large')
        
        # Plot fm4
        ax3.pcolormesh(features_2_x_grid, features_2_y_grid, np.transpose(features_2[3,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax3.axes.xaxis.set_visible(False)
        ax3.axes.yaxis.set_visible(False)
        ax3.set_aspect(0.25, adjustable='box')
        ax3.set_title('Feature Map 4',fontsize='x-large')
        
        # Set title and save
        plt.suptitle('Second Convolutional Layer',fontsize='xx-large')
        plt.savefig(path+'/layer_2_feature_maps.png', dpi=100)
        plt.close()
        
        # Make third features map figure
        plt.cla()
        plt.clf()
        fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7), (ax8, ax9), (ax10, ax11), (ax12, ax13), (ax14, ax15)) = plt.subplots(8, 2, constrained_layout=True)
        fig.set_size_inches(16,21.3333)
        
        # Plot fm1
        ax0.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[0,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax0.axes.xaxis.set_visible(False)
        ax0.axes.yaxis.set_visible(False)
        ax0.set_aspect(0.25, adjustable='box')
        ax0.set_title('Feature Map 1',fontsize='x-large')
        
        # Plot fm2
        ax1.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[1,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        ax1.set_aspect(0.25, adjustable='box')
        ax1.set_title('Feature Map 2',fontsize='x-large')
        
        # Plot fm3
        ax2.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[2,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax2.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        ax2.set_aspect(0.25, adjustable='box')
        ax2.set_title('Feature Map 3',fontsize='x-large')
        
        # Plot fm4
        ax3.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[3,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax3.axes.xaxis.set_visible(False)
        ax3.axes.yaxis.set_visible(False)
        ax3.set_aspect(0.25, adjustable='box')
        ax3.set_title('Feature Map 4',fontsize='x-large')
        
        # Plot fm5
        ax4.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[4,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax4.axes.xaxis.set_visible(False)
        ax4.axes.yaxis.set_visible(False)
        ax4.set_aspect(0.25, adjustable='box')
        ax4.set_title('Feature Map 5',fontsize='x-large')
        
        # Plot fm6
        ax5.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[5,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax5.axes.xaxis.set_visible(False)
        ax5.axes.yaxis.set_visible(False)
        ax5.set_aspect(0.25, adjustable='box')
        ax5.set_title('Feature Map 6',fontsize='x-large')
        
        # Plot fm7
        ax6.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[6,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax6.axes.xaxis.set_visible(False)
        ax6.axes.yaxis.set_visible(False)
        ax6.set_aspect(0.25, adjustable='box')
        ax6.set_title('Feature Map 7',fontsize='x-large')
        
        # Plot fm8
        ax7.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[7,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax7.axes.xaxis.set_visible(False)
        ax7.axes.yaxis.set_visible(False)
        ax7.set_aspect(0.25, adjustable='box')
        ax7.set_title('Feature Map 8',fontsize='x-large')
        
        # Plot fm9
        ax8.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[8,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax8.axes.xaxis.set_visible(False)
        ax8.axes.yaxis.set_visible(False)
        ax8.set_aspect(0.25, adjustable='box')
        ax8.set_title('Feature Map 9',fontsize='x-large')
        
        # Plot fm10
        ax9.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[9,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax9.axes.xaxis.set_visible(False)
        ax9.axes.yaxis.set_visible(False)
        ax9.set_aspect(0.25, adjustable='box')
        ax9.set_title('Feature Map 10',fontsize='x-large')
        
        # Plot fm11
        ax10.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[10,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax10.axes.xaxis.set_visible(False)
        ax10.axes.yaxis.set_visible(False)
        ax10.set_aspect(0.25, adjustable='box')
        ax10.set_title('Feature Map 11',fontsize='x-large')
        
        # Plot fm12
        ax11.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[11,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax11.axes.xaxis.set_visible(False)
        ax11.axes.yaxis.set_visible(False)
        ax11.set_aspect(0.25, adjustable='box')
        ax11.set_title('Feature Map 12',fontsize='x-large')
        
        # Plot fm13
        ax12.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[12,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax12.axes.xaxis.set_visible(False)
        ax12.axes.yaxis.set_visible(False)
        ax12.set_aspect(0.25, adjustable='box')
        ax12.set_title('Feature Map 13',fontsize='x-large')
        
        # Plot fm14
        ax13.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[13,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax13.axes.xaxis.set_visible(False)
        ax13.axes.yaxis.set_visible(False)
        ax13.set_aspect(0.25, adjustable='box')
        ax13.set_title('Feature Map 14',fontsize='x-large')
        
        # Plot fm15
        ax14.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[14,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax14.axes.xaxis.set_visible(False)
        ax14.axes.yaxis.set_visible(False)
        ax14.set_aspect(0.25, adjustable='box')
        ax14.set_title('Feature Map 15',fontsize='x-large')
        
        # Plot fm16
        ax15.pcolormesh(features_3_x_grid, features_3_y_grid, np.transpose(features_3[15,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
        ax15.axes.xaxis.set_visible(False)
        ax15.axes.yaxis.set_visible(False)
        ax15.set_aspect(0.25, adjustable='box')
        ax15.set_title('Feature Map 16',fontsize='x-large')
        
        # Set title and save
        plt.suptitle('Third Convolutional Layer',fontsize='xx-large')
        plt.savefig(path+'/layer_3_feature_maps.png', dpi=100)
        plt.close()