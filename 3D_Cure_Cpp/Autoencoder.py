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
    
    def __init__(self, alpha, decay, x_dim_input, y_dim_input, num_filter_1, num_filter_2, bottleneck, num_output_layers, frame_buffer_size, load_previous):
        
        # Initialize model
        self.model = Autoencoder_NN.NN(x_dim_input, y_dim_input, num_filter_1, num_filter_2, bottleneck, num_output_layers);
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
        self.x_dim = x_dim_input
        self.y_dim = y_dim_input
        self.out_size = bottleneck
        self.num_output_layers = num_output_layers
        
        # Memory for MSE loss
        self.tot_MSE_loss = 0.0
        
        # Initialize frame buffer
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = []
        self.cure_buffer = []
        
        # Test frames
        self.test_frame_buffer = []
        self.test_cure_buffer = []
        
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
        
    def load(self, path):
        # Copy NN at path to current module
        print("\nLoading: " + path + "\n")
        if not os.path.isdir(path):
            print("Could not find" + path + "\n")
        else:
            with open(path+"/output", 'rb') as file:
                load_data = pickle.load(file)
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
       
    def encode(self, frame, cure):
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
    
    def update(self, frame, cure):
        
        # Store the current frame
        self.frame_buffer.append(np.array(frame))
        if len(self.test_frame_buffer) < self.frame_buffer_size:
            self.test_frame_buffer.append(np.array(frame))
        
        # Store the current cure
        if self.num_output_layers >= 3:
            self.cure_buffer.append(np.array(cure))
            if len(self.test_cure_buffer) < self.frame_buffer_size:
                self.test_cure_buffer.append(np.array(cure))
            
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
                    if self.num_output_layers >= 2:
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
                        
                    # Combine and format target data
                    if self.num_output_layers == 3:
                        cure = torch.tensor(self.cure_buffer[rand_indcies[i]])
                        cure = cure.reshape(1,1,cure.shape[0],cure.shape[1]).float()
                        target = torch.cat((frame, front_dist, cure), 1)
                        target = target.to(self.device)
                        
                    elif self.num_output_layers == 2:
                        target = torch.cat((frame, front_dist), 1)
                        target = target.to(self.device)
                        
                    elif self.num_output_layers == 1:
                        target = frame
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
            self.cure_buffer = []
        
        
        return self.tot_MSE_loss
    
    def display_and_save(self, MSE_loss):
        print("Saving autoencoder results...")

        data = {
            'MSE_loss' : np.array(MSE_loss),
            'Test_frames' : self.test_frame_buffer,
            'Test_cure' : self.test_cure_buffer,
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
    
    def get_temp_error(self, frames):
        # Step through all frames and average the temperature field reconstruction error
        print("Getting temperature field reconstruction error...")
        temp_error = 0.0
        with torch.no_grad():
            for curr_step in range(len(frames)):
                # Format frame data
                frame = torch.tensor(frames[curr_step])
                frame = frame.reshape(1,1,frame.shape[0],frame.shape[1]).float()
            
                # Forward propogate the frame through the autoencoder
                frame = frame.to(self.device)
                rebuilt_frame = self.model.forward(frame)
                
                # Calcualte temperature reconstruction error
                temp = frame[0,0,:,:].to('cpu').numpy().squeeze()
                rebuilt_temp = rebuilt_frame[0,0,:,:].to('cpu').numpy().squeeze()
                temp_error = temp_error + np.mean((rebuilt_temp-temp)**2.0)*100.0
                
        # Get average reconstruction error
        temp_error = temp_error / len(frames)
        return temp_error
            
    
    def render(self, frames, cures):
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
            fig, ((ax0, ax2), (ax1, ax3), (ax5, ax4)) = plt.subplots(3, 2)
            fig.set_size_inches(16,8)
            ax5.axis('off')
            
            # Plot frame
            c0 = ax0.pcolormesh(x_grid, y_grid, np.transpose(frames[curr_step]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
            cbar0 = fig.colorbar(c0, ax=ax0)
            cbar0.set_label('Temperature [-]',labelpad=20,fontsize='large')
            cbar0.ax.tick_params(labelsize=12)
            ax0.tick_params(axis='x',labelsize=12)
            ax0.tick_params(axis='y',labelsize=12)
            ax0.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax0.set_title('True Temperature Field',fontsize='x-large')
            
            # Rebuilt temperature
            c1 = ax1.pcolormesh(x_grid, y_grid, np.transpose(self.forward(frames[curr_step])[0,:,:]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
            cbar1 = fig.colorbar(c1, ax=ax1)
            cbar1.set_label('Temperature [-]',labelpad=20,fontsize='large')
            cbar1.ax.tick_params(labelsize=12)
            ax1.tick_params(axis='x',labelsize=12)
            ax1.tick_params(axis='y',labelsize=12)
            ax1.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax1.set_title('Rebuilt Temperature Field',fontsize='x-large')
            
            # Cure frame
            c2 = ax2.pcolormesh(x_grid, y_grid, np.transpose(cures[curr_step]), shading='gouraud', cmap='YlOrRd', vmin=0.0, vmax=1.0)
            cbar2 = fig.colorbar(c2, ax=ax2)
            cbar2.set_label('Degree Cure [-]',labelpad=20,fontsize='large')
            cbar2.ax.tick_params(labelsize=12)
            ax2.tick_params(axis='x',labelsize=12)
            ax2.tick_params(axis='y',labelsize=12)
            ax2.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax2.set_title('True Cure Field',fontsize='x-large')
            
            # Rebuilt cure degree
            c3 = ax3.pcolormesh(x_grid, y_grid, np.transpose(self.forward(frames[curr_step])[2,:,:]), shading='gouraud', cmap='YlOrRd', vmin=0.0, vmax=1.0)
            cbar3 = fig.colorbar(c3, ax=ax3)
            cbar3.set_label('Degree Cure [-]',labelpad=20,fontsize='large')
            cbar3.ax.tick_params(labelsize=12)
            ax3.tick_params(axis='x',labelsize=12)
            ax3.tick_params(axis='y',labelsize=12)
            ax3.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax3.set_title('Rebuilt Cure Field',fontsize='x-large')
            
            # Rebuilt front location
            c4 = ax4.pcolormesh(x_grid, y_grid, np.transpose(self.forward(frames[curr_step])[1,:,:]), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
            cbar4 = fig.colorbar(c4, ax=ax4)
            cbar4.set_label('Dense Front [-]',labelpad=20,fontsize='large')
            cbar4.ax.tick_params(labelsize=12)
            ax4.tick_params(axis='x',labelsize=12)
            ax4.tick_params(axis='y',labelsize=12)
            ax4.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax4.set_title('Rebuilt Front Location',fontsize='x-large')
            
            # Set title and save
            plt.savefig(path+"/"+str(curr_step).zfill(4)+'.png', dpi=100)
            plt.close()
                
        
if __name__ == '__main__':
    # autoencoder_1 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 1, 20, False)
    # autoencoder_1.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64")
    # autoencoder_2 = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 2, 20, False)
    # autoencoder_2.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_obj")
    # autoencoder_3 = Autoencoder(1.0e-3, 1.0, 360, 40, 12, 12, 64, 1, 20, False)
    # autoencoder_3.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-12-12_64")
    # autoencoder_4 = Autoencoder(1.0e-3, 1.0, 360, 40, 12, 12, 64, 2, 20, False)
    # autoencoder_4.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-12-12_64_obj")
    # autoencoder_5 = Autoencoder(1.0e-3, 1.0, 360, 40, 12, 16, 64, 1, 20, False)
    # autoencoder_5.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-12-16_64")
    # autoencoder_6 = Autoencoder(1.0e-3, 1.0, 360, 40, 12, 16, 64, 2, 20, False)
    # autoencoder_6.load("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-12-16_64_obj")
    
    # with open("validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64/test_frames", 'rb') as file:
    #     test_frames = pickle.load(file)
    # test_frames = test_frames['data']['Test_frames']
    
    # temp_error_1 = autoencoder_1.get_temp_error(test_frames)
    # temp_error_2 = autoencoder_2.get_temp_error(test_frames)
    # temp_error_3 = autoencoder_3.get_temp_error(test_frames)
    # temp_error_4 = autoencoder_4.get_temp_error(test_frames)
    # temp_error_5 = autoencoder_5.get_temp_error(test_frames)
    # temp_error_6 = autoencoder_6.get_temp_error(test_frames)
    
    # temp_error_1 = [temp_error_1, temp_error_3, temp_error_5]
    # temp_error_2 = [temp_error_2, temp_error_4, temp_error_6]
    # fig,ax = plt.subplots()
    # index = np.arange(3)
    # bar_width=0.3
    # opacity=0.60
    # rects_1=plt.bar(index, temp_error_1, bar_width, alpha=opacity, color='r', label='Default')
    # rects_2=plt.bar(index + bar_width, temp_error_2, bar_width, alpha=opacity, color='b', label='Aux 1')
    # plt.xticks(index+0.5*bar_width, ('1-8-16','1-12-12','1-12-16'), fontsize='large')
    # plt.yticks(fontsize='large')
    # plt.ylabel('MSE [%]',fontsize='large')
    # plt.xlabel('Encoder Filter Count',fontsize='large')
    # plt.legend(fontsize='large',loc='upper right')
    # plt.title("Temperature Field Reconstruction",fontsize='xx-large')
    # plt.gcf().set_size_inches(8.5, 5.5)
    # save_file = "validation/DCPD_GC2_Autoencoder/0%_Cropped/temp_reconstruction.png"
    # plt.savefig(save_file, dpi = 500)
    # plt.close()
    
    autoencoder = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 3, 20, False)
    autoencoder.load("results/Auto_1")
    
    with open("results/Auto_1/test_frames", 'rb') as file:
        data = pickle.load(file)
    test_frames = data['data']['Test_frames']
    test_cures = data['data']['Test_cure']
    
    autoencoder.render(test_frames, test_cures)