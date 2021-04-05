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
    
    def __init__(self, alpha, decay, x_dim_input, y_dim_input, num_filter_1, num_filter_2, bottleneck, num_output_layers, frame_buffer_size, objective_fnc):
        
        # Initialize model
        self.model = Autoencoder_NN.NN(x_dim_input, y_dim_input, num_filter_1, num_filter_2, bottleneck, num_output_layers);
            
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
        if num_output_layers > 3:
            raise RuntimeError('Number of output layers must be greater than 0 and less than 4.')
            
        # Memory for MSE loss
        self.tot_MSE_loss = 0.0
        
        # Objective fnc type
        self.objective_fnc = objective_fnc
        if objective_fnc > num_output_layers:
            raise RuntimeError('Objective function must be greater than 0 and less than or equal to the number of output layers.')
        
        # Initialize frame buffer
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = []
        self.cure_buffer = []
        
        # Test frames
        self.test_frame_buffer = []
        self.test_cure_buffer = []
        
    def load(self, path):
        # Copy NN at path to current module
        print("\nLoading: " + path)
        if not os.path.isdir(path):
            raise RuntimeError("Could not find " + path)
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
    
    def update(self, frame, cure):
        
        # Store the current frame
        self.frame_buffer.append(np.array(frame))
        if len(self.test_frame_buffer) < self.frame_buffer_size:
            self.test_frame_buffer.append(np.array(frame))
        
        # Store the current cure
        if self.objective_fnc >= 3:
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
                    if self.objective_fnc >= 2:
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
                    if self.objective_fnc == 3:
                        cure = torch.tensor(self.cure_buffer[rand_indcies[i]])
                        cure = cure.reshape(1,1,cure.shape[0],cure.shape[1]).float()
                        target = torch.cat((frame, front_dist, cure), 1)
                        target = target.to(self.device)
                        
                    elif self.objective_fnc == 2:
                        target = torch.cat((frame, front_dist), 1)
                        target = target.to(self.device)
                        
                    elif self.objective_fnc == 1:
                        target = frame
                        target = target.to(self.device)
                
                # Forward propogate the frame through the autoencoder
                frame = frame.to(self.device)
                rebuilt_frame = self.model.forward(frame)
            
                # Get rebuilt loss
                if self.objective_fnc == 3:
                    curr_MSE_loss = self.criterion_1(rebuilt_frame, target)
                elif self.objective_fnc == 2:
                    curr_MSE_loss = self.criterion_1(rebuilt_frame[0,0:2,:,:], target[0,0:2,:,:])
                elif self.objective_fnc == 1:
                    curr_MSE_loss = self.criterion_1(rebuilt_frame[0,0,:,:], target[0,0,:,:])
                        
                # Calculate loss and take optimization step and learning rate step
                self.optimizer.zero_grad()
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
            
    
    def render(self, frames, cures, path):
        print("Rendering...")
        x_grid, y_grid = np.meshgrid(np.linspace(0,1,self.x_dim), np.linspace(0,1,self.y_dim))
        
        # Find save paths
        path = path + "/video"
        if not os.path.isdir(path):
            os.mkdir(path)
        
        for curr_step in range(len(frames)):

            # Calculate front location
            with torch.no_grad():
                # Format frame data
                frame = torch.tensor(frames[curr_step])
                frame = frame.reshape(1,1,frame.shape[0],frame.shape[1]).float()
                
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
                front_dist=front_dist.numpy().squeeze()

            # Make fig for temperature, cure, and input
            plt.cla()
            plt.clf()
            fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2)
            fig.set_size_inches(16,8)
            
            # Plot frame
            c0 = ax0.pcolormesh(x_grid, y_grid, np.transpose(frames[curr_step]), shading='gouraud', cmap='jet', vmin=0.0, vmax=1.0)
            cbar0 = fig.colorbar(c0, ax=ax0)
            cbar0.set_label('Temperature [-]',labelpad=20,fontsize='large')
            cbar0.ax.tick_params(labelsize=12)
            ax0.tick_params(axis='x',labelsize=12)
            ax0.tick_params(axis='y',labelsize=12)
            ax0.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax0.set_title('True Temperature Field (Known)',fontsize='x-large')
            
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
            ax2.set_title('True Cure Field (Unknown)',fontsize='x-large')
            
            # Inferred cure degree
            c3 = ax3.pcolormesh(x_grid, y_grid, np.transpose(self.forward(frames[curr_step])[2,:,:]), shading='gouraud', cmap='YlOrRd', vmin=0.0, vmax=1.0)
            cbar3 = fig.colorbar(c3, ax=ax3)
            cbar3.set_label('Degree Cure [-]',labelpad=20,fontsize='large')
            cbar3.ax.tick_params(labelsize=12)
            ax3.tick_params(axis='x',labelsize=12)
            ax3.tick_params(axis='y',labelsize=12)
            ax3.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax3.set_title('Inferred Cure Field',fontsize='x-large')
            
            # Front frame
            c4 = ax4.pcolormesh(x_grid, y_grid, np.transpose(front_dist), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
            cbar4 = fig.colorbar(c4, ax=ax4)
            cbar4.set_label('Front Field [-]',labelpad=20,fontsize='large')
            cbar4.ax.tick_params(labelsize=12)
            ax4.tick_params(axis='x',labelsize=12)
            ax4.tick_params(axis='y',labelsize=12)
            ax4.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax4.set_title('True Front Location (Unknown)',fontsize='x-large')
            
            # Inferred front location
            c5 = ax5.pcolormesh(x_grid, y_grid, np.transpose(self.forward(frames[curr_step])[1,:,:]), shading='gouraud', cmap='binary', vmin=0.0, vmax=1.0)
            cbar5 = fig.colorbar(c5, ax=ax5)
            cbar5.set_label('Front Field [-]',labelpad=20,fontsize='large')
            cbar5.ax.tick_params(labelsize=12)
            ax5.tick_params(axis='x',labelsize=12)
            ax5.tick_params(axis='y',labelsize=12)
            ax5.set_aspect(self.y_dim/self.x_dim, adjustable='box')
            ax5.set_title('Inferred Front Location',fontsize='x-large')
            
            # Set title and save
            plt.suptitle("Autoencoder Performance for DCPD with GC2",fontsize='xx-large')
            plt.savefig(path+"/"+str(curr_step).zfill(4)+'.png', dpi=100)
            plt.close()
                
        
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
    path = "validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-2"
    
    autoencoder = Autoencoder(1.0e-3, 1.0, 360, 40, 8, 16, 64, 3, 20, False)
    autoencoder.load(path)
    
    with open(path+"/test_frames", 'rb') as file:
        data = pickle.load(file)
    test_frames = data['data']['Test_frames']
    test_cures = data['data']['Test_cure']
    
    autoencoder.render(test_frames, test_cures, path)
    
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