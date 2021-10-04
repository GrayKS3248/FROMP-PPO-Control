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
import pandas as pd

class Autoencoder:
    
    # Initializes an object of the autoencoder class. This class compresses input images to a linear 128 element latent representation 
    # via a compressed form of the AlexNet CNN. This class requires that input images and target images have the same dimensions.
    # @param Learning rate
    # @param Exponential learning rate decay
    # @param Path from which a previous model will be loaded
    # @param First dimension of the input and target images
    # @param Second dimension of the input and target images
    # @param Minimum value of input. Used for normalization
    # @param Maximum value of input. Used for normalization
    # @param The number of targets reconstructed by the autoencoder from the input image
    # @param Stdev of white noise added to the input image before convolution
    # @param Verbose
    def __init__(self, alpha, decay, load_path="", dim_1=0, dim_2=0, norm_min=0.0, norm_max=1.0, num_targets=0, noise_stdev=0.0, verbose=True):
        
        # Initialize hyperparameters
        self.alpha_zero = alpha
        self.alpha_decay = decay
        self.noise_stdev = noise_stdev
        if noise_stdev==0.0:
            self.noisy = False
        else:
            self.noisy = True
        
        # Initialize or load model
        if (load_path==""):
            
            # Store NN size parameters
            self.dim_1 = dim_1
            self.dim_2 = dim_2
            self.norm_min = norm_min
            self.norm_max = norm_max
            self.num_targets = num_targets
            
            # Initialize model
            self.model = cnn(np.max((self.dim_1, self.dim_2)), self.num_targets)
            
            # Initialize buffer for training curve and lr curve
            self.loss_curve = []
            self.lr_curve = []
            
            # Initialize variables to track snapshots during training
            self.batch_num = 0
            self.input_snapshots = [[], [], []]
            self.targets_snapshots = [[], [], []]
            self.outputs_snapshots = [[], [], []]
            self.loss_snapshots = [[], [], []]
            self.batch_num_snapshots = []
       
        else:
            self.load(load_path, verbose=verbose)
            
        # Initialize optimizer, scheduler, and criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha_zero)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.alpha_decay)
        self.criterion_BCE = torch.nn.BCELoss()
        self.criterion_MSE = torch.nn.MSELoss()
        
        # Load model onto GPU
        self.device = self.get_device()
        self.model.to(self.device)
        
        # Create laplacian filter
        self.laplacian = torch.nn.Conv2d(1,1,7,stride=1,padding=0)
        self.laplacian.weight = torch.nn.Parameter(torch.tensor([[[[0.0, 0.0, 0.0, 1.0/90.0, 0.0, 0.0, 0.0], 
                                                                   [0.0, 0.0, 0.0, -3.0/20.0, 0.0, 0.0, 0.0], 
                                                                   [0.0, 0.0, 0.0, 3.0/2.0, 0.0, 0.0, 0.0], 
                                                                   [1.0/90.0, -3.0/20.0, 3.0/2.0, -49.0/9.0, 3.0/2.0, -3.0/20.0, 1.0/90.0], 
                                                                   [0.0, 0.0, 0.0, 3.0/2.0, 0.0, 0.0, 0.0], 
                                                                   [0.0, 0.0, 0.0, -3.0/20.0, 0.0, 0.0, 0.0], 
                                                                   [0.0, 0.0, 0.0, 1.0/90.0, 0.0, 0.0, 0.0]]]]))
        self.laplacian.weight.requires_grad = False
        self.laplacian.bias = torch.nn.Parameter(torch.tensor([0.0]))
        self.laplacian.bias.requires_grad = False
        self.laplacian.to(self.device)
        
        # User readout
        if verbose:
            print("Device(")
            print("  " + self.device)
            print(")\n")
            print(self.model)
            print("\nAutoencoder Parameters(")
            print("  (Dimensions): " + str(dim_1) + "x" + str(dim_2))
            print("  (Num Targets): " + str(num_targets))
            print("  (Noise Stdev): {0:.3f}".format(noise_stdev))
            print(")\n")
        
    # Loads a saved autoencoder at path/output
    # @param Path from which the autoencoder will be loaded
    def load(self, load_path, verbose=True):
        
        # Copy NN at path to current module
        if verbose:
            print("Loading: " + load_path + "/output \n")
        if not os.path.isdir(load_path):
            raise RuntimeError("Could not find " + load_path)
        else:
            with open(load_path+"/output", 'rb') as file:
                loaded_data = pickle.load(file)

            # Load network size parameters
            self.dim_1 = loaded_data['dim_1']
            self.dim_2 = loaded_data['dim_2']
            self.norm_min = loaded_data['norm_min']
            self.norm_max = loaded_data['norm_max']
            self.num_targets = loaded_data['num_targets']
            
            # Load model parameters
            self.model = cnn(np.max((self.dim_1, self.dim_2)), self.num_targets)
            self.model.load_state_dict(loaded_data['model'].state_dict())
            
            # Initialize buffer for training curve and lr curve
            self.loss_curve = loaded_data['loss_curve']
            self.lr_curve = loaded_data['lr_curve']
            
            # Initialize snapshot variables
            self.batch_num = loaded_data['batch_num']
            self.input_snapshots = loaded_data['input_snapshots']
            self.targets_snapshots = loaded_data['targets_snapshots']
            self.outputs_snapshots = loaded_data['outputs_snapshots']
            self.batch_num_snapshots = loaded_data['batch_num_snapshots']
            self.loss_snapshots = loaded_data['loss_snapshots']
   
    # MSE of laplacian
    # @param The input image
    # @param The target image
    def criterion_laploss(self,input_image,target_image):
        input_laplacian = self.laplacian(input_image)
        target_laplacian = self.laplacian(target_image)
        input_laplacian = (input_laplacian - torch.min(input_laplacian).item()) / (torch.max(input_laplacian).item() - torch.min(input_laplacian).item())
        target_laplacian = (target_laplacian - torch.min(target_laplacian).item()) / (torch.max(target_laplacian).item() - torch.min(target_laplacian).item())
        loss = self.criterion_MSE(input_laplacian, target_laplacian)
        return loss
    
    # Gets the cpu or gpu on which to run
    # @return device code
    def get_device(self):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device
 
    # Converts any non square input image into a square input image via linear interpolation about the shortest axis
    # @param The input image to be converted
    # @return The squared input image
    def convert_to_square(self, input_image):
        
        # Extract dimensions
        input_image = np.array(input_image)
        largest_dim = max(input_image.shape)
        smallest_dim = min(input_image.shape)
        smallest_ax = np.argmin(input_image.shape)
        new_image = np.zeros((largest_dim, largest_dim))
        new_image_mask = np.zeros(smallest_dim)
        start_point_of_interpolant = 0
        
        # Extract information regarding the interpolation
        length_of_interpolant = largest_dim // (smallest_dim - 1)
        num_of_interpolants_one_longer = largest_dim - (length_of_interpolant * (smallest_dim - 1)) - 1
        spacing_on_longer_interpolants = (smallest_dim - 1) / num_of_interpolants_one_longer
        indicies_of_longer_interpolants = np.round(np.arange(0, smallest_dim - 1, spacing_on_longer_interpolants))
        
        # Interpolate
        for curr_interpolant_index in range(input_image.shape[smallest_ax]-1):
            if (indicies_of_longer_interpolants == curr_interpolant_index).any():
                length_of_curr_interpolant = length_of_interpolant+2
            else:
               length_of_curr_interpolant = length_of_interpolant+1
            curr_interpolant = np.transpose(np.linspace( input_image[:,curr_interpolant_index], input_image[:,curr_interpolant_index+1], length_of_curr_interpolant) )
            new_image[:,start_point_of_interpolant:start_point_of_interpolant+length_of_curr_interpolant] = curr_interpolant
            new_image_mask[curr_interpolant_index] = start_point_of_interpolant
            start_point_of_interpolant = start_point_of_interpolant + length_of_curr_interpolant-1
        
        # Complete last entry of image mask
        new_image_mask[-1] = largest_dim - 1
        
        return new_image, new_image_mask
    
    # Normalizes input image so that the min is 0 and max is 1
    # @param The input image to be normalized
    # @return The normalized input image
    def normalize(self, input_image):
        
        new_image = np.zeros(input_image.shape)
        
        # Static range normalization
        if self.norm_min != self.norm_max:
            new_image = (input_image - self.norm_min) / (self.norm_max - self.norm_min)
        else:
            new_image = input_image / self.norm_min
        
        # Normalized range assurance
        new_image[new_image>1.0] = 1.0
        new_image[new_image<0.0] = 0.0
        
        return new_image
    
    # Denormalizes output image so that the mean and stdev match the original image
    # @param The output image to be denormalized
    # @return The denormalized output image
    def denormalize(self, output_image):
        new_image = output_image * (self.norm_max - self.norm_min) + self.norm_min
            
        return new_image
 
    # Converts any squared output image back to its original dimensions via linear regression along the shortest axis
    # @param The squared output image to be converted
    # @param Mask indicating the interpolant index at each point of square output image
    # @return The output image in its original dimensions
    def convert_to_orig_dim(self, output_image, sq_image_mask):
        
        # Create linear regression across each super sample and grab intercept values
        new_image = np.zeros((self.dim_1, self.dim_2))
        for i in range(len(sq_image_mask) - 1):
            x_coords = np.arange(sq_image_mask[i], sq_image_mask[i+1]+1) - sq_image_mask[i]
            y_coords = output_image[:,np.int(sq_image_mask[i]):np.int(sq_image_mask[i+1]+1)]
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            slope_intercept = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T,A)),A.T),y_coords.T)
            if i == 0:
                new_image[:, i] = slope_intercept[1]
            else:
                new_image[:, i] = 0.5 * (new_image[:, i] + slope_intercept[1])
            new_image[:,i+1] = slope_intercept[1] + slope_intercept[0] * x_coords[-1]
            
        return new_image   
 
    # Forward propagates an input image through the autoencoder
    # @param The input image to be propogated
    # @return The propogated output images
    def forward(self, input_image):
        
        # Convert input to proper data type
        with torch.no_grad():
            
            # Square and normalize input
            sq_input_image, sq_image_mask = self.convert_to_square(input_image)
            norm_sq_input_image = self.normalize(sq_input_image)
                
            # Convert to tensor of correct dimensions
            norm_sq_input_image = torch.tensor(norm_sq_input_image)
            norm_sq_input_image = norm_sq_input_image.reshape(1,1,norm_sq_input_image.shape[0],norm_sq_input_image.shape[1]).float()
            norm_sq_input_image = norm_sq_input_image.to(self.device)
            
            # Forward through model
            norm_sq_output_image = self.model.forward(norm_sq_input_image).to('cpu').squeeze().numpy()
            
            # Return to original distribution and dimensions
            output_images = []
            for target in range(self.num_targets):
                if self.num_targets == 1:
                    sq_output_image = self.denormalize(norm_sq_output_image)
                else: 
                    sq_output_image = self.denormalize(norm_sq_output_image[target])
                output_images.append(self.convert_to_orig_dim(sq_output_image, sq_image_mask))
            output_images = np.array(output_images)
                
        # Return the encoded frame of the proper data type
        return output_images
       
    # Encodes a given input image
    # @param The input image to be encoded
    # @return The linear latent representation
    def encode(self, input_image):

        # Convert input to proper data type
        with torch.no_grad():
            
            # Square and normalize input
            sq_input_image, sq_image_mask = self.convert_to_square(input_image)
            norm_sq_input_image = self.normalize(sq_input_image)
            
            # Convert to tensor of correct dimensions
            norm_sq_input_image = torch.tensor(norm_sq_input_image)
            norm_sq_input_image = norm_sq_input_image.reshape(1,1,norm_sq_input_image.shape[0],norm_sq_input_image.shape[1]).float()
            norm_sq_input_image = norm_sq_input_image.to(self.device)
            
            # Forward through model
            latent_representation = self.model.encode(norm_sq_input_image).to('cpu').squeeze().numpy()
        
        # Return the encoded frame of the proper data type
        return latent_representation.tolist()
    
    # Generates a candidate image given a latent representation
    # @param The latent seed of generation
    # @return The generated image
    def generate(self, latent_seed):

        # Convert input to proper data type
        with torch.no_grad():
            
            # Convert to tensor of correct dimensions
            latent_seed = torch.tensor(latent_seed)
            latent_seed = latent_seed.reshape(1,latent_seed.shape[0]).float()
            latent_seed = latent_seed.to(self.device)
            
            # Forward through model
            norm_sq_output_image = self.model.generate(latent_seed).to('cpu').squeeze().numpy()
        
            # Return to original distribution and dimensions
            output_images = []
            _, sq_image_mask = self.convert_to_square(np.random.rand(self.dim_1,self.dim_2))
            for target in range(self.num_targets):
                if self.num_targets == 1:
                    sq_output_image = self.denormalize(norm_sq_output_image)
                else: 
                    sq_output_image = self.denormalize(norm_sq_output_image[target])
                output_images.append(self.convert_to_orig_dim(sq_output_image, sq_image_mask))
            output_images = np.array(output_images)
                
        # Return the encoded frame of the proper data type
        return output_images
        
    # Adds white noise to image based on initialization function noise stdev.
    # @param The input image to which noise is added
    # @return The noisy input image
    def add_noise(self, input_image):
        if self.noisy:
            norm_input_image = self.normalize(input_image)
            noisy_norm_input_image = norm_input_image + np.random.normal(0.0, self.noise_stdev, input_image.shape)
            noisy_norm_input_image[noisy_norm_input_image>1.0]=1.0
            noisy_norm_input_image[noisy_norm_input_image<0.0]=0.0
            noisy_image = self.denormalize(noisy_norm_input_image)
        else:
            noisy_image = input_image
            
        return noisy_image
    
    # Performs one epoch of stochastic gradient descent. ALL INPUT AND TARGET IMAGES MUST USE SCALE
    # @param Full batch of input images
    # @param Full batch of target images
    # @return Average training loss over entire batch
    def learn(self, input_batch, targets_batch, take_snapshot=False):
        
        # Snapshot mechanics
        if take_snapshot:
            loss_buffer = []
            input_buffer = []
            targets_buffer = []
        
        # Perform one step of stochastic gradient descent for each member of the batch
        avg_loss = 0.0
        rand_indcies = np.random.permutation(len(input_batch[:,0,0]))
        for i in range(len(input_batch[:,0,0])):
            
            # Get temperature and cure at random location from bacth
            input_image = input_batch[rand_indcies[i],:,:]
            targets = targets_batch[:,rand_indcies[i],:,:]
                    
            # Square, normalize, and add noise to input
            noisy_input_image = self.add_noise(input_image)
            noisy_sq_input_image, _ = self.convert_to_square(noisy_input_image)
            noisy_norm_sq_input_image = self.normalize(noisy_sq_input_image)
            
            # Convert to tensor of correct dimensions
            noisy_norm_sq_input_image = torch.tensor(noisy_norm_sq_input_image, requires_grad=True)
            noisy_norm_sq_input_image = noisy_norm_sq_input_image.reshape(1,1,noisy_norm_sq_input_image.shape[0],noisy_norm_sq_input_image.shape[1]).float()
            noisy_norm_sq_input_image = noisy_norm_sq_input_image.to(self.device)
            
            # Forward through model
            norm_sq_output_images = self.model.forward(noisy_norm_sq_input_image)
            
            # Square and normalize targets
            norm_sq_targets = np.zeros((self.num_targets, max(self.dim_1,self.dim_2), max(self.dim_1,self.dim_2)))
            for j in range(self.num_targets):
                sq_targets, _ = self.convert_to_square(targets[j,:,:])
                norm_sq_targets[j,:,:] = self.normalize(sq_targets)
    
            # Convert to tensor of correct dimensions
            norm_sq_targets = torch.tensor(norm_sq_targets, requires_grad=False)
            norm_sq_targets = norm_sq_targets.reshape(1,norm_sq_targets.shape[0],norm_sq_targets.shape[1],norm_sq_targets.shape[2]).float()
            norm_sq_targets = norm_sq_targets.to(self.device)
            
            # Take optimization step
            loss = self.criterion_BCE(norm_sq_output_images, norm_sq_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Sum the epoch's total loss
            avg_loss = avg_loss + loss.item()
            
            # Snapshot mechanics
            if take_snapshot:
                loss_buffer.append(loss.item())
                input_buffer.append(noisy_input_image)
                targets_buffer.append(targets)
            
        # Step the learning rate scheduler
        lr = self.lr_scheduler.get_last_lr()[0]
        self.lr_scheduler.step()

        # Append training data
        self.loss_curve.append(avg_loss / len(input_batch[:,0,0]))
        self.lr_curve.append(lr)

        # Snapshot mechanics
        self.batch_num = self.batch_num + 1
        if take_snapshot:
            index_of_min_loss = np.argmin(loss_buffer)
            index_of_avg_loss = np.argmin(abs(np.array(loss_buffer) - np.mean(loss_buffer)))
            index_of_max_loss = np.argmax(loss_buffer)
            self.input_snapshots[0].append(input_buffer[index_of_min_loss])
            self.input_snapshots[1].append(input_buffer[index_of_avg_loss])
            self.input_snapshots[2].append(input_buffer[index_of_max_loss])
            self.targets_snapshots[0].append(targets_buffer[index_of_min_loss])
            self.targets_snapshots[1].append(targets_buffer[index_of_avg_loss])
            self.targets_snapshots[2].append(targets_buffer[index_of_max_loss])
            self.outputs_snapshots[0].append(self.forward(input_buffer[index_of_min_loss]))
            self.outputs_snapshots[1].append(self.forward(input_buffer[index_of_avg_loss]))
            self.outputs_snapshots[2].append(self.forward(input_buffer[index_of_max_loss]))
            self.loss_snapshots[0].append(np.round(loss_buffer[index_of_min_loss],3))
            self.loss_snapshots[1].append(np.round(loss_buffer[index_of_avg_loss],3))
            self.loss_snapshots[2].append(np.round(loss_buffer[index_of_max_loss],3))
            self.batch_num_snapshots.append(self.batch_num)
        
        # Return the average RMS reconstruction error
        return avg_loss / len(input_batch[:,0,0]), lr
    
    # Saves the training data and trained model
    # @param Boolean flag indicating to draw loss curve and learning rate curve for all training
    # @param Boolean flag indicating to render snapshots taken during all training
    def save(self, draw=True, render=True):
        print("\nSaving autoencoder results...")

        # Store data to dictionary
        data = {
            "dim_1" : self.dim_1,
            "dim_2" : self.dim_2,
            "norm_min" : self.norm_min,
            "norm_max" : self.norm_max,
            "num_targets" : self.num_targets,
            "loss_curve" : self.loss_curve,
            "lr_curve" : self.lr_curve,
            "model" : self.model.to('cpu'),
            "batch_num" : self.batch_num,
            "input_snapshots" : self.input_snapshots,
            "targets_snapshots" : self.targets_snapshots,
            "outputs_snapshots" : self.outputs_snapshots,
            "batch_num_snapshots" : self.batch_num_snapshots,
            "loss_snapshots" : self.loss_snapshots,
        }
        self.model.to(self.device)

        # Find save paths
        curr_dir_num = 1
        path = "../results/AE_" + str(curr_dir_num)
        done = False
        while not done:
            if not os.path.isdir(path):
                os.mkdir(path)
                done = True
            else:
                curr_dir_num = curr_dir_num + 1
                path = "../results/AE_" + str(curr_dir_num)

        # Pickle all important outputs
        save_file = path + "/output"
        with open(save_file, 'wb') as file:
            pickle.dump(data, file)
        self.path = path
        
        # Draw and render
        if draw:
            self.draw()
        if render:
            self.render()

    # Draws and saves the loss and learning rate curves
    def draw(self):
        print("Plotting autoencoder training curve...")
        
        # Get the moving average and stdev of the learning curve
        window = len(self.loss_curve) // 50
        if window > 1:
            rolling_std = np.array(pd.Series(self.loss_curve).rolling(window).std())
            rolling_avg = np.array(pd.Series(self.loss_curve).rolling(window).mean())
            rolling_std = rolling_std[~np.isnan(rolling_std)]
            rolling_avg = rolling_avg[~np.isnan(rolling_avg)]
            
            # Draw training curve with rolling values
            plt.clf()
            plt.title("Loss Curve, Window = " + str(window),fontsize='xx-large')
            plt.xlabel("Batch",fontsize='large')
            plt.ylabel("MSE",fontsize='large')
            plt.plot(np.array([*range(len(rolling_avg))])+(len(self.loss_curve)-len(rolling_avg)+1),rolling_avg,lw=2.5,c='r')
            plt.fill_between(np.array([*range(len(rolling_avg))])+(len(self.loss_curve)-len(rolling_avg)+1),rolling_avg+rolling_std,rolling_avg-rolling_std,color='r',alpha=0.2,lw=0.0)
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            save_file = self.path + "/rolling_loss.png"
            plt.savefig(save_file, dpi = 500)
            plt.close()
        
        # Draw training curve
        plt.clf()
        plt.title("Loss Curve",fontsize='xx-large')
        plt.xlabel("Batch",fontsize='large')
        plt.ylabel("MSE",fontsize='large')
        plt.plot([*range(len(self.loss_curve))],self.loss_curve,lw=2.5,c='r')
        plt.yscale("log")
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = self.path + "/loss.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()
        
        # Draw learning rate curve
        plt.clf()
        plt.title("Learning Rate",fontsize='xx-large')
        plt.xlabel("Batch",fontsize='large')
        plt.ylabel("Learning Rate",fontsize='large')
        plt.plot([*range(len(self.lr_curve))],self.lr_curve,lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = self.path + "/lr.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()
    
    # Renders snapshots taken during training to single folder
    def render(self):
        print("Rendering...")
        
        # Find save paths
        if not os.path.isdir(self.path + "/video"):
            os.mkdir(self.path + "/video")
            
        # Create grids
        x_grid, y_grid = np.meshgrid(np.linspace(0,1,self.dim_1), np.linspace(0,1,self.dim_2))
            
        for snapshot in range(len(self.batch_num_snapshots)):
                   
            # Minimum loss figure
            plt.cla()
            plt.clf()
            fig = plt.figure(constrained_layout=True, figsize=(8, 8))
            subfigs = fig.subfigures(3, 1)
            input_ax = subfigs[0].subplots(1,1)
            target_axs = subfigs[1].subplots(1, self.num_targets)
            output_axs = subfigs[2].subplots(1, self.num_targets)
            
            # Minimum loss input image
            input_ax.pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.input_snapshots[0][snapshot])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
            input_ax.tick_params(axis='x',labelsize=12)
            input_ax.tick_params(axis='y',labelsize=12)
            input_ax.set_aspect(0.25, adjustable='box')
            input_ax.set_title('Input',fontsize='x-large')
            
            # Minimum loss target image
            if self.num_targets == 1:
                target_axs.pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.targets_snapshots[0][snapshot][0])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                target_axs.tick_params(axis='x',labelsize=12)
                target_axs.tick_params(axis='y',labelsize=12)
                target_axs.set_aspect(0.25, adjustable='box')
                target_axs.set_title('Target',fontsize='x-large')
            else:
                for target in range(self.num_targets):
                    target_axs[target].pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.targets_snapshots[0][snapshot][target])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                    target_axs[target].tick_params(axis='x',labelsize=12)
                    target_axs[target].tick_params(axis='y',labelsize=12)
                    target_axs[target].set_aspect(0.25, adjustable='box')
                    target_axs[target].set_title('Target '+str(target+1),fontsize='x-large')
            
            # Minimum loss output image
            if self.num_targets == 1:
                output_axs.pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.outputs_snapshots[0][snapshot][0])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                output_axs.tick_params(axis='x',labelsize=12)
                output_axs.tick_params(axis='y',labelsize=12)
                output_axs.set_aspect(0.25, adjustable='box')
                output_axs.set_title('Output',fontsize='x-large')
            else:
                for output in range(self.num_targets):
                    output_axs[output].pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.outputs_snapshots[0][snapshot][output])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                    output_axs[output].tick_params(axis='x',labelsize=12)
                    output_axs[output].tick_params(axis='y',labelsize=12)
                    output_axs[output].set_aspect(0.25, adjustable='box')
                    output_axs[output].set_title('Output '+str(target+1),fontsize='x-large')
            
            # Save and close minimum loss figure
            plt.suptitle('Minimum Loss in Batch ' + str(self.batch_num_snapshots[snapshot])+"\nLoss = " + '{:.3f}'.format(self.loss_snapshots[0][snapshot]) + "\n",fontsize='xx-large')
            plt.savefig(self.path + "/video/min_in_"+str(snapshot).zfill(3)+'.png', dpi=100)
            plt.close()
            
            # Average loss figure
            plt.cla()
            plt.clf()
            fig = plt.figure(constrained_layout=True, figsize=(8, 8))
            subfigs = fig.subfigures(3, 1)
            input_ax = subfigs[0].subplots(1,1)
            target_axs = subfigs[1].subplots(1, self.num_targets)
            output_axs = subfigs[2].subplots(1, self.num_targets)
            
            # Average loss input image
            input_ax.pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.input_snapshots[1][snapshot])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
            input_ax.tick_params(axis='x',labelsize=12)
            input_ax.tick_params(axis='y',labelsize=12)
            input_ax.set_aspect(0.25, adjustable='box')
            input_ax.set_title('Input',fontsize='x-large')
            
            # Average loss target image
            if self.num_targets == 1:
                target_axs.pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.targets_snapshots[1][snapshot][0])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                target_axs.tick_params(axis='x',labelsize=12)
                target_axs.tick_params(axis='y',labelsize=12)
                target_axs.set_aspect(0.25, adjustable='box')
                target_axs.set_title('Target',fontsize='x-large')
            else:
                for target in range(self.num_targets):
                    target_axs[target].pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.targets_snapshots[1][snapshot][target])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                    target_axs[target].tick_params(axis='x',labelsize=12)
                    target_axs[target].tick_params(axis='y',labelsize=12)
                    target_axs[target].set_aspect(0.25, adjustable='box')
                    target_axs[target].set_title('Target '+str(target+1),fontsize='x-large')
            
            # Average loss output image
            if self.num_targets == 1:
                output_axs.pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.outputs_snapshots[1][snapshot][0])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                output_axs.tick_params(axis='x',labelsize=12)
                output_axs.tick_params(axis='y',labelsize=12)
                output_axs.set_aspect(0.25, adjustable='box')
                output_axs.set_title('Output',fontsize='x-large')
            else:
                for output in range(self.num_targets):
                    output_axs[output].pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.outputs_snapshots[1][snapshot][output])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                    output_axs[output].tick_params(axis='x',labelsize=12)
                    output_axs[output].tick_params(axis='y',labelsize=12)
                    output_axs[output].set_aspect(0.25, adjustable='box')
                    output_axs[output].set_title('Output '+str(target+1),fontsize='x-large')
            
            # Save and close average loss figure
            plt.suptitle('Average Loss in Batch ' + str(self.batch_num_snapshots[snapshot])+"\nLoss = " + '{:.3f}'.format(self.loss_snapshots[1][snapshot]) + "\n",fontsize='xx-large')
            plt.savefig(self.path + "/video/avg_in_"+str(snapshot).zfill(3)+'.png', dpi=100)
            plt.close()
            
            # Maximum loss figure
            plt.cla()
            plt.clf()
            fig = plt.figure(constrained_layout=True, figsize=(8, 8))
            subfigs = fig.subfigures(3, 1)
            input_ax = subfigs[0].subplots(1,1)
            target_axs = subfigs[1].subplots(1, self.num_targets)
            output_axs = subfigs[2].subplots(1, self.num_targets)
            
            # Maximum loss input image
            input_ax.pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.input_snapshots[2][snapshot])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
            input_ax.tick_params(axis='x',labelsize=12)
            input_ax.tick_params(axis='y',labelsize=12)
            input_ax.set_aspect(0.25, adjustable='box')
            input_ax.set_title('Input',fontsize='x-large')
            
            # Maximum loss target image
            if self.num_targets == 1:
                target_axs.pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.targets_snapshots[2][snapshot][0])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                target_axs.tick_params(axis='x',labelsize=12)
                target_axs.tick_params(axis='y',labelsize=12)
                target_axs.set_aspect(0.25, adjustable='box')
                target_axs.set_title('Target',fontsize='x-large')
            else:
                for target in range(self.num_targets):
                    target_axs[target].pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.targets_snapshots[2][snapshot][target])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                    target_axs[target].tick_params(axis='x',labelsize=12)
                    target_axs[target].tick_params(axis='y',labelsize=12)
                    target_axs[target].set_aspect(0.25, adjustable='box')
                    target_axs[target].set_title('Target '+str(target+1),fontsize='x-large')
            
            # Maximum loss output image
            if self.num_targets == 1:
                output_axs.pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.outputs_snapshots[2][snapshot][0])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                output_axs.tick_params(axis='x',labelsize=12)
                output_axs.tick_params(axis='y',labelsize=12)
                output_axs.set_aspect(0.25, adjustable='box')
                output_axs.set_title('Output',fontsize='x-large')
            else:
                for output in range(self.num_targets):
                    output_axs[output].pcolormesh(x_grid, y_grid, np.transpose(self.normalize(self.outputs_snapshots[2][snapshot][output])), shading='nearest', cmap='jet', vmin=0.0, vmax=1.0)
                    output_axs[output].tick_params(axis='x',labelsize=12)
                    output_axs[output].tick_params(axis='y',labelsize=12)
                    output_axs[output].set_aspect(0.25, adjustable='box')
                    output_axs[output].set_title('Output '+str(target+1),fontsize='x-large')
            
            # Save and close maximum loss figure
            plt.suptitle('Maximum Loss in Batch ' + str(self.batch_num_snapshots[snapshot])+"\nLoss = " + '{:.3f}'.format(self.loss_snapshots[2][snapshot]) + "\n",fontsize='xx-large')
            plt.savefig(self.path + "/video/max_in_"+str(snapshot).zfill(3)+'.png', dpi=100)
            plt.close()