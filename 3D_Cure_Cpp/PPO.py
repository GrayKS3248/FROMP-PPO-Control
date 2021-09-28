# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:41:07 2020

@author: Grayson Schaer
"""

# Deep RL networks + autoencoder
from CNN_MLP_Actor import Model as actor_nn
from CNN_MLP_Critic import Model as critic_nn

# Number manipulation
import torch
import numpy as np
from scipy import interpolate

# File saving and formatting
import pickle
import os

# Rendering and plotting
import matplotlib.pyplot as plt

class Agent:

    def __init__(self, num_additional_states, num_inputs, steps_per_trajectory, trajectories_per_batch,
                 epochs_per_batch, gamma, lamb, epsilon, alpha, decay_rate, load_path, reset_std):

        # Find load file for either offline trained autoencoder or previously trained PPO agent
        if not os.path.exists(load_path + "/output"):
            raise RuntimeError("Could not find load file: " + load_path)
        with open(load_path + "/output", 'rb') as file:
            loaded_data = pickle.load(file)
            
        # Load offline trained autoencoder
        if ( ('model' in loaded_data) and ('dim_1' in loaded_data) and ('dim_2' in loaded_data) ):
            autoencoder_model = loaded_data['model']
            self.x_dim = loaded_data['dim_1']
            self.y_dim = loaded_data['dim_2']
            self.num_additional_states = num_additional_states
            self.num_inputs = num_inputs
            
            self.actor = actor_nn(np.max((self.x_dim, self.y_dim)), self.num_additional_states+2+self.num_inputs, self.num_inputs)
            self.actor.conv1.load_state_dict(autoencoder_model.conv1.state_dict())
            self.actor.conv2.load_state_dict(autoencoder_model.conv2.state_dict())
            self.actor.conv3.load_state_dict(autoencoder_model.conv3.state_dict())
            self.actor.conv4.load_state_dict(autoencoder_model.conv4.state_dict())
            self.actor.conv5.load_state_dict(autoencoder_model.conv5.state_dict())
            self.actor.fc1.load_state_dict(autoencoder_model.fc1.state_dict())
            
            self.critic = critic_nn(np.max((self.x_dim, self.y_dim)), self.num_additional_states+2+self.num_inputs)
            self.critic.conv1.load_state_dict(autoencoder_model.conv1.state_dict())
            self.critic.conv2.load_state_dict(autoencoder_model.conv2.state_dict())
            self.critic.conv3.load_state_dict(autoencoder_model.conv3.state_dict())
            self.critic.conv4.load_state_dict(autoencoder_model.conv4.state_dict())
            self.critic.conv5.load_state_dict(autoencoder_model.conv5.state_dict())
            self.critic.fc1.load_state_dict(autoencoder_model.fc1.state_dict())
        
            self.prev_r_per_episode = np.array([])
            self.prev_value_error = np.array([])
            self.prev_actor_lr = np.array([])
            self.prev_critic_lr = np.array([])
            self.prev_x_loc_stdev = np.array([])
            self.prev_y_loc_stdev = np.array([])
            self.prev_mag_stdev = np.array([])
            
        # Load previously trained PPO agent 
        elif ( ('actor' in loaded_data) and ('critic' in loaded_data) ):
            self.actor = loaded_data['actor']
            self.critic = loaded_data['critic']
            self.num_inputs = loaded_data['actor'].num_outputs
            self.num_additional_states = loaded_data['actor'].num_additional_inputs - 2 - self.num_inputs
            self.x_dim = loaded_data['actor'].dim
            self.y_dim = loaded_data['actor'].dim
            self.prev_r_per_episode = loaded_data['r_per_episode']
            self.prev_value_error = loaded_data['value_error']
            self.prev_actor_lr = loaded_data['actor_lr']
            self.prev_critic_lr = loaded_data['critic_lr']
            self.prev_x_loc_stdev = loaded_data['x_loc_stdev']
            self.prev_y_loc_stdev = loaded_data['y_loc_stdev']
            self.prev_mag_stdev = loaded_data['mag_stdev']
            
        # Throw error if autoencoder or PPO agent not found
        else:
            raise RuntimeError("Invalid data type at: " + load_path)
            
        # Batch memory
        self.state_images = [[]]
        self.state_vectors = [[]]
        self.actions = [[]]
        self.rewards = [[]]
        self.old_log_probs = [[]]
        self.advantage_estimates = [[]]
        self.value_targets = [[]]

        # Training parameters
        self.steps_per_trajectory = steps_per_trajectory
        self.trajectories_per_batch = trajectories_per_batch
        self.steps_per_batch = steps_per_trajectory * trajectories_per_batch
        self.epochs_per_batch = epochs_per_batch

        # Hyperparameters
        self.gamma = gamma
        self.lamb = lamb
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay_rate = decay_rate
        
        # Reset the actor stdev if requested
        if reset_std:
            self.actor.stdev = torch.nn.Parameter(-0.9*torch.ones(self.actor.fc5.out_features,dtype=torch.double).double())
            
        # Create optimizer and lr scheduler for actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters() , lr=self.alpha)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.actor_optimizer, gamma=self.decay_rate)
        
        # Create optimizer and lr scheduler for critic
        self.critic_optimizer =  torch.optim.Adam(self.critic.parameters() , lr=self.alpha)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.critic_optimizer, gamma=self.decay_rate)

        # Get device
        self.device = self.get_device()
        self.actor.to(self.device)
        self.critic.to(self.device)
        print("Device(")
        print("  " + self.device)
        print(")\n")
        print("Actor " + str(self.actor)) 
        print("")
        print("Critic " + str(self.critic)) 

    # Gets the cpu or gpu on which to run NN
    # @return device code
    def get_device(self):
        
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    # Converts any non square input image into a square input image via linear interpolation about the shortest axis
    # @param The state image to be converted
    # @return The squared state image
    def convert_to_square(self, state_image):
        
        # Extract dimensions
        state_image = np.array(state_image)
        largest_dim = max(state_image.shape)
        smallest_dim = min(state_image.shape)
        smallest_ax = np.argmin(state_image.shape)
        new_image = np.zeros((largest_dim, largest_dim))
        start_point_of_interpolant = 0
        
        # Extract information regarding the interpolation
        length_of_interpolant = largest_dim // (smallest_dim - 1)
        num_of_interpolants_one_longer = largest_dim - (length_of_interpolant * (smallest_dim - 1)) - 1
        spacing_on_longer_interpolants = (smallest_dim - 1) / num_of_interpolants_one_longer
        indicies_of_longer_interpolants = np.round(np.arange(0, smallest_dim - 1, spacing_on_longer_interpolants))
        
        # Interpolate
        for curr_interpolant_index in range(state_image.shape[smallest_ax]-1):
            if (indicies_of_longer_interpolants == curr_interpolant_index).any():
                length_of_curr_interpolant = length_of_interpolant+2
            else:
               length_of_curr_interpolant = length_of_interpolant+1
            curr_interpolant = np.transpose(np.linspace( state_image[:,curr_interpolant_index], state_image[:,curr_interpolant_index+1], length_of_curr_interpolant) )
            new_image[:,start_point_of_interpolant:start_point_of_interpolant+length_of_curr_interpolant] = curr_interpolant
            start_point_of_interpolant = start_point_of_interpolant + length_of_curr_interpolant-1
        
        return new_image
    
    # Normalizes input image so that the min is 0 and max is 1
    # @param The input image to be normalized
    # @param Reference value against which to normalize the min and max of the input image
    # @return The normalized input image and range of prenormalization
    def normalize(self, state_image, state_image_ref):
        
        # Normalize from 0 to 1
        min_of_input = np.min(state_image)
        max_of_input = np.max(state_image)
        
        if min_of_input != max_of_input:
            new_image = (state_image - min_of_input) / (max_of_input - min_of_input)
        else:
            new_image = state_image / min_of_input
            
        # Normalize min and max of input image based on reference
        min_of_input = min_of_input / state_image_ref
        max_of_input = max_of_input / state_image_ref
        
        return new_image, min_of_input, max_of_input

    # Calcuates determinisitc action given state and policy.
    # @ param state_image - The state image in which the policy is applied to calculate the action
    # @param Reference value against which to normalize the min and max of the input image
    # @ param additional_states - Any additional states generated by external estimators used to inform action generation
    # @ param inputs - The inputs over which the policy is applied
    # @ return action - The calculated deterministic action based on the state and policy
    def get_greedy_action(self, state_image, state_image_ref, additional_states, inputs):

        # Get the gaussian distribution parameters used to sample the action for the old and new policy
        with torch.no_grad():
            
            # Square and normalize state image
            sq_state_image = self.convert_to_square(state_image)
            norm_sq_state_image, min_of_state_image, max_of_state_image = self.normalize(sq_state_image, state_image_ref)
            
            # Convert to tensor of correct dimensions
            norm_sq_state_image = torch.tensor(norm_sq_state_image)
            norm_sq_state_image = norm_sq_state_image.reshape(1,1,norm_sq_state_image.shape[0],norm_sq_state_image.shape[1]).float()
            norm_sq_state_image = norm_sq_state_image.to(self.device)
            
            # Populate state vector
            states = []
            states.append(min_of_state_image)
            states.append(max_of_state_image)
            for i in range(self.num_additional_states):
                states.append(additional_states[i])
            for i in range(self.num_inputs):
                states.append(inputs[i])
                
            # Format state vector
            states = torch.tensor(states)
            states = states.reshape(1,states.shape[0]).float().to(self.device)
            
            # Forward propogate formatted state
            means, stdevs = self.actor.forward(norm_sq_state_image, states)
            means = means.squeeze().to('cpu')
            
        # Return the actions
        actions = []
        for i in range(self.num_inputs):
            actions.append(means[i].item())
        return tuple(actions)

    # Calcuates stochastic action given state and policy.
    # @ param state_image - The state image in which the policy is applied to calculate the action
    # @param Reference value against which to normalize the min and max of the input image
    # @ param additional_states - Any additional states generated by external estimators used to inform action generation
    # @ param inputs - The inputs over which the policy is applied
    # @ return action - The calculated stochastic action based on the state and policy
    # @ return stdev - The calculated stdev based on the policy
    def get_action(self, state_image, state_image_ref, additional_states, inputs):
        
        # Get the gaussian distribution parameters used to sample the action for the old and new policy
        with torch.no_grad():
            
            # Square and normalize state image
            sq_state_image = self.convert_to_square(state_image)
            norm_sq_state_image, min_of_state_image, max_of_state_image = self.normalize(sq_state_image, state_image_ref)
            
            # Convert to tensor of correct dimensions
            norm_sq_state_image = torch.tensor(norm_sq_state_image)
            norm_sq_state_image = norm_sq_state_image.reshape(1,1,norm_sq_state_image.shape[0],norm_sq_state_image.shape[1]).float()
            norm_sq_state_image = norm_sq_state_image.to(self.device)
            
            # Populate state vector
            states = []
            states.append(min_of_state_image)
            states.append(max_of_state_image)
            for i in range(self.num_additional_states):
                states.append(additional_states[i])
            for i in range(self.num_inputs):
                states.append(inputs[i])
                
            # Format state vector
            states = torch.tensor(states)
            states = states.reshape(1,states.shape[0]).float().to(self.device)
            
            # Forward propogate formatted state
            means, stdevs = self.actor.forward(norm_sq_state_image, states)
            means = means.squeeze().to('cpu')
            
            # Sample the actions
            action = []
            stdev = []
            for i in range(self.num_inputs):
                dist = torch.distributions.normal.Normal(means[i].item(), stdevs[i].to('cpu'))
                action.append(dist.sample().item())
                stdev.append(stdevs[i].item())

        # Return the actions and stdevs
        return tuple(action) + tuple(stdev)

    # Updates the trajectory memory given an arbitrary time step
    # @ param state_image - The state image in which the policy is applied to calculate the action
    # @param Reference value against which to normalize the min and max of the input image
    # @ param additional_states - Any additional states generated by external estimators used to inform action generation
    # @ param inputs - The inputs over which the policy wass applied
    # @ param action - actions to be added to trajectory memory
    # @ param reward - reward to be added to trajectory memory
    def update_agent(self, state_image, state_image_ref, additional_states, inputs, action, reward):     
    
        # Update the action and reward memory
        self.actions[-1].append(np.array(action))
        self.rewards[-1].append(reward)
        
        # Get the current (will become the old during learning) log probs
        with torch.no_grad():
            # Square and normalize state image
            sq_state_image = self.convert_to_square(state_image)
            norm_sq_state_image, min_of_state_image, max_of_state_image = self.normalize(sq_state_image, state_image_ref)        
    
            # Update the state image memory
            self.state_images[-1].append(norm_sq_state_image)
            
            # Convert to tensor of correct dimensions
            norm_sq_state_image = torch.tensor(norm_sq_state_image)
            norm_sq_state_image = norm_sq_state_image.reshape(1,1,norm_sq_state_image.shape[0],norm_sq_state_image.shape[1]).float()
            norm_sq_state_image = norm_sq_state_image.to(self.device)
    
            # Populate state vector
            states = []
            states.append(min_of_state_image)
            states.append(max_of_state_image)
            for i in range(self.num_additional_states):
                states.append(additional_states[i])
            for i in range(self.num_inputs):
                states.append(inputs[i])   
            
            # Update the state vector memory
            self.state_vectors[-1].append(np.array(states))
            
            # Format state vector
            states = torch.tensor(states)
            states = states.reshape(1,states.shape[0]).float().to(self.device)
            
            # Forward propogate formatted state
            means, stdevs = self.actor.forward(norm_sq_state_image, states)
            means = means.squeeze().to('cpu')
            stdevs = stdevs.to('cpu')
            
            # Get log prob of actions selected
            action = torch.tensor(action)
            dist = torch.distributions.normal.Normal(means, stdevs)

            # Update the old log prob memory
            self.old_log_probs[-1].append(dist.log_prob(action).sum())
            
            # Gather current value estimates. Will be used for advantage estimate and value target calculations
            self.value_targets[-1].append(self.critic.forward(norm_sq_state_image, states).item())

        # If the current trajectory is complete, calculate advantage estimates, value targets, and add another trajectory column to the batch memory
        if len(self.state_images[-1]) == self.steps_per_trajectory:
            
            # Bootstrap value estimates with 0.0
            self.value_targets[-1].append(0.0)
            
            # Compute deltas for GAE
            self.advantage_estimates[-1] = np.array(self.rewards[-1]) + (self.gamma * np.array(self.value_targets[-1][1:])) - np.array(self.value_targets[-1][:-1])

            # Calculate advantage estimates using GAE
            for t in reversed(range(len(self.state_images[-1]) - 1)):
                self.advantage_estimates[-1][t] = self.advantage_estimates[-1][t] + (self.gamma * self.lamb * self.advantage_estimates[-1][t + 1])
                    
            # Get the value targets using TD(0)
            for t in reversed(range(len(self.state_images[-1]))):
                self.value_targets[-1][t] = self.rewards[-1][t] + (self.gamma * self.value_targets[-1][t + 1])
            self.value_targets[-1] = self.value_targets[-1][:-1]
                
            # Add another trajectory column to the batch memory so long as the batch is not full
            if len(self.state_images) != self.trajectories_per_batch:
                
                self.state_images.append([])
                self.state_vectors.append([])
                self.actions.append([])
                self.rewards.append([])
                self.old_log_probs.append([])
                self.value_targets.append([])
                self.advantage_estimates.append([])
                
                return []
              
            # If a batch is complete, learn
            else:
                # Convert batch data 
                with torch.no_grad():
                    self.state_images = torch.reshape(torch.tensor(self.state_images, dtype=torch.float), (self.steps_per_batch, 1, np.max((self.x_dim, self.y_dim)), np.max((self.x_dim, self.y_dim)))).to(self.device)
                    self.state_vectors = torch.reshape(torch.tensor(self.state_vectors, dtype=torch.float), (self.steps_per_batch, self.num_additional_states+2+self.num_inputs)).to(self.device)
                    self.actions = torch.reshape(torch.tensor(self.actions, dtype=torch.double), (self.steps_per_batch, self.num_inputs)).to(self.device)
                    self.rewards = torch.reshape(torch.tensor(self.rewards, dtype=torch.double), (-1,))
                    self.old_log_probs = torch.reshape(torch.tensor(self.old_log_probs, dtype=torch.double), (-1,)).to(self.device)
                    self.value_targets = torch.reshape(torch.tensor(self.value_targets, dtype=torch.double), (-1,)).to(self.device)
                    self.advantage_estimates = torch.reshape(torch.tensor(self.advantage_estimates, dtype=torch.double), (-1,)).to(self.device)
                    self.advantage_estimates = (self.advantage_estimates - torch.mean(self.advantage_estimates)) / torch.std(self.advantage_estimates)
                
                # Store learning rates
                output = []
                output.append(self.actor_lr_scheduler.get_last_lr()[0])
                output.append(self.critic_lr_scheduler.get_last_lr()[0])
                
                # Actor optimization
                for i in range(self.epochs_per_batch):
                    self.actor_optimizer.zero_grad()
                    means, stdevs = self.actor.forward(self.state_images, self.state_vectors)
                    means = means.to(self.device)
                    stdevs = stdevs.to(self.device)
                    log_prob = torch.tensor(0.0).to(self.device)
                    for j in range(self.num_inputs):
                        dist = torch.distributions.normal.Normal(means[:,j], stdevs[j])                        
                        log_prob = log_prob + dist.log_prob(self.actions[:,j])
                    prob_ratio = torch.exp(log_prob - self.old_log_probs)
                    loss = -(torch.min(self.advantage_estimates*prob_ratio, self.advantage_estimates*torch.clamp(prob_ratio, 1-self.epsilon, 1+self.epsilon))).mean()
                    loss.backward()
                    self.actor_optimizer.step()
                self.actor_lr_scheduler.step()
                
                # Critic optimization
                for i in range(self.epochs_per_batch):
                    self.critic_optimizer.zero_grad()
                    value_estimates = self.critic.forward(self.state_images, self.state_vectors).double()
                    loss = torch.nn.MSELoss()(value_estimates[:, 0], self.value_targets)
                    with torch.no_grad():
                        output.append(loss.item())
                    loss.backward()
                    self.critic_optimizer.step()
                self.critic_lr_scheduler.step()
                
                # After learning, reset the memory
                self.state_images = [[]]
                self.state_vectors = [[]]
                self.inputs = [[]]
                self.actions = [[]]
                self.rewards = [[]]
                self.old_log_probs = [[]]
                self.value_targets = [[]]
                self.advantage_estimates = [[]]
                
                return output
        
        return []

class Save_Plot_Render:

    def __init__(self):
        
        # Set all save params and values to trival
        self.r_per_episode = []
        self.value_error = []
        self.actor_lr = []
        self.critic_lr = []
        self.x_loc_stdev = []
        self.y_loc_stdev = []
        self.mag_stdev = []
        self.input_location_x = []
        self.input_location_y = []
        self.input_percent = []
        self.temperature_field = []
        self.cure_field = []
        self.fine_temperature_field = []
        self.fine_cure_field = []
        self.fine_mesh_loc = []
        self.global_fine_mesh_x = []
        self.global_fine_mesh_y = []
        self.max_temperature_field = []
        self.max_theta_field = []
        self.front_curve = []
        self.front_fit = []
        self.front_velocity = []
        self.front_temperature = []
        self.front_shape_param = []
        self.target = []
        self.time = []
        self.reward = []
        self.mesh_x_z0 = []
        self.mesh_y_z0 = []
        self.mesh_y_x0 = []
        self.mesh_z_x0 = []
        self.max_input_mag = []
        self.exp_const = []
        self.control_speed = []
    
    def store_training_curves(self, r_per_episode, value_error):
        self.r_per_episode = np.array(r_per_episode)
        self.value_error = np.array(value_error)
    
    def store_lr_curves(self, actor_lr, critic_lr):
        self.actor_lr = np.array(actor_lr)
        self.critic_lr = np.array(critic_lr)
        
    def store_stdev_history(self, x_loc_stdev, y_loc_stdev, mag_stdev):
        self.x_loc_stdev = np.array(x_loc_stdev)
        self.y_loc_stdev = np.array(y_loc_stdev)
        self.mag_stdev = np.array(mag_stdev)
    
    def store_input_history(self, input_location_x, input_location_y, input_percent):
        self.input_location_x = np.array(input_location_x)
        self.input_location_y = np.array(input_location_y)
        self.input_percent = np.array(input_percent)
    
    def store_field_history(self, temperature_field, cure_field, fine_temperature_field, fine_cure_field, fine_mesh_loc):
        self.temperature_field = np.array(temperature_field)
        self.cure_field = np.array(cure_field)
        self.fine_temperature_field = np.array(fine_temperature_field)
        self.fine_cure_field = np.array(fine_cure_field)
        self.fine_mesh_loc = np.array(fine_mesh_loc)
    
    def store_front_history(self, front_curve, front_fit, front_velocity, front_temperature, front_shape_param):
        self.front_curve = np.array(front_curve)
        self.front_fit = np.array(front_fit)
        self.front_velocity = np.array(front_velocity)
        self.front_temperature = np.array(front_temperature)
        self.front_shape_param = np.array(front_shape_param)
    
    def store_target_and_time(self, target, time, reward):
        self.target = np.array(target)
        self.time = np.array(time)
        self.reward = np.array(reward)
    
    def store_top_mesh(self, mesh_x_z0, mesh_y_z0):
        self.mesh_x_z0 = np.array(mesh_x_z0)
        self.mesh_y_z0 = np.array(mesh_y_z0)
    
    def store_input_params(self, max_input_mag, exp_const):
        self.max_input_mag = max_input_mag
        self.exp_const = exp_const
    
    def store_options(self, control_speed):
        self.control_speed = (control_speed == 1)
    
    def get_max_temperature_field(self):
        
        # Define fine resoluation mesh over which temperature data will be interpolated
        global_fine_x_linspace = np.linspace(0.0, self.fine_mesh_loc[0,1], len(self.fine_cure_field[0]))
        global_fine_y_linspace = np.linspace(0.0, self.mesh_y_z0[0, -1], len(self.fine_cure_field[0][0]))
        global_fine_x_linspace = np.linspace(0.0, self.mesh_x_z0[-1,0], np.int32(np.round(self.mesh_x_z0[-1,0] / global_fine_x_linspace[1])) + 1)
        global_fine_mesh_y, global_fine_mesh_x = np.meshgrid(global_fine_y_linspace, global_fine_x_linspace)
        
        # Create interpolated_coarse_temperature_field and populate with zeros
        interpolated_coarse_temperature_field = np.zeros( (len(global_fine_x_linspace), len(global_fine_y_linspace)) )
        extended_fine_temperature_field = np.zeros( (len(global_fine_x_linspace), len(global_fine_y_linspace)) )
        max_temperature_field = np.zeros( (len(self.time), len(global_fine_x_linspace), len(global_fine_y_linspace)) )
                
        # Define coordinates over which coarse temperature data is collected
        coarse_x_coords = self.mesh_x_z0[:,0]
        coarse_y_coords = self.mesh_y_z0[0,:]
        
        # Determine number of indicies in fine mesh data
        fine_mesh_indicies = len(self.fine_temperature_field[0,:,0])
        
        # Determine the maximum temperature at fine resolution at each frame
        for i in range(len(self.temperature_field[:,0,0])):
            
            # At each frame, interpolate the coarse temperautre field to the fine field resolution
            f = interpolate.interp2d(coarse_x_coords, coarse_y_coords, np.transpose(self.temperature_field[i]))
            interpolated_coarse_temperature_field = np.transpose(f(global_fine_x_linspace, global_fine_y_linspace))
            
            # At each frame, determine the global indices of the fine mesh
            mean_fine_mesh_x_location = (self.fine_mesh_loc[i][1] + self.fine_mesh_loc[i][0]) / 2.0
            mean_fine_mesh_x_index = np.argmin(abs(global_fine_x_linspace - mean_fine_mesh_x_location))
            fine_mesh_start_index = mean_fine_mesh_x_index - fine_mesh_indicies // 2
            if fine_mesh_start_index < 0:
                fine_mesh_start_index = 0
            fine_mesh_end_index = fine_mesh_start_index + fine_mesh_indicies
            if fine_mesh_end_index >= len(global_fine_x_linspace):
                fine_mesh_end_index = len(global_fine_x_linspace) - 1
                fine_mesh_start_index = fine_mesh_end_index - fine_mesh_indicies
            
            # Extend the fine mesh
            extended_fine_temperature_field = np.zeros( (len(global_fine_x_linspace), len(global_fine_y_linspace)) )
            extended_fine_temperature_field[fine_mesh_start_index:fine_mesh_end_index, :] = self.fine_temperature_field[i,:,:]
            
            # Compare the fine temperature mesh to the fine resolution corase temperature mesh
            max_temperature_field[i,:,:] = np.maximum(interpolated_coarse_temperature_field, extended_fine_temperature_field)
            
        # Return the maximum temperature field and mesh used to plot it
        return global_fine_mesh_x, global_fine_mesh_y, max_temperature_field
            
            
    
    def save(self, agent):
        print("\n\nSaving agent results...")
        
        # Find save paths
        done = False
        curr_folder = 1
        while not done:
            path = "results/PPO_"+str(curr_folder)
            video_1_path = "results/PPO_"+str(curr_folder)+"/video_1/"
            video_2_path = "results/PPO_"+str(curr_folder)+"/video_2/"
            if not os.path.isdir(path):
                os.mkdir(path)
                os.mkdir(video_1_path)
                os.mkdir(video_2_path)
                done = True
            else:
                curr_folder = curr_folder + 1
                
        # Store save paths
        self.path = path
        self.video_1_path = video_1_path
        self.video_2_path = video_2_path
        
        # Determine mean front locations
        self.mean_front_x_locations = np.zeros(len(self.front_curve))
        self.mean_front_y_locations = np.zeros(len(self.front_curve))
        for curr_frame in range(len(self.front_curve)):
            curr_mean_x = self.front_curve[curr_frame][0]
            curr_mean_x = curr_mean_x[curr_mean_x > 0.0]
            if len(curr_mean_x) != 0: 
                self.mean_front_x_locations[curr_frame] = np.mean(curr_mean_x)
            else:
                self.mean_front_x_locations[curr_frame] = 0.0
            curr_mean_y = self.front_curve[curr_frame][1]
            curr_mean_y = curr_mean_y[curr_mean_y > 0.0]
            if len(curr_mean_y) != 0: 
                self.mean_front_y_locations[curr_frame] = np.mean(curr_mean_y)
            else:
                self.mean_front_y_locations[curr_frame] = 0.0
        
        # Calculate the max temperature field
        self.global_fine_mesh_x, self.global_fine_mesh_y, self.max_temperature_field = self.get_max_temperature_field()
        
        # Concatenate previous training results
        if len(agent.prev_r_per_episode) != 0:
            self.r_per_episode = np.concatenate((agent.prev_r_per_episode, self.r_per_episode))
            self.value_error = np.concatenate((agent.prev_value_error, self.value_error))
            self.actor_lr = np.concatenate((agent.prev_actor_lr, self.actor_lr))
            self.critic_lr = np.concatenate((agent.prev_critic_lr, self.critic_lr))
            self.x_loc_stdev = np.concatenate((agent.prev_x_loc_stdev, self.x_loc_stdev))
            self.y_loc_stdev = np.concatenate((agent.prev_y_loc_stdev, self.y_loc_stdev))
            self.mag_stdev = np.concatenate((agent.prev_mag_stdev, self.mag_stdev))
        
        # Compile all stored data to dictionary
        data = {
            'r_per_episode' : self.r_per_episode,
            'value_error' : self.value_error,
            'actor_lr' : self.actor_lr,
            'critic_lr' : self.critic_lr,
            'x_loc_stdev': self.x_loc_stdev,
            'y_loc_stdev': self.y_loc_stdev,
            'mag_stdev': self.mag_stdev,
            'input_location_x': self.input_location_x,
            'input_location_y': self.input_location_y,
            'input_percent': self.input_percent,
            'temperature_field': self.temperature_field,
            'cure_field': self.cure_field,
            'fine_temperature_field': self.fine_temperature_field,
            'fine_cure_field': self.fine_cure_field,
            'fine_mesh_loc': self.fine_mesh_loc,
            'global_fine_mesh_x': self.global_fine_mesh_x, 
            'global_fine_mesh_y': self.global_fine_mesh_y,
            'max_temperature_field': self.max_temperature_field,
            'front_curve': self.front_curve,
            'front_fit' : self.front_fit,
            'mean_front_x_locations': self.mean_front_x_locations,
            'mean_front_y_locations': self.mean_front_y_locations,
            'front_velocity': self.front_velocity,
            'front_temperature': self.front_temperature,
            'front_shape_param': self.front_shape_param,
            'target': self.target,
            'time': self.time,
            'reward': self.reward,
            'mesh_x_z0' : self.mesh_x_z0,
            'mesh_y_z0' : self.mesh_y_z0,
            'max_input_mag' : self.max_input_mag,
            'exp_const' : self.exp_const,
            'control_speed' : self.control_speed,
            'actor': agent.actor,
            'critic': agent.critic
        }
        
        # Save the stored data
        with open(self.path + "/output", 'wb') as file:
            pickle.dump(data, file)
            
    def save_without_agent(self):
        print("\n\nSaving simulation results...")
        
        # Find save paths
        done = False
        curr_folder = 1
        while not done:
            path = "results/SIM_"+str(curr_folder)
            video_1_path = "results/SIM_"+str(curr_folder)+"/video_1/"
            video_2_path = "results/SIM_"+str(curr_folder)+"/video_2/"
            if not os.path.isdir(path):
                os.mkdir(path)
                os.mkdir(video_1_path)
                os.mkdir(video_2_path)
                done = True
            else:
                curr_folder = curr_folder + 1
                
        # Store save paths
        self.path = path
        self.video_1_path = video_1_path
        self.video_2_path = video_2_path
        
        # Determine mean front locations
        self.mean_front_x_locations = np.zeros(len(self.front_curve))
        self.mean_front_y_locations = np.zeros(len(self.front_curve))
        for curr_frame in range(len(self.front_curve)):
            curr_mean_x = self.front_curve[curr_frame][0]
            curr_mean_x = curr_mean_x[curr_mean_x > 0.0]
            if len(curr_mean_x) != 0: 
                self.mean_front_x_locations[curr_frame] = np.mean(curr_mean_x)
            else:
                self.mean_front_x_locations[curr_frame] = 0.0
            curr_mean_y = self.front_curve[curr_frame][1]
            curr_mean_y = curr_mean_y[curr_mean_y > 0.0]
            if len(curr_mean_y) != 0: 
                self.mean_front_y_locations[curr_frame] = np.mean(curr_mean_y)
            else:
                self.mean_front_y_locations[curr_frame] = 0.0
        
        # Calculate the max temperature field
        self.global_fine_mesh_x, self.global_fine_mesh_y, self.max_temperature_field = self.get_max_temperature_field()
        
        # Compile all stored data to dictionary
        data = {
            'input_location_x': self.input_location_x,
            'input_location_y': self.input_location_y,
            'input_percent': self.input_percent,
            'temperature_field': self.temperature_field,
            'cure_field': self.cure_field,
            'fine_temperature_field': self.fine_temperature_field,
            'fine_cure_field': self.fine_cure_field,
            'fine_mesh_loc': self.fine_mesh_loc,
            'global_fine_mesh_x': self.global_fine_mesh_x, 
            'global_fine_mesh_y': self.global_fine_mesh_y,
            'max_temperature_field': self.max_temperature_field,
            'front_curve': self.front_curve,
            'front_fit' : self.front_fit,
            'mean_front_x_locations': self.mean_front_x_locations,
            'mean_front_y_locations': self.mean_front_y_locations,
            'front_velocity': self.front_velocity,
            'front_temperature': self.front_temperature,
            'front_shape_param': self.front_shape_param,
            'target': self.target,
            'time': self.time,
            'reward': self.reward,
            'mesh_x_z0' : self.mesh_x_z0,
            'mesh_y_z0' : self.mesh_y_z0,
            'max_input_mag' : self.max_input_mag,
            'exp_const' : self.exp_const,
            'control_speed' : self.control_speed,
        }
        
        # Save the stored data
        with open(self.path + "/output", 'wb') as file:
            pickle.dump(data, file)
    
    def plot(self):
        print("Plotting simulation results...")
    
        # Plot the trajectory
        if self.control_speed:
            # Plot speed trajectory
            plt.clf()
            plt.title("Front Velocity",fontsize='xx-large')
            plt.xlabel("Simulation Time [s]",fontsize='large')
            plt.ylabel("Front Velocity [mm/s]",fontsize='large')
            plt.plot(self.time, 1000.0*self.front_velocity,c='r',lw=2.5)
            plt.plot(self.time, 1000.0*self.target,c='b',ls='--',lw=2.5)
            plt.legend(('Actual','Target'),loc='upper right',fontsize='large')
            plt.ylim(0.0, 1500.0*np.max(self.target))
            plt.xlim(0.0, np.round(self.time[-1]))
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/trajectory.png", dpi = 500)
            plt.close()
            
            # Plot front temperature trajectory
            sorted_mean_front_x_locations = 1000.0*np.array(sorted(self.mean_front_x_locations))
            sorted_front_temperature = np.array([x for _, x in sorted(zip(self.mean_front_x_locations, self.front_temperature))])-273.15
            plt.clf()
            plt.title("Front Temperature",fontsize='xx-large')
            plt.xlabel("Location [mm]",fontsize='large')
            plt.ylabel("Front Temperature [C]",fontsize='large')
            plt.plot(sorted_mean_front_x_locations, sorted_front_temperature, c='r', lw=2.5)
            plt.ylim(0.0, 1.025*max(self.front_temperature-273.15))
            plt.xlim(0.0, 1000.0*self.mesh_x_z0[-1,0])
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/temp.png", dpi = 500)
            plt.close()
            
        else:
            # Plot speed trajectory
            plt.clf()
            plt.title("Front Velocity",fontsize='xx-large')
            plt.xlabel("Simulation Time [s]",fontsize='large')
            plt.ylabel("Front Velocity [mm/s]",fontsize='large')
            plt.plot(self.time, 1000.0*self.front_velocity,c='r',lw=2.5)
            plt.ylim(0.0, 1.025*max(1000.0*self.front_velocity))
            plt.xlim(0.0, np.round(self.time[-1]))
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/speed.png", dpi = 500)
            plt.close()
            
            # Plot front temperature trajectory
            sorted_mean_front_x_locations = 1000.0*np.array(sorted(self.mean_front_x_locations))
            sorted_front_temperature = np.array([x for _, x in sorted(zip(self.mean_front_x_locations, self.front_temperature))])-273.15
            plt.clf()
            plt.title("Front Temperature",fontsize='xx-large')
            plt.xlabel("Location [mm]",fontsize='large')
            plt.ylabel("Front Temperature [C]",fontsize='large')
            plt.plot(sorted_mean_front_x_locations, sorted_front_temperature,c='r',lw=2.5)
            plt.plot(sorted_mean_front_x_locations, self.target-273.15,c='b',ls='--',lw=2.5)
            plt.legend(('Actual','Target'),loc='upper right',fontsize='large')
            plt.ylim(0.0, 1.5*(np.max(self.target)-273.15))
            plt.xlim(0.0, 1000.0*self.mesh_x_z0[-1,0])
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/trajectory.png", dpi = 500)
            plt.close()
            
        # Plot reward trajectory
        plt.clf()
        plt.title("Reward During Trajectory",fontsize='xx-large')
        plt.xlabel("Simulation Time [s]",fontsize='large')
        plt.ylabel("Reward [-]",fontsize='large')
        plt.plot(self.time, self.reward[:,0],c='k',lw=2.5)
        plt.plot(self.time, self.reward[:,1],c='r',lw=1.0)
        plt.plot(self.time, self.reward[:,2],c='b',lw=1.0)
        plt.plot(self.time, self.reward[:,3],c='g',lw=1.0)
        plt.plot(self.time, self.reward[:,4],c='m',lw=1.0)
        plt.plot(self.time, self.reward[:,5],c='c',lw=1.0)
        plt.legend(('Total','Input Loc','Input Mag','Max Temp','Shape','Target'),loc='upper right',fontsize='large')
        plt.xlim(0.0, np.round(self.time[-1]))
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        plt.savefig(self.path + "/reward.png", dpi = 500)
        plt.close()

        #Plot actor learning curve
        if(len(self.r_per_episode)!=0):
            plt.clf()
            plt.title("Actor Learning Curve, Episode-Wise",fontsize='xx-large')
            plt.xlabel("Episode",fontsize='large')
            plt.ylabel("Average Reward per Simulation Step [-]",fontsize='large')
            plt.plot([*range(len(self.r_per_episode))],self.r_per_episode,lw=2.5,c='r')
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/actor_learning.png", dpi = 500)
            plt.close()

        # Plot value learning curve
        if(len(self.value_error)!=0):
            plt.clf()
            title_str = "Critic Learning Curve"
            plt.title(title_str,fontsize='xx-large')
            plt.xlabel("Optimization Step",fontsize='large')
            plt.ylabel("MSE Loss [-]",fontsize='large')
            plt.plot([*range(len(self.value_error))],self.value_error,lw=2.5,c='r')
            plt.yscale("log")
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/critic_learning.png", dpi = 500)
            plt.close()
            
        # Plot actor learning rate curve
        if(len(self.actor_lr)!=0):
            plt.clf()
            title_str = "Actor Learning Rate"
            plt.title(title_str,fontsize='xx-large')
            plt.xlabel("Batch",fontsize='large')
            plt.ylabel("Alpha [-]",fontsize='large')
            plt.plot([*range(len(self.actor_lr))],self.actor_lr,lw=2.5,c='r')
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/actor_alpha.png", dpi = 500)
            plt.close()
            
        # Plot critic learning rate curve
        if(len(self.actor_lr)!=0):
            plt.clf()
            title_str = "Critic Learning Rate"
            plt.title(title_str,fontsize='xx-large')
            plt.xlabel("Batch",fontsize='large')
            plt.ylabel("Alpha [-]",fontsize='large')
            plt.plot([*range(len(self.critic_lr))],self.critic_lr,lw=2.5,c='r')
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/critic_alpha.png", dpi = 500)
            plt.close()

        # Plot x loc stdev curve
        if(len(self.x_loc_stdev)!=0):
            plt.clf()
            plt.title("X Position Stdev, Episode-Wise",fontsize='xx-large')
            plt.xlabel("Episode",fontsize='large')
            plt.ylabel("X Position Stdev [mm]",fontsize='large')
            plt.plot([*range(len(self.x_loc_stdev))],1000.0*np.array(self.x_loc_stdev),lw=2.5,c='r')
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/x_loc_stdev.png", dpi = 500)
            plt.close()

        # Plot y loc stdev curve
        if(len(self.y_loc_stdev)!=0):
            plt.clf()
            plt.title("Y Position Stdev, Episode-Wise",fontsize='xx-large')
            plt.xlabel("Episode",fontsize='large')
            plt.ylabel("Y Position Stdev [mm]",fontsize='large')
            plt.plot([*range(len(self.y_loc_stdev))],1000.0*np.array(self.y_loc_stdev),lw=2.5,c='r')
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/y_loc_stdev.png", dpi = 500)
            plt.close()

        # Plot magnitude stdev curve
        if(len(self.mag_stdev)!=0):
            plt.clf()
            plt.title("Magnitude Stdev, Episode-Wise",fontsize='xx-large')
            plt.xlabel("Episode",fontsize='large')
            plt.ylabel('Magnitude Stdev [KW/m^2-s]',fontsize='large')
            plt.plot([*range(len(self.mag_stdev))],0.001*np.array(self.mag_stdev),lw=2.5,c='r')
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/mag_stdev.png", dpi = 500)
            plt.close()
    
    def render(self):
        print("Rendering simulation results...")
        
        # Determine the front temperature in steady state propogation (region wherein front velocity is +- 0.25 stdev of the non zero mean front velocity
        # and the front temperature is +- 0.25 stdev of the non zero mean front temperature)
        lower_steady_state_vels = np.int32((self.front_velocity >= np.mean(self.front_velocity[self.front_velocity!=0]) - 0.25*np.std(self.front_velocity[self.front_velocity!=0])))
        upper_steady_state_vels = np.int32((self.front_velocity <= np.mean(self.front_velocity[self.front_velocity!=0]) + 0.25*np.std(self.front_velocity[self.front_velocity!=0])))
        lower_steady_state_temps = np.int32((self.front_temperature >= np.mean(self.front_temperature[self.front_velocity!=0]) - 0.25*np.std(self.front_temperature[self.front_velocity!=0])))
        upper_steady_state_temps = np.int32((self.front_temperature <= np.mean(self.front_temperature[self.front_velocity!=0]) + 0.25*np.std(self.front_temperature[self.front_velocity!=0])))
        steady_state = (lower_steady_state_vels + upper_steady_state_vels + lower_steady_state_temps + upper_steady_state_temps) == 4        
        max_steady_state_front_temp = np.max(self.front_temperature[steady_state])
        
        # Calculate the min and max temperatures of the extended front curve
        initial_temperature = self.front_temperature[0]
        
        # Determine fit space for front fit
        fit_y_coords = np.linspace(0.0, self.mesh_y_z0[0][-1], 100)
        
        for curr_step in range(len(self.time)):
        
            # Calculate input field
            input_percent = self.input_percent[curr_step]
            input_location_x = self.input_location_x[curr_step]
            input_location_y = self.input_location_y[curr_step]
            input_mesh = input_percent*self.max_input_mag*np.exp(((self.mesh_x_z0-input_location_x)**2*self.exp_const) +
               														   (self.mesh_y_z0-input_location_y)**2*self.exp_const)
            input_mesh[input_mesh<0.01*self.max_input_mag] = 0.0
               
            # Make fig for temperature, cure, and input
            plt.cla()
            plt.clf()
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
            fig.set_size_inches(11,8.5)
               
            # Calculate fine mesh
            x_linspace = np.linspace(self.fine_mesh_loc[curr_step][0], self.fine_mesh_loc[curr_step][1], len(self.fine_cure_field[curr_step]))
            y_linspace = np.linspace(self.mesh_y_z0[0][0], self.mesh_y_z0[0][len(self.mesh_y_z0[0])-1], len(self.fine_cure_field[curr_step][0]))
            fine_mesh_y, fine_mesh_x = np.meshgrid(y_linspace, x_linspace)
            
            # Plot max normalized temperature
            normalized_max_temp = (np.max(self.max_temperature_field[:curr_step+1,:,:],axis=0) - initial_temperature) / (max_steady_state_front_temp - initial_temperature)
            c0 = ax0.pcolormesh(1000.0*self.global_fine_mesh_x, 1000.0*self.global_fine_mesh_y, normalized_max_temp, shading='gouraud', cmap='jet', vmin=0.8, vmax=1.2)
            cbar0 = fig.colorbar(c0, ax=ax0)
            cbar0.set_label("Max ϴ [ΔT / $ΔT_{steady}$]",labelpad=20,fontsize='large')
            cbar0.ax.tick_params(labelsize=12)
            ax0.set_xlabel('X Position [mm]',fontsize='large')
            ax0.set_ylabel('Y Position [mm]',fontsize='large')
            ax0.tick_params(axis='x',labelsize=12)
            ax0.tick_params(axis='y',labelsize=12)
            ax0.set_aspect('equal', adjustable='box')
            ax0.set_title('Front Temp = '+'{:.2f}'.format(self.front_temperature[curr_step]-273.15)+' C | '+
                          'Front Speed = '+'{:.2f}'.format(self.front_velocity[curr_step]*1000.0)+' mm/s | '+
                          'Front Shape = '+'{:.2f}'.format(self.front_shape_param[curr_step])+'\n',fontsize='large', fontname = 'monospace')
               
            # Plot cure
            c1 = ax1.pcolormesh(1000.0*self.mesh_x_z0, 1000.0*self.mesh_y_z0, self.cure_field[curr_step,:,:], shading='gouraud', cmap='YlOrBr', vmin=0.0, vmax=1.0)
            ax1.pcolormesh(1000.0*fine_mesh_x, 1000.0*fine_mesh_y, self.fine_cure_field[curr_step,:,:], shading='gouraud', cmap='YlOrBr', vmin=0.0, vmax=1.0)
            cbar1 = fig.colorbar(c1, ax=ax1)
            cbar1.set_label('Degree Cure [-]', labelpad=20,fontsize='large')
            cbar1.ax.tick_params(labelsize=12)
            ax1.set_xlabel('X Position [mm]',fontsize='large')
            ax1.set_ylabel('Y Position [mm]',fontsize='large')
            ax1.tick_params(axis='x',labelsize=12)
            ax1.tick_params(axis='y',labelsize=12)
            ax1.set_aspect('equal', adjustable='box')
            ax1.set_title('Reward = '+'{:.2f}'.format(self.reward[curr_step,0]),fontsize='large', fontname = 'monospace')
                    
            # Determine front locations based on front curve data
            front_x_location = self.front_curve[curr_step][0]
            front_y_location = self.front_curve[curr_step][1]
            front_x_location = front_x_location[front_x_location >= 0.0]
            front_y_location = front_y_location[front_y_location >= 0.0]
            front_x_location = 1000.0*front_x_location
            front_y_location = 1000.0*front_y_location
            front_instances = len(front_x_location)
        
            # Determine front locations based on fit data
            fit_x_coords = np.zeros(len(fit_y_coords))
            for order in range(len(self.front_fit[curr_step])):
                fit_x_coords = fit_x_coords + self.front_fit[curr_step][order] * (fit_y_coords**order)
        
            # Plot input
            c2 = ax2.pcolormesh(1000.0*self.mesh_x_z0, 1000.0*self.mesh_y_z0, 1.0e-3*input_mesh, shading='gouraud', cmap='coolwarm', vmin=0.0, vmax=1.0e-3*self.max_input_mag)
            if front_instances != 0:
                ax2.plot(front_x_location, front_y_location, 's', color='black', markersize=2.25, markeredgecolor='black')
                ax2.plot(1000.0*fit_x_coords, 1000.0*fit_y_coords, lw=2.0, c='m')
            ax2.axvline(1000.0*self.fine_mesh_loc[curr_step][0], color='red', alpha=0.70, ls='--')
            ax2.axvline(1000.0*self.fine_mesh_loc[curr_step][1], color='red', alpha=0.70, ls='--')
            cbar2 = fig.colorbar(c2, ax=ax2)
            cbar2.set_label('Input Heat [KW/m^2]',labelpad=20,fontsize='large')
            cbar2.ax.tick_params(labelsize=12)
            ax2.set_xlabel('X Position [mm]',fontsize='large')
            ax2.set_ylabel('Y Position [mm]',fontsize='large')
            ax2.set_xlim(0.0, 1000.0*self.mesh_x_z0[-1][0])
            ax2.set_ylim(0.0, 1000.0*self.mesh_y_z0[0][-1])
            ax2.tick_params(axis='x',labelsize=12)
            ax2.tick_params(axis='y',labelsize=12)
            ax2.set_aspect('equal', adjustable='box')
            if self.max_input_mag > 0.0:
                ax2.set_title('Input Power = '+'{:.2f}'.format(self.input_percent[curr_step]*100.0)+' %' ,fontsize='large', fontname = 'monospace')
            else:
                ax2.set_title('Input Power = '+'{:.2f}'.format(0.00)+' %' ,fontsize='large', fontname = 'monospace')
            
            # Set title and save
            title_str = "Time From Trigger: "+'{:.2f}'.format(self.time[curr_step])+'s'
            fig.suptitle(title_str,fontsize='xx-large', fontname = 'monospace')
            plt.savefig(self.video_1_path+str(curr_step).zfill(4)+'.png', dpi=100)
            plt.close()
            
            # Get the trimmed temperature and cure curves for the front curve plotting
            fine_front_temp_curve = self.fine_temperature_field[curr_step][:,len(self.fine_temperature_field[curr_step][0])//2]
            fine_front_cure_curve = self.fine_cure_field[curr_step][:,len(self.fine_cure_field[curr_step][0])//2]
            fine_x_coords = np.linspace(self.fine_mesh_loc[curr_step][0], self.fine_mesh_loc[curr_step][1], len(self.fine_temperature_field[0,:,0]))
            beg_plot = max((self.mean_front_x_locations[curr_step] - 0.40*(self.fine_mesh_loc[curr_step][1]-self.fine_mesh_loc[curr_step][0])), 0.0)
            end_plot = min((self.mean_front_x_locations[curr_step] + 0.40*(self.fine_mesh_loc[curr_step][1]-self.fine_mesh_loc[curr_step][0])), self.mesh_x_z0[-1,0])
            
            # Set up front plot
            plt.cla()
            plt.clf()
            fig, ax1 = plt.subplots()
            fig.set_size_inches(8.5,5.5)
            
            # Plot front temperature
            ax1.set_xlabel("X Location [mm]",fontsize='large')
            ax1.set_ylabel("Degree Cure [-]",fontsize='large',color='r')
            ax1.plot(1000.0*fine_x_coords, fine_front_cure_curve, c='r', lw=2.5)
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_xlim(1000.0*beg_plot, 1000.0*end_plot)
            ax1.tick_params(axis='x', labelsize=12)
            ax1.tick_params(axis='y', labelsize=12, labelcolor='r')
            
            # Plot front cure
            ax2 = ax1.twinx()
            ax2.set_ylabel("ϴ [ΔT / $ΔT_{steady}$]",fontsize='large',color='b')
            ax2.plot(1000.0*fine_x_coords, (fine_front_temp_curve-initial_temperature)/(max_steady_state_front_temp-initial_temperature), c='b', lw=2.5)  
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_xlim(1000.0*beg_plot, 1000.0*end_plot)
            ax2.tick_params(axis='x', labelsize=12)
            ax2.tick_params(axis='y', labelsize=12, labelcolor='b')
            
            # Save and close figure
            title_str = "Time From Trigger: "+'{:.2f}'.format(self.time[curr_step])+'s'
            fig.suptitle(title_str,fontsize='xx-large', fontname = 'monospace')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.video_2_path+str(curr_step).zfill(4)+'.png', dpi=100)
            plt.close()