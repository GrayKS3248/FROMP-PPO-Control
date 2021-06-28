# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:41:07 2020

@author: Grayson Schaer
"""

# Deep RL networks + autoencoder
from CNN_MLP_Actor import Model as actor_nn
from CNN_MLP_Critic import Model as critic_nn
from Autoencoder import Autoencoder

# Number manipulation
import torch
import numpy as np

# File saving and formatting
import pickle
import os

# Rendering and plotting
import matplotlib.pyplot as plt

class Agent:

    def __init__(self, steps_per_trajectory, trajectories_per_batch, epochs_per_batch, 
                 gamma, lamb, epsilon, alpha, decay_rate, autoencoder_path):

        # Batch memory
        self.states = [[]]
        self.inputs = [[]]
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
        
        # Find load file for encoder trained offline
        if not os.path.exists(autoencoder_path + "/output"):
            raise RuntimeError("Could not find load file: " + autoencoder_path)
        with open(autoencoder_path + "/output", 'rb') as file:
            loaded_encoder_data = pickle.load(file)
        autoencoder = loaded_encoder_data['autoencoder']
        
        # Set encoder parameters from loaded data
        self.x_dim = loaded_encoder_data['x_dim']
        self.y_dim = loaded_encoder_data['y_dim']
        self.bottleneck = loaded_encoder_data['bottleneck']
        self.kernal_size = loaded_encoder_data['kernal_size']
        
        # Build actor
        self.actor = actor_nn(self.x_dim, self.y_dim, self.bottleneck, self.kernal_size)
        
        # Copy offline trained encoder parameters to actor CNN
        self.actor.conv1.load_state_dict(autoencoder.conv1.state_dict())
        self.actor.conv2.load_state_dict(autoencoder.conv2.state_dict())
        self.actor.conv3.load_state_dict(autoencoder.conv3.state_dict())
        self.actor.fc0.load_state_dict(autoencoder.fc1.state_dict())
        
        # Create optimizer and lr scheduler for actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters() , lr=self.alpha)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.actor_optimizer, gamma=self.decay_rate)
        
        # Build critic
        self.critic = critic_nn(self.x_dim, self.y_dim, self.bottleneck, self.kernal_size)
        
        # Copy offline trained encoder parameters to critc CNN 
        self.critic.conv1.load_state_dict(autoencoder.conv1.state_dict())
        self.critic.conv2.load_state_dict(autoencoder.conv2.state_dict())
        self.critic.conv3.load_state_dict(autoencoder.conv3.state_dict())
        self.critic.fc0.load_state_dict(autoencoder.fc1.state_dict())
        
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

    # Calcuates determinisitc action given state and policy.
    # @ param state - The state in which the policy is applied to calculate the action
    # @ return action - The calculated deterministic action based on the state and policy
    def get_greedy_action(self, state, input_x, input_y, input_mag):

        # Get the gaussian distribution parameters used to sample the action for the old and new policy
        with torch.no_grad():
            # Format input state
            state = torch.tensor(state)
            state = state.reshape(1,1,state.shape[0],state.shape[1]).float().to(self.device)
            
            # Format input 
            input = torch.tensor([input_x, input_y, input_mag])
            input = input.reshape(1,input.shape[0]).float().to(self.device)
            
            # Forward propogate formatted state
            means, stdev_1, stdev_2, stdev_3 = self.actor.forward(state, input)
            means = means.squeeze().to('cpu')
            
        # Return the means
        action_1 = means[0].item()
        action_2 = means[1].item()
        action_3 = means[2].item()
        return action_1, action_2, action_3

    # Calcuates stochastic action given state and policy.
    # @ param state - The state in which the policy is applied to calculate the action
    # @ return action - The calculated stochastic action based on the state and policy
    # @ return stdev - The calculated stdev based on the policy
    def get_action(self, state, input_x, input_y, input_mag):

        # Get the gaussian distribution parameters used to sample the action for the old and new policy
        with torch.no_grad():
            # Format state
            state = torch.tensor(state)
            state = state.reshape(1,1,state.shape[0],state.shape[1]).float().to(self.device)
            
            # Format input 
            input = torch.tensor([input_x, input_y, input_mag])
            input = input.reshape(1,input.shape[0]).float().to(self.device)
        
            # Forward propogate formatted state
            means, stdev_1, stdev_2, stdev_3 = self.actor.forward(state, input)
            means = means.squeeze().to('cpu')
            
            # Sample the first action
            dist_1 = torch.distributions.normal.Normal(means[0].item(), stdev_1.to('cpu'))
            action_1 = dist_1.sample().item()
            stdev_1 = stdev_1.item()
    
            # Sample the second action
            dist_2 = torch.distributions.normal.Normal(means[1].item(), stdev_2.to('cpu'))
            action_2 = dist_2.sample().item()
            stdev_2 = stdev_2.item()
    
            # Sample the second action
            dist_3 = torch.distributions.normal.Normal(means[2].item(), stdev_3.to('cpu'))
            action_3 = dist_3.sample().item()
            stdev_3 = stdev_3.item()

        return action_1, stdev_1, action_2, stdev_2, action_3, stdev_3

    # Updates the trajectory memory given an arbitrary time step
    # @ param state - state to be added to trajectory memory
    # @ param action_1 - first action to be added to trajectory memory
    # @ param action_2 - second action to be added to trajectory memory
    # @ param action_3 - third action to be added to trajectory memory
    # @ param reward - reward to be added to trajectory memory
    def update_agent(self, state, input_x, input_y, input_mag, action_1, action_2, action_3, reward):

        # Update the state, action, and reward memory
        self.states[-1].append(state)
        self.inputs[-1].append([input_x, input_y, input_mag])
        self.actions[-1].append([action_1, action_2, action_3])
        self.rewards[-1].append(reward)
        
        # Get the current (will become the old during learning) log probs
        with torch.no_grad():
            # Convert the state type
            state = torch.tensor(state)
            state = state.reshape(1,1,state.shape[0],state.shape[1]).float().to(self.device)
            
            # Format input 
            input = torch.tensor([input_x, input_y, input_mag])
            input = input.reshape(1,input.shape[0]).float().to(self.device)
            
            # Calculate the distributions provided by actor
            means, stdev_1, stdev_2, stdev_3 = self.actor.forward(state, input)
            means = means.squeeze().to('cpu')
            stdevs = torch.tensor([stdev_1.to('cpu'), stdev_2.to('cpu'), stdev_3.to('cpu')])
            
            # Get log prob of actions selected
            actions = torch.tensor([action_1, action_2, action_3])
            dist = torch.distributions.normal.Normal(means, stdevs)
            self.old_log_probs[-1].append(dist.log_prob(actions).sum())
            
            # Gather current value estimates. Will be used for advantage estimate and value target calculations
            self.value_targets[-1].append(self.critic.forward(state, input).item())

        # If the current trajectory is complete, calculate advantage estimates, value targets, and add another trajectory column to the batch memory
        if len(self.states[-1]) == self.steps_per_trajectory:
            
            # Bootstrap value estimates with 0.0
            self.value_targets[-1].append(0.0)
            
            # Compute deltas for GAE
            self.advantage_estimates[-1] = np.array(self.rewards[-1]) + (self.gamma * np.array(self.value_targets[-1][1:])) - np.array(self.value_targets[-1][:-1])

            # Calculate advantage estimates using GAE
            for t in reversed(range(len(self.states[-1]) - 1)):
                self.advantage_estimates[-1][t] = self.advantage_estimates[-1][t] + (self.gamma * self.lamb * self.advantage_estimates[-1][t + 1])
                    
            # Get the value targets using TD(0)
            for t in reversed(range(len(self.states[-1]))):
                self.value_targets[-1][t] = self.rewards[-1][t] + (self.gamma * self.value_targets[-1][t + 1])
            self.value_targets[-1] = self.value_targets[-1][:-1]
                
            # Add another trajectory column to the batch memory so long as the batch is not full
            if len(self.states) != self.trajectories_per_batch:
                
                self.states.append([])
                self.inputs.append([])
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
                    self.states = torch.reshape(torch.tensor(self.states, dtype=torch.float), (self.steps_per_batch, 1, self.x_dim, self.y_dim)).to(self.device)
                    self.inputs = torch.reshape(torch.tensor(self.inputs, dtype=torch.float), (self.steps_per_batch, 3)).to(self.device)
                    self.actions = torch.reshape(torch.tensor(self.actions, dtype=torch.double), (self.steps_per_batch,3)).to(self.device)
                    self.rewards = torch.reshape(torch.tensor(self.rewards, dtype=torch.double), (-1,))
                    self.old_log_probs = torch.reshape(torch.tensor(self.old_log_probs, dtype=torch.double), (-1,)).to(self.device)
                    self.value_targets = torch.reshape(torch.tensor(self.value_targets, dtype=torch.double), (-1,)).to(self.device)
                    self.advantage_estimates = torch.reshape(torch.tensor(self.advantage_estimates, dtype=torch.double), (-1,)).to(self.device)
                    self.advantage_estimates = (self.advantage_estimates - torch.mean(self.advantage_estimates)) / torch.std(self.advantage_estimates)
                
                # Actor optimization
                for i in range(self.epochs_per_batch):
                    self.actor_optimizer.zero_grad()
                    means, stdev_1, stdev_2, stdev_3 = self.actor.forward(self.states, self.inputs)
                    dist_1 = torch.distributions.normal.Normal(means[:,0], stdev_1)
                    dist_2 = torch.distributions.normal.Normal(means[:,1], stdev_2)
                    dist_3 = torch.distributions.normal.Normal(means[:,2], stdev_3)
                    log_prob = dist_1.log_prob(self.actions[:,0]) + dist_2.log_prob(self.actions[:,1]) + dist_3.log_prob(self.actions[:,2])
                    prob_ratio = torch.exp(log_prob - self.old_log_probs)
                    loss = -(torch.min(self.advantage_estimates*prob_ratio, self.advantage_estimates*torch.clamp(prob_ratio, 1-self.epsilon, 1+self.epsilon))).mean()
                    loss.backward()
                    self.actor_optimizer.step()
                    self.actor_lr_scheduler.step()
                
                # Critic optimization
                critic_losses = []
                for i in range(self.epochs_per_batch):
                    self.critic_optimizer.zero_grad()
                    value_estimates = self.critic.forward(self.states, self.inputs).double()
                    loss = torch.nn.MSELoss()(value_estimates[:, 0], self.value_targets)
                    with torch.no_grad():
                        critic_losses.append(loss.item())
                    loss.backward()
                    self.critic_optimizer.step()
                    self.actor_lr_scheduler.step()
                
                # After learning, reset the memory
                self.states = [[]]
                self.inputs = [[]]
                self.actions = [[]]
                self.rewards = [[]]
                self.old_log_probs = [[]]
                self.value_targets = [[]]
                self.advantage_estimates = [[]]
                
                return critic_losses
        
        return []

class Save_Plot_Render:

    def __init__(self):
        
        # Set all save params and values to trival
        self.r_per_episode = []
        self.value_error = []
        self.x_rate_stdev = []
        self.y_rate_stdev = []
        self.mag_stdev = []
        self.input_location_x = []
        self.input_location_y = []
        self.input_percent = []
        self.temperature_field = []
        self.cure_field = []
        self.fine_temperature_field = []
        self.fine_cure_field = []
        self.fine_mesh_loc = []
        self.front_curve = []
        self.front_velocity = []
        self.front_temperature = []
        self.front_shape_param = []
        self.target = []
        self.time = []
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
    
    def store_stdev_history(self, x_rate_stdev, y_rate_stdev, mag_stdev):
        self.x_rate_stdev = np.array(x_rate_stdev)
        self.y_rate_stdev = np.array(y_rate_stdev)
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
    
    def store_front_history(self, front_curve, front_velocity, front_temperature, front_shape_param):
        self.front_curve = np.array(front_curve)
        self.front_velocity = np.array(front_velocity)
        self.front_temperature = np.array(front_temperature)
        self.front_shape_param = np.array(front_shape_param)
    
    def store_target_and_time(self, target, time):
        self.target = np.array(target)
        self.time = np.array(time)
    
    def store_top_mesh(self, mesh_x_z0, mesh_y_z0):
        self.mesh_x_z0 = np.array(mesh_x_z0)
        self.mesh_y_z0 = np.array(mesh_y_z0)
    
    def store_input_params(self, max_input_mag, exp_const):
        self.max_input_mag = max_input_mag
        self.exp_const = exp_const
    
    def store_options(self, control_speed):
        self.control_speed = (control_speed == 1)
    
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
        
        # Compile all stored data to dictionary
        data = {
            'r_per_episode' : self.r_per_episode,
            'value_error' : self.value_error,
            'x_rate_stdev': self.x_rate_stdev,
            'y_rate_stdev': self.y_rate_stdev,
            'mag_stdev': self.mag_stdev,
            'input_location_x': self.input_location_x,
            'input_location_y': self.input_location_y,
            'input_percent': self.input_percent,
            'temperature_field': self.temperature_field,
            'cure_field': self.cure_field,
            'fine_temperature_field': self.fine_temperature_field,
            'fine_cure_field': self.fine_cure_field,
            'fine_mesh_loc': self.fine_mesh_loc,
            'front_curve': self.front_curve,
            'mean_front_x_locations': self.mean_front_x_locations,
            'mean_front_y_locations': self.mean_front_y_locations,
            'front_velocity': self.front_velocity,
            'front_temperature': self.front_temperature,
            'front_shape_param': self.front_shape_param,
            'target': self.target,
            'time': self.time,
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
            'front_curve': self.front_curve,
            'mean_front_x_locations': self.mean_front_x_locations,
            'mean_front_y_locations': self.mean_front_y_locations,
            'front_velocity': self.front_velocity,
            'front_temperature': self.front_temperature,
            'front_shape_param': self.front_shape_param,
            'target': self.target,
            'time': self.time,
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
            plt.legend(('Actual','Target'),loc='best',fontsize='large')
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
            plt.legend(('Actual','Target'),loc='best',fontsize='large')
            plt.ylim(0.0, 1.5*(np.max(self.target)-273.15))
            plt.xlim(0.0, 1000.0*self.mesh_x_z0[-1,0])
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/trajectory.png", dpi = 500)
            plt.close()

        #Plot actor learning curve
        if(len(self.r_per_episode)!=0):
            plt.clf()
            plt.title("Actor Learning Curve, Episode-Wise",fontsize='xx-large')
            plt.xlabel("Episode",fontsize='large')
            plt.ylabel("Average Reward per Simulation Step",fontsize='large')
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
            plt.ylabel("MSE Loss",fontsize='large')
            plt.plot([*range(len(self.value_error))],self.value_error,lw=2.5,c='r')
            plt.yscale("log")
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/critic_learning.png", dpi = 500)
            plt.close()

        # Plot x rate stdev curve
        if(len(self.x_rate_stdev)!=0):
            plt.clf()
            plt.title("Laser X Position Rate Stdev",fontsize='xx-large')
            plt.xlabel("Episode",fontsize='large')
            plt.ylabel("Laser X Position Rate Stdev [mm/s]",fontsize='large')
            plt.plot([*range(len(self.x_rate_stdev))],1000.0*np.array(self.x_rate_stdev),lw=2.5,c='r')
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/x_rate_stdev.png", dpi = 500)
            plt.close()

        # Plot y rate stdev curve
        if(len(self.y_rate_stdev)!=0):
            plt.clf()
            plt.title("Laser Y Position Rate Stdev",fontsize='xx-large')
            plt.xlabel("Episode",fontsize='large')
            plt.ylabel("Laser Y Position Rate Stdev [mm/s]",fontsize='large')
            plt.plot([*range(len(self.y_rate_stdev))],1000.0*np.array(self.y_rate_stdev),lw=2.5,c='r')
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/y_rate_stdev.png", dpi = 500)
            plt.close()

        # Plot magnitude stdev curve
        if(len(self.mag_stdev)!=0):
            plt.clf()
            plt.title("Laser Magnitude Rate Stdev",fontsize='xx-large')
            plt.xlabel("Episode",fontsize='large')
            plt.ylabel('Laser Magnitude Rate Stdev [KW/m^2-s]',fontsize='large')
            plt.plot([*range(len(self.mag_stdev))],0.001*np.array(self.mag_stdev),lw=2.5,c='r')
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/mag_rate_stdev.png", dpi = 500)
            plt.close()
    
    def render(self):
        print("Rendering simulation results...")
        
        # Make videos of the best temperature field trajecotry and cure field trajectories as function of time
        min_temp = min(10.0*np.floor((np.min(self.temperature_field)-273.15)/10.0), 10.0*np.floor((np.min(self.fine_temperature_field)-273.15)/10.0))
        max_temp = max(10.0*np.ceil((np.max(self.temperature_field)-273.15)/10.0), 10.0*np.floor((np.min(self.fine_temperature_field)-273.15)/10.0))
        min_front_temp = np.min(self.fine_temperature_field[:,:,len(self.fine_temperature_field[0][0])//2])
        max_front_temp = np.max(self.fine_temperature_field[:,:,len(self.fine_temperature_field[0][0])//2])
        
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
            
            # Plot temperature
            c0 = ax0.pcolormesh(1000.0*self.mesh_x_z0, 1000.0*self.mesh_y_z0, self.temperature_field[curr_step,:,:]-273.15, shading='gouraud', cmap='jet', vmin=min_temp, vmax=max_temp)
            ax0.pcolormesh(1000.0*fine_mesh_x, 1000.0*fine_mesh_y, self.fine_temperature_field[curr_step,:,:]-273.15, shading='gouraud', cmap='jet', vmin=min_temp, vmax=max_temp)
            cbar0 = fig.colorbar(c0, ax=ax0)
            cbar0.set_label('Temperature [C]',labelpad=20,fontsize='large')
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
        
            # Determine front locations
            front_x_location = self.front_curve[curr_step][0]
            front_y_location = self.front_curve[curr_step][1]
            front_x_location = front_x_location[front_x_location >= 0.0]
            front_y_location = front_y_location[front_y_location >= 0.0]
            front_x_location = 1000.0*front_x_location
            front_y_location = 1000.0*front_y_location
            front_instances = len(front_x_location)
        
            # Plot input
            c2 = ax2.pcolormesh(1000.0*self.mesh_x_z0, 1000.0*self.mesh_y_z0, 1.0e-3*input_mesh, shading='gouraud', cmap='coolwarm', vmin=0.0, vmax=1.0e-3*self.max_input_mag)
            if front_instances != 0:
                ax2.plot(front_x_location, front_y_location, 's', color='black', markersize=2.25, markeredgecolor='black')
            ax2.axvline(1000.0*self.fine_mesh_loc[curr_step][0], color='red', alpha=0.70, ls='--')
            ax2.axvline(1000.0*self.fine_mesh_loc[curr_step][1], color='red', alpha=0.70, ls='--')
            cbar2 = fig.colorbar(c2, ax=ax2)
            cbar2.set_label('Input Heat [KW/m^2]',labelpad=20,fontsize='large')
            cbar2.ax.tick_params(labelsize=12)
            ax2.set_xlabel('X Position [mm]',fontsize='large')
            ax2.set_ylabel('Y Position [mm]',fontsize='large')
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
            ax2.set_ylabel("Normalized Temperature [-]",fontsize='large',color='b')
            ax2.plot(1000.0*fine_x_coords, (fine_front_temp_curve-min_front_temp)/(max_front_temp-min_front_temp), c='b', lw=2.5)  
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
            
if __name__ == "__main__":
    
    with open("results/PPO_1/output", 'rb') as file:
        data = pickle.load(file)