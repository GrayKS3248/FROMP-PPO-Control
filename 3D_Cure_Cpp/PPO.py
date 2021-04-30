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
from scipy import signal
from scipy.stats import multivariate_normal

class Agent:

    def __init__(self, steps_per_trajectory, trajectories_per_batch, epochs_per_batch, 
                 gamma, lamb, epsilon, alpha, decay_rate, autoencoder_path):

        # Batch memory
        self.states = [[]]
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
        print("\n")
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
    def get_greedy_action(self, state):

        # Get the gaussian distribution parameters used to sample the action for the old and new policy
        with torch.no_grad():
            # Format input state
            state = torch.tensor(state)
            state = state.reshape(1,1,state.shape[0],state.shape[1]).float()
            state = state.to(self.device)
            
            # Forward propogate formatted state
            means, stdev_1, stdev_2, stdev_3 = self.actor.forward(state)
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
    def get_action(self, state):

        # Get the gaussian distribution parameters used to sample the action for the old and new policy
        with torch.no_grad():
            # Format input state
            state = torch.tensor(state)
            state = state.reshape(1,1,state.shape[0],state.shape[1]).float()
            state = state.to(self.device)
            
            # Forward propogate formatted state
            means, stdev_1, stdev_2, stdev_3 = self.actor.forward(state)
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
    def update_agent(self, state, action_1, action_2, action_3, reward):

        # Update the state, action, and reward memory
        self.states[-1].append(state)
        self.actions[-1].append([action_1, action_2, action_3])
        self.rewards[-1].append(reward)
        
        # Get the current (will become the old during learning) log probs
        with torch.no_grad():
            # Convert the state type
            state = torch.tensor(state)
            state = state.reshape(1,1,state.shape[0],state.shape[1]).float()
            state = state.to(self.device)
            
            # Calculate the distributions provided by actor
            means, stdev_1, stdev_2, stdev_3 = self.actor.forward(state)
            means = means.squeeze().to('cpu')
            stdevs = torch.tensor([stdev_1.to('cpu'), stdev_2.to('cpu'), stdev_3.to('cpu')])
            
            # Get log prob of actions selected
            actions = torch.tensor([action_1, action_2, action_3])
            dist = torch.distributions.normal.Normal(means, stdevs)
            self.old_log_probs[-1].append(dist.log_prob(actions).sum())
            
            # Gather current value estimates. Will be used for advantage estimate and value target calculations
            self.value_targets[-1].append(self.critic.forward(state).item())

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
                    self.actions = torch.reshape(torch.tensor(self.actions, dtype=torch.double), (self.steps_per_batch,3)).to(self.device)
                    self.rewards = torch.reshape(torch.tensor(self.rewards, dtype=torch.double), (-1,))
                    self.old_log_probs = torch.reshape(torch.tensor(self.old_log_probs, dtype=torch.double), (-1,)).to(self.device)
                    self.value_targets = torch.reshape(torch.tensor(self.value_targets, dtype=torch.double), (-1,)).to(self.device)
                    self.advantage_estimates = torch.reshape(torch.tensor(self.advantage_estimates, dtype=torch.double), (-1,)).to(self.device)
                    self.advantage_estimates = (self.advantage_estimates - torch.mean(self.advantage_estimates)) / torch.std(self.advantage_estimates)
                
                # Actor optimization
                for i in range(self.epochs_per_batch):
                    self.actor_optimizer.zero_grad()
                    means, stdev_1, stdev_2, stdev_3 = self.actor.forward(self.states)
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
                    value_estimates = self.critic.forward(self.states).double()
                    loss = torch.nn.MSELoss()(value_estimates[:, 0], self.value_targets)
                    with torch.no_grad():
                        critic_losses.append(loss.item())
                    loss.backward()
                    self.critic_optimizer.step()
                    self.actor_lr_scheduler.step()
                
                # After learning, reset the memory
                self.states = [[]]
                self.actions = [[]]
                self.rewards = [[]]
                self.old_log_probs = [[]]
                self.value_targets = [[]]
                self.advantage_estimates = [[]]
                
                return critic_losses
        
        return []
        
if __name__ == "__main__":
    agent = Agent(100, 20, 20, 0.99, 0.95, 0.20, 1e-3, 0.998, "results/ks3_obj1_bn64_U")
    
    random_state = np.random.rand(360,40)
    a1, s1, a2, s2, a3, s3 = agent.get_action(random_state)
    
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
        self.front_location = []
        self.front_velocity = []
        self.front_temperature = []
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
    
    def store_field_history(self, temperature_field, cure_field):
        self.temperature_field = np.array(temperature_field)
        self.cure_field = np.array(cure_field)
    
    def store_front_history(self, front_location, front_velocity, front_temperature):
        self.front_location = np.array(front_location)
        self.front_velocity = np.array(front_velocity)
        self.front_temperature = np.array(front_temperature)
    
    def store_target_and_time(self, target, time):
        self.target = np.array(target)
        self.time = np.array(time)
    
    def store_top_mesh(self, mesh_x_z0, mesh_y_z0):
        self.mesh_x_z0 = np.array(mesh_x_z0)
        self.mesh_y_z0 = np.array(mesh_y_z0)
    
    def store_left_mesh(self, mesh_y_x0, mesh_z_x0):
        self.mesh_y_x0 = np.array(mesh_y_x0)
        self.mesh_z_x0 = np.array(mesh_z_x0)
    
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
            video_path = "results/PPO_"+str(curr_folder)+"/video/"
            if not os.path.isdir(path):
                os.mkdir(path)
                os.mkdir(video_path)
                done = True
            else:
                curr_folder = curr_folder + 1
                
        # Store save paths
        self.path = path
        self.video_path = video_path
        
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
            'front_location': self.front_location,
            'front_velocity': self.front_velocity,
            'front_temperature': self.front_temperature,
            'target': self.target,
            'time': self.time,
            'mesh_x_z0' : self.mesh_x_z0,
            'mesh_y_z0' : self.mesh_y_z0,
            'max_input_mag' : self.max_input_mag,
            'exp_const' : self.exp_const,
            'mesh_y_x0' : self.mesh_y_x0,
            'mesh_z_x0' : self.mesh_z_x0,
            'control_speed' : self.control_speed,
            'actor': agent.actor,
            'critic': agent.critic
        }
        
        # Save the stored data
        with open(self.path + "/output", 'wb') as file:
            pickle.dump(data, file)
    
    def plot(self):
        print("Plotting agent results...")
    
        # Plot the trajectory
        if self.control_speed:
            # Plot speed trajectory
            plt.clf()
            plt.title("Front Velocity",fontsize='xx-large')
            plt.xlabel("Simulation Time [s]",fontsize='large')
            plt.ylabel("Front Velocity [mm/s]",fontsize='large')
            plt.plot(self.time, 1000.0*(np.mean(self.front_velocity,axis=(0,1))),c='r',lw=2.5)
            plt.plot(self.time, 1000.0*(self.target),c='b',ls='--',lw=2.5)
            plt.legend(('Actual','Target'),loc='best',fontsize='large')
            plt.ylim(0.0, 1500.0*np.max(self.target))
            plt.xlim(0.0, np.round(self.time[-1]))
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/trajectory.png", dpi = 500)
            plt.close()
            
            # Plot front temperature trajectory
            plt.clf()
            plt.title("Front Temperature",fontsize='xx-large')
            plt.xlabel("Location [mm]",fontsize='large')
            plt.ylabel("Front Temperature [C]",fontsize='large')
            plt.plot(np.mean(1000.0*np.array(self.front_location),axis=(0,1)), (np.mean(self.front_temperature,axis=(0,1))-273.15),c='r',lw=2.5)
            plt.ylim(0.0, np.max(1.025*(np.mean(self.front_temperature,axis=(0,1))-273.15)))
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
            plt.plot(self.time, 1000.0*(np.mean(self.front_velocity,axis=(0,1))),c='r',lw=2.5)
            plt.ylim(0.0, np.max(1025.0*np.array(np.mean(self.front_velocity,axis=(0,1)))))
            plt.xlim(0.0, np.round(self.time[-1]))
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/speed.png", dpi = 500)
            plt.close()
            
            # Plot front temperature trajectory
            plt.clf()
            plt.title("Front Temperature",fontsize='xx-large')
            plt.xlabel("Location [mm]",fontsize='large')
            plt.ylabel("Front Temperature [C]",fontsize='large')
            plt.plot(np.mean(1000.0*np.array(self.front_location),axis=(0,1)), (np.mean(self.front_temperature,axis=(0,1))-273.15),c='r',lw=2.5)
            plt.plot(np.mean(1000.0*np.array(self.front_location),axis=(0,1)), (self.target-273.15),c='b',ls='--',lw=2.5)
            plt.legend(('Actual','Target'),loc='best',fontsize='large')
            plt.ylim(0.0, 1.5*(np.max(self.target)-273.15))
            plt.xlim(0.0, 1000.0*self.mesh_x_z0[-1,0])
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig(self.path + "/trajectory.png", dpi = 500)
            plt.close()

        #Plot actor learning curve
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
        print("Rendering agent results...")
        
        # Make videos of the best temperature field trajecotry and cure field trajectories as function of time
        min_temp = 10.0*np.floor((np.min(self.temperature_field)-273.15)/10.0)
        max_temp = 10.0*np.ceil((np.max(self.temperature_field)-273.15)/10.0)
        
        # Determine front shape deltas
        front_mean_loc = np.mean(1000.0*np.array(self.front_location),axis=(0,1))
        min_loc = 0.5*np.floor((np.min(np.min(1000.0*np.array(self.front_location),axis=(0,1)) - front_mean_loc))/0.5)
        max_loc = 0.5*np.ceil((np.max(np.max(1000.0*np.array(self.front_location),axis=(0,1)) - front_mean_loc))/0.5)
        
        # Determine front speed deltas
        max_vel = 0.5*np.ceil((np.max(1000.0*self.front_velocity))/0.5)
        
        # Determine radius of convolution
        radius_of_conv = int(np.round(len(self.mesh_y_x0)*len(self.mesh_y_x0[0])/100)*2.0-1.0)
        
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
        
        	# Plot temperature
        	c0 = ax0.pcolormesh(1000.0*self.mesh_x_z0, 1000.0*self.mesh_y_z0, self.temperature_field[:,:,curr_step]-273.15, shading='gouraud', cmap='jet', vmin=min_temp, vmax=max_temp)
        	cbar0 = fig.colorbar(c0, ax=ax0)
        	cbar0.set_label('Temperature [C]',labelpad=20,fontsize='large')
        	cbar0.ax.tick_params(labelsize=12)
        	ax0.set_xlabel('X Position [mm]',fontsize='large')
        	ax0.set_ylabel('Y Position [mm]',fontsize='large')
        	ax0.tick_params(axis='x',labelsize=12)
        	ax0.tick_params(axis='y',labelsize=12)
        	ax0.set_aspect('equal', adjustable='box')
        	ax0.set_title('Max Temperature = '+'{:.2f}'.format(np.max(self.temperature_field[:,:,curr_step]-273.15))+' C',fontsize='large')
        
        	# Plot cure
        	c1 = ax1.pcolormesh(1000.0*self.mesh_x_z0, 1000.0*self.mesh_y_z0, self.cure_field[:,:,curr_step], shading='gouraud', cmap='YlOrBr', vmin=0.0, vmax=1.0)
        	cbar1 = fig.colorbar(c1, ax=ax1)
        	cbar1.set_label('Degree Cure [-]', labelpad=20,fontsize='large')
        	cbar1.ax.tick_params(labelsize=12)
        	ax1.set_xlabel('X Position [mm]',fontsize='large')
        	ax1.set_ylabel('Y Position [mm]',fontsize='large')
        	ax1.tick_params(axis='x',labelsize=12)
        	ax1.tick_params(axis='y',labelsize=12)
        	ax1.set_aspect('equal', adjustable='box')
        
        	# Plot input
        	c2 = ax2.pcolormesh(1000.0*self.mesh_x_z0, 1000.0*self.mesh_y_z0, 1.0e-3*input_mesh, shading='gouraud', cmap='coolwarm', vmin=0.0, vmax=1.0e-3*self.max_input_mag)
        	ax2.plot(1000.0*self.front_location[:,0,curr_step].squeeze(), 1000.0*self.mesh_y_z0[0,:], 'k-', lw=1.5)
        	cbar2 = fig.colorbar(c2, ax=ax2)
        	cbar2.set_label('Input Heat [KW/m^2]',labelpad=20,fontsize='large')
        	cbar2.ax.tick_params(labelsize=12)
        	ax2.set_xlabel('X Position [mm]',fontsize='large')
        	ax2.set_ylabel('Y Position [mm]',fontsize='large')
        	ax2.tick_params(axis='x',labelsize=12)
        	ax2.tick_params(axis='y',labelsize=12)
        	ax2.set_aspect('equal', adjustable='box')
        
        	# Set title and save
        	title_str = "Time From Trigger: "+'{:.2f}'.format(self.time[curr_step])+'s'
        	fig.suptitle(title_str,fontsize='xx-large')
        	plt.savefig(self.video_path+str(curr_step).zfill(4)+'.png', dpi=100)
        	plt.close()
        
        	# Make fig for front location and velocity
        	plt.cla()
        	plt.clf()
        	fig, (ax0, ax1) = plt.subplots(1,2,subplot_kw={"projection": "3d"})
        	fig.set_size_inches(14.0,8.0)
        
        	# Convolve front location data
        	back_msaa_index = np.clip(curr_step-5,0,len(self.time)-1)
        	front_msaa_index = np.clip(curr_step+5,0,len(self.time)-1)
        	front_delta_loc = np.mean(1000.0*np.array(self.front_location[:,:,back_msaa_index:front_msaa_index]),axis=2) - np.mean(front_mean_loc[back_msaa_index:front_msaa_index])
        	front_delta_min = np.min(front_delta_loc)
        	front_delta_max = np.max(front_delta_loc)
        	if not ((front_delta_loc<=1.0e-4).all() and (front_delta_loc>=-1.0e-4).all()):
        		x,y=np.meshgrid(np.linspace(-1,1,radius_of_conv),np.linspace(-1,1,radius_of_conv))
        		win=multivariate_normal.pdf(np.dstack((x,y)),mean=[0,0],cov=[[1.0,0.0],[0.0,1.0]])
        		padded = front_delta_loc
        		for i in range(int((radius_of_conv+1)/2)-1):
        			padded = np.append(padded[:,0].reshape(len(padded[:,0]),1),padded,axis=1)
        			padded = np.append(padded[0,:].reshape(1,len(padded[0,:])),padded,axis=0)
        			padded = np.append(padded,padded[:,-1].reshape(len(padded[:,-1]),1),axis=1)
        			padded = np.append(padded,padded[-1,:].reshape(1,len(padded[-1,:])),axis=0)
        		out = signal.convolve2d(padded,win,mode='valid')
        		out=out*((front_delta_max-front_delta_min)/(np.max(out)-np.min(out)))
        		out=out-np.mean(out)
        	else:
        		out = front_delta_loc
        
        	# Plot front location
        	ax0.plot_surface(1000.0*self.mesh_y_x0, 1000.0*self.mesh_z_x0,out,cmap='coolwarm',vmin=min_loc,vmax=max_loc,alpha=1.0)
        	ax0.set_xlabel('Y Position [mm]',fontsize='large',labelpad=15)
        	ax0.set_ylabel('Z Position [mm]',fontsize='large',labelpad=15)
        	ax0.set_zlabel('Lengthwise Delta [mm]',fontsize='large',labelpad=20)
        	ax0.tick_params(axis='x',labelsize=12,pad=10)
        	ax0.tick_params(axis='y',labelsize=12,pad=10)
        	ax0.tick_params(axis='z',labelsize=12,pad=10)
        	ax0.set_zlim(min_loc,max_loc)
        	ax0.set_title("Front Shape",fontsize='xx-large')
        
        	# Covolve front speed data
        	back_msaa_index = np.clip(curr_step-5,0,len(self.time)-1)
        	front_msaa_index = np.clip(curr_step+5,0,len(self.time)-1)
        	curr_front_vel = np.mean(1000.0*np.array(self.front_velocity[:,:,back_msaa_index:front_msaa_index]),axis=2)
        	front_vel_min = np.min(curr_front_vel)
        	front_vel_max = np.max(curr_front_vel)
        	if not ((curr_front_vel<=1.0e-4).all() and (curr_front_vel>=-1.0e-4).all()):
        		x,y=np.meshgrid(np.linspace(-1,1,radius_of_conv),np.linspace(-1,1,radius_of_conv))
        		win=multivariate_normal.pdf(np.dstack((x,y)),mean=[0,0],cov=[[1.0,0.0],[0.0,1.0]])
        		padded = curr_front_vel
        		for i in range(int((radius_of_conv+1)/2)-1):
        			padded = np.append(padded[:,0].reshape(len(padded[:,0]),1),padded,axis=1)
        			padded = np.append(padded[0,:].reshape(1,len(padded[0,:])),padded,axis=0)
        			padded = np.append(padded,padded[:,-1].reshape(len(padded[:,-1]),1),axis=1)
        			padded = np.append(padded,padded[-1,:].reshape(1,len(padded[-1,:])),axis=0)
        		out = signal.convolve2d(padded,win,mode='valid')
        		out=out*((front_vel_max-front_vel_min)/(np.max(out)-np.min(out)))
        		out=out-np.mean(out)+np.mean(curr_front_vel)
        	else:
        		out = curr_front_vel
        
        	# Plot front speed
        	ax1.plot_surface(1000.0*self.mesh_y_x0,1000.0*self.mesh_z_x0,out,cmap='coolwarm',vmin=0.0,vmax=max_vel,alpha=1.0)
        	ax1.set_xlabel('Y Position [mm]',fontsize='large',labelpad=15)
        	ax1.set_ylabel('Z Position [mm]',fontsize='large',labelpad=15)
        	ax1.set_zlabel('Front Speed [mm/s]',fontsize='large',labelpad=20)
        	ax1.tick_params(axis='x',labelsize=12,pad=10)
        	ax1.tick_params(axis='y',labelsize=12,pad=10)
        	ax1.tick_params(axis='z',labelsize=12,pad=10)
        	ax1.set_zlim(0.0,max_vel)
        	ax1.set_title("Front Speed",fontsize='xx-large')
        
        	# Set title and save
        	title_str = "Time From Trigger: "+'{:.2f}'.format(self.time[curr_step])+'s'
        	fig.suptitle(title_str,fontsize='xx-large')
        	plt.savefig(self.video_path+"f_"+str(curr_step).zfill(4)+'.png', dpi=100)
        	plt.close()