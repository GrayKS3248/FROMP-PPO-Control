# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:41:07 2020

@author: Grayson Schaer
"""

import numpy as np
import torch
from MLP_1_Output import Model as mlp_1
from MLP_3_Output import Model as mlp_3
import pickle
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import multivariate_normal

class Agent:

    def __init__(self, num_states, steps_per_trajectory, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, alpha, decay_rate, load_agent, reset_stdev, path):

        # Policy and value estimation network
        # Input is the state
        # Output is the mean of the gaussian distribution from which actions are sampled
        self.actor = mlp_3(num_inputs=num_states, num_outputs=3, num_hidden_layers=2, num_neurons_in_layer=160)
        self.actor_optimizer =  torch.optim.Adam(self.actor.parameters() , lr=alpha)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.actor_optimizer, gamma=decay_rate)

        # Old policy and value estimation network used to calculate clipped surrogate objective
        # Input is the state
        # Output is the mean of the gaussian distribution from which actions are sampled
        self.old_actor = mlp_3(num_inputs=num_states, num_outputs=3, num_hidden_layers=2, num_neurons_in_layer=160)
        self.old_actor.load_state_dict(self.actor.state_dict())

        # Critic NN that estimates the value function
        # Input is the state
        # Output is the estimated value of that state
        self.critic =  mlp_1(num_inputs=num_states, num_outputs=1, num_hidden_layers=2, num_neurons_in_layer=160)
        self.critic_optimizer =  torch.optim.Adam(self.critic.parameters() , lr=alpha)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.critic_optimizer, gamma=decay_rate)

        # Trajectory memory
        self.num_states = num_states
        self.steps_per_trajectory = steps_per_trajectory
        self.trajectory_index  = 0
        self.trajectories_per_batch = trajectories_per_batch
        self.trajectory_states = np.zeros((trajectories_per_batch*steps_per_trajectory, num_states))
        self.trajectory_actions = np.zeros((3, trajectories_per_batch*steps_per_trajectory))
        self.trajectory_rewards = np.zeros(trajectories_per_batch*steps_per_trajectory)

        # Hyperparameters
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        self.gamma = gamma
        self.lamb = lamb
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.gamma_lamb_reduction_array= np.zeros(steps_per_trajectory)
        for curr_step in range(steps_per_trajectory):
            self.gamma_lamb_reduction_array[curr_step] = (self.gamma * self.lamb) ** (curr_step)

        # Training memory
        self.value_estimation_error = []
        
        # Copy previous agent
        if load_agent == 1:
            self.copy(reset_stdev, path)

    # Copies the actor and critic NNs from another agent to this agent
    # @param reset_stdev - boolean that determines if the stdev of the agent is reset or not
    # @param path - the path from which the previous agent is loaded
    def copy(self, reset_stdev, path):
        # Find load path
        print("\nLoading: " + path + "\n")
        if not os.path.isdir(path):
            raise RuntimeError("Could not find previous folder: " + path)
        path = path + "/output"
        if not os.path.exists(path):
            raise RuntimeError("Could not find previous output file: " + path)
        with open(path, 'rb') as file:
            input_data = pickle.load(file)
        if 'agent' in input_data:
            old_agent = input_data['agent']
        else:
            old_agent = input_data['logbook']['agents'][0]
        
        # Copy the actor NN
        self.actor.load_state_dict(old_agent.actor.state_dict())

        # Reset the stdev
        if reset_stdev:
            self.actor.stdev_1 = torch.nn.Parameter(2.0 * torch.ones(1, dtype=torch.double).double())
            self.actor.stdev_2 = torch.nn.Parameter(2.0 * torch.ones(1, dtype=torch.double).double())
            self.actor.stdev_3 = torch.nn.Parameter(2.0 * torch.ones(1, dtype=torch.double).double())

        # Build the optimizer
        self.actor_optimizer =  torch.optim.Adam(self.actor.parameters() , lr=self.alpha)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.actor_optimizer, gamma=self.decay_rate)

        # Copy the old actor NN
        self.old_actor.load_state_dict(self.actor.state_dict())

        # Copy the critic NN
        self.critic.load_state_dict(old_agent.critic.state_dict())
        self.critic_optimizer =  torch.optim.Adam(self.critic.parameters() , lr=self.alpha)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.critic_optimizer, gamma=self.decay_rate)

    # Clips an input float to the range [min_val, max_val]
    # @ param num - the float to be clipped
    # @ param min_val - the minimal inclusive value num can take
    # @ param max_val - the maximum inclusive value num can take
    # @ return - the clipped float
    def clip_float(self, num, min_val, max_val):
        return max(min(num, max_val), min_val)

    # Calcuates determinisitc action given state and policy.
    # @ param state - The state in which the policy is applied to calculate the action
    # @ return action - The calculated deterministic action based on the state and policy
    def get_greedy_action(self, state):

        # Get the gaussian distribution parameters used to sample the action for the old and new policy
        with torch.no_grad():
            means, stdev_1, stdev_2, stdev_3 = self.actor.forward(torch.tensor(state, dtype=torch.float))

        # Return the means
        action_1 = float(means[0])
        action_2 = float(means[1])
        action_3 = float(means[2])
        return action_1, action_2, action_3, 0.0, 0.0, 0.0

    # Calcuates stochastic action given state and policy.
    # @ param state - The state in which the policy is applied to calculate the action
    # @ return action - The calculated stochastic action based on the state and policy
    # @ return stdev - The calculated stdev based on the policy
    def get_action(self, state):

        # Get the gaussian distribution parameters used to sample the action for the old and new policy
        means, stdev_1, stdev_2, stdev_3 = self.actor.forward(torch.tensor(state, dtype=torch.float))

        # Sample the first action
        dist_1 = torch.distributions.normal.Normal(means[0], stdev_1)
        action_1 = dist_1.sample().item()
        stdev_1 = stdev_1.detach().item()

        # Sample the second action
        dist_2 = torch.distributions.normal.Normal(means[1], stdev_2)
        action_2 = dist_2.sample().item()
        stdev_2 = stdev_2.detach().item()

        # Sample the second action
        dist_3 = torch.distributions.normal.Normal(means[2], stdev_3)
        action_3 = dist_3.sample().item()
        stdev_3 = stdev_3.detach().item()

        return action_1, stdev_1, action_2, stdev_2, action_3, stdev_3


    # Gets the autodifferentiable probability ratios for the given trajectory
    # @ param minibatch_indices - list of indices that represent set of minibatches over which SGD will occur
    # @ return - the differentiable probability ratios
    def get_action_prob_ratio_minibatches(self, minibatch_indices):

        probability_ratio_minibatches = []

        for curr_epoch in range(self.num_epochs):
            # Get the current and old action distribution parameters
            state_minibatch = torch.tensor(self.trajectory_states[minibatch_indices[curr_epoch,:], :], dtype=torch.float)
            means_minibatch, stdev_1, stdev_2, stdev_3 = self.actor.forward(state_minibatch)
            with torch.no_grad():
                old_means_minibatch, old_stdev_1, old_stdev_2, old_stdev_3 = self.old_actor.forward(state_minibatch)

            # Set the distributions based on the parameters above
            dist_1 = torch.distributions.normal.Normal(means_minibatch[:,0], stdev_1)
            dist_2 = torch.distributions.normal.Normal(means_minibatch[:,1], stdev_2)
            dist_3 = torch.distributions.normal.Normal(means_minibatch[:,2], stdev_3)
            old_dist_1 = torch.distributions.normal.Normal(old_means_minibatch[:,0], old_stdev_1)
            old_dist_2 = torch.distributions.normal.Normal(old_means_minibatch[:,1], old_stdev_2)
            old_dist_3 = torch.distributions.normal.Normal(old_means_minibatch[:,2], old_stdev_3)

            # Get the probability ratios
            action_minibatch = torch.tensor(self.trajectory_actions[:,minibatch_indices[curr_epoch,:]], dtype=torch.float)
            numerator_1 = torch.exp(dist_1.log_prob(action_minibatch[0,:])).double()
            numerator_2 = torch.exp(dist_2.log_prob(action_minibatch[1,:])).double()
            numerator_3 = torch.exp(dist_3.log_prob(action_minibatch[2,:])).double()
            denominator_1 =  torch.exp(old_dist_1.log_prob(action_minibatch[0,:])).double()
            denominator_2 =  torch.exp(old_dist_2.log_prob(action_minibatch[1,:])).double()
            denominator_3 =  torch.exp(old_dist_3.log_prob(action_minibatch[2,:])).double()
            probability_ratio = (numerator_1 * numerator_2 * numerator_3) / (denominator_1 * denominator_2 * denominator_3)

            probability_ratio_minibatches.append(probability_ratio)

        return probability_ratio_minibatches


    # Calculates the target value function of a given state at time t based on the TD(0) algorithm
    # @ param minibatch_indices - list of indices that represent set of minibatches over which SGD will occur
    # @ return - value targets for the minibatches
    def get_value_target_minibatches(self, minibatch_indices):

        value_target_minibatches = []
        value_estimates, next_value_estimates = self.get_value_estimate()
        next_value_estimates = next_value_estimates.reshape(self.trajectories_per_batch*self.steps_per_trajectory)

        for curr_epoch in range(self.num_epochs):
            # Calculate V(s_{t+1})
            next_value_estimate_minibatch = next_value_estimates[minibatch_indices[curr_epoch]]

            # Use the TD(0) algorithm to calculate the target value function
            reward_minibatch = torch.tensor(self.trajectory_rewards[minibatch_indices[curr_epoch]])
            with torch.no_grad():
                value_target_minibatch = reward_minibatch + self.gamma * torch.tensor(next_value_estimate_minibatch).double()
            value_target_minibatches.append(value_target_minibatch)

        return value_target_minibatches


    # Estimates the value function of a given state trajectory based on the critic NN
    # @ param minibatch_indices - list of indices that represent set of minibatches over which SGD will occur
    # @ return - value estimation of the stored states and the value estimation of the next stored states
    def get_value_estimate_minibatches(self, minibatch_indices):

        value_estimate_minibatches = []

        for curr_epoch in range(self.num_epochs):
            # Get the value estimation based on the critic NN
            state_minibatch = torch.tensor(self.trajectory_states[minibatch_indices[curr_epoch,:], :], dtype=torch.float)
            value_estimate = self.critic.forward(state_minibatch).double().squeeze()
            value_estimate_minibatches.append(value_estimate)

        return value_estimate_minibatches


    # Estimates the value function of a given state trajectory based on the critic NN
    # @ return - value estimation of the stored states and the value estimation of the stored states + 1
    def get_value_estimate(self):

        # Get the value estimation based on the critic NN
        with torch.no_grad():
            state = torch.tensor(self.trajectory_states, dtype=torch.float)
            value_estimates = self.critic.forward(state).double().numpy().reshape(self.trajectories_per_batch,self.steps_per_trajectory)

        # Get the next value estimate
        next_value_estimates = np.roll(value_estimates, -1, axis=1)
        next_value_estimates[:,-1] = next_value_estimates[:,-2]

        return value_estimates, next_value_estimates


    # Calculates the advantage function sequence for a given trajectory
    # @ param minibatch_indices - list of indices that represent set of minibatches over which SGD will occur
    # @ return - sequence of advantage function values based on GAE algrothim
    def get_advantage_minibatches(self, minibatch_indices):

        # Split rewards and value estimates into their respective trajecotries
        value_estimates, next_value_estimates = self.get_value_estimate()
        rewards = self.trajectory_rewards.reshape(self.trajectories_per_batch,self.steps_per_trajectory)

        # Calculate the deltas
        deltas = rewards + (self.gamma * next_value_estimates) - value_estimates

        # Initialize the advantage array
        advantages = np.zeros((self.trajectories_per_batch,self.steps_per_trajectory))

        # Use deltas to find the advantage function values by GAE
        for curr_step in range(self.steps_per_trajectory):
            deltas_subset = deltas[:, curr_step:]
            gamma_lamb_reduction_subset = self.gamma_lamb_reduction_array[0:(self.steps_per_trajectory - curr_step)]
            advantages[:, curr_step] = np.sum(deltas_subset * gamma_lamb_reduction_subset, axis=1)

        # Format the output
        advantages = torch.tensor(advantages.reshape(self.trajectories_per_batch*self.steps_per_trajectory)).double()
        advantages = (advantages - torch.mean(advantages)) / torch.sqrt(torch.var(advantages))

        advantage_estimate_minibatches = []
        for curr_epoch in range(self.num_epochs):
            advantage_estimate_minibatches.append(advantages[minibatch_indices[curr_epoch]])

        return advantage_estimate_minibatches


    # Updates the trajectory memory given an arbitrary time step
    # @ param state - state to be added to trajectory memory
    # @ param action_1 - first action to be added to trajectory memory
    # @ param action_2 - second action to be added to trajectory memory
    # @ param action_3 - third action to be added to trajectory memory
    # @ param reward - reward to be added to trajectory memory
    def update_agent(self, state, action_1, action_2, action_3, reward):

        # Calculate the location in the memory matrices to insert new data
        current_trajectory = self.trajectory_index // self.steps_per_trajectory
        current_step = self.trajectory_index - current_trajectory * self.steps_per_trajectory

        # Update the state, action, and reward memory
        self.trajectory_states[self.trajectory_index,:] = state
        self.trajectory_actions[:, self.trajectory_index] = np.array([action_1, action_2, action_3])
        self.trajectory_rewards[self.trajectory_index] = reward

        # Update the trajectory index
        self.trajectory_index = self.trajectory_index + 1

        # If a trajectory batch is complete, apply the learning algorithm
        if current_step == self.steps_per_trajectory - 1 and current_trajectory == self.trajectories_per_batch - 1:
            
            # Define the minibatches to be used for minibatch SGD
            minibatch_indices = np.random.permutation(len(self.trajectory_actions[0,:])).astype(int).reshape(self.num_epochs, self.minibatch_size)
            
            # Calculate the value estimates and targets in minibatches
            value_estimate_minibatches = self.get_value_estimate_minibatches(minibatch_indices)
            value_target_minibatches =  self.get_value_target_minibatches(minibatch_indices)
            
            # Calculate the advantage estimates and probability ratios in minibatches
            advantage_estimate_minibatches = self.get_advantage_minibatches(minibatch_indices)
            probability_ratio_minibatches = self.get_action_prob_ratio_minibatches(minibatch_indices)
            
            # Copy the current state dictionary to the old actor and let the actor and critic learn
            self.old_actor.load_state_dict(self.actor.state_dict())
            self.learn(advantage_estimate_minibatches, probability_ratio_minibatches, value_estimate_minibatches, value_target_minibatches)
            
            # After learning, reset the memory
            self.trajectory_index  = 0
            self.trajectory_states = np.zeros((self.trajectories_per_batch*self.steps_per_trajectory, self.num_states))
            self.trajectory_actions = np.zeros((3, self.trajectories_per_batch*self.steps_per_trajectory))
            self.trajectory_rewards = np.zeros(self.trajectories_per_batch*self.steps_per_trajectory)
        
        # Give indication that update was successful
        return True;


    # Updates the actor's NN weights based on a batch of trajectories
    # @ param advantage_estimate_minibatches - non differentiable minibatch set of advantage estimates
    # @ param probability_ratio_minibatches - differentiable minibatch set of action probability ratios
    # @ param value_estimate_minibatches - differentiable minibatch set of value estimates
    # @ param value_target_minibatches - non differentiable minibatch set of value targets
    def learn(self, advantage_estimate_minibatches, probability_ratio_minibatches, value_estimate_minibatches, value_target_minibatches):

        actor_loss = 0.0
        critic_loss = 0.0

        for curr_epoch in range(self.num_epochs):
            # Apply the clipped surrogate objective algorithm
            surrogate_objective = (advantage_estimate_minibatches[curr_epoch] * probability_ratio_minibatches[curr_epoch])
            clipped_objective = (probability_ratio_minibatches[curr_epoch].clamp(1-self.epsilon, 1+self.epsilon) * advantage_estimate_minibatches[curr_epoch])

            # Define CSO
            clipped_surrogate_objective = torch.min(surrogate_objective, clipped_objective)

            # Calculate the loss function for the value estimation
            squared_error_loss = (value_target_minibatches[curr_epoch] - value_estimate_minibatches[curr_epoch])**2

            # Calculate the total loss function
            actor_loss = actor_loss - (clipped_surrogate_objective).mean()
            critic_loss = critic_loss + (squared_error_loss).mean()

        self.value_estimation_error.append(critic_loss.item())

        # Conduct minibatch stochastic gradient descent on actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()

        # Conduct minibatch stochastic gradient descent on critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()
        
class Save_Plot_Render:

    def __init__(self):
        
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
        self.best_reward = []
        self.mesh_x_z0 = []
        self.mesh_y_z0 = []
        self.mesh_y_x0 = []
        self.mesh_z_x0 = []
        self.max_input_mag = []
        self.exp_const = []
        self.control_speed = []
        self.render = []
    
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
    
    def store_target_time_and_best(self, target, time, best_reward):
        self.target = np.array(target)
        self.time = np.array(time)
        self.best_reward = best_reward
    
    def store_top_mesh(self, mesh_x_z0, mesh_y_z0):
        self.mesh_x_z0 = np.array(mesh_x_z0)
        self.mesh_y_z0 = np.array(mesh_y_z0)
    
    def store_left_mesh(self, mesh_y_x0, mesh_z_x0):
        self.mesh_y_x0 = np.array(mesh_y_x0)
        self.mesh_z_x0 = np.array(mesh_z_x0)
    
    def store_input_params(self, max_input_mag, exp_const):
        self.max_input_mag = max_input_mag
        self.exp_const = exp_const
    
    def store_options(self, control_speed, render):
        self.control_speed = (control_speed == 1)
        self.render = (render == 1)
    
    def save(self, agent):
        print("Saving agent results...")
        
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
            'best_reward': self.best_reward,
            'mesh_x_z0' : self.mesh_x_z0,
            'mesh_y_z0' : self.mesh_y_z0,
            'max_input_mag' : self.max_input_mag,
            'exp_const' : self.exp_const,
            'mesh_y_x0' : self.mesh_y_x0,
            'mesh_z_x0' : self.mesh_z_x0,
            'control_speed' : self.control_speed,
			'render': self.render,
        }
        
        # Save the stored data
        output = { 'data':data, 'agent':agent }
        with open(self.path + "/output", 'wb') as file:
            pickle.dump(output, file)
    
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
        plt.ylabel("Laser X Position Rate Stdev [m/s]",fontsize='large')
        plt.plot([*range(len(self.x_rate_stdev))],np.array(self.x_rate_stdev),lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        plt.savefig(self.path + "/x_rate_stdev.png", dpi = 500)
        plt.close()

        # Plot y rate stdev curve
        plt.clf()
        plt.title("Laser Y Position Rate Stdev",fontsize='xx-large')
        plt.xlabel("Episode",fontsize='large')
        plt.ylabel("Laser Y Position Rate Stdev [m/s]",fontsize='large')
        plt.plot([*range(len(self.y_rate_stdev))],np.array(self.y_rate_stdev),lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        plt.savefig(self.path + "/y_rate_stdev.png", dpi = 500)
        plt.close()

        # Plot magnitude stdev curve
        plt.clf()
        plt.title("Laser Magnitude Stdev",fontsize='xx-large')
        plt.xlabel("Episode",fontsize='large')
        plt.ylabel('Laser Magnitude Stdev [K/s]',fontsize='large')
        plt.plot([*range(len(self.mag_stdev))],np.array(self.mag_stdev),lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        plt.savefig(self.path + "/mag_stdev.png", dpi = 500)
        plt.close()
    
    def render(self):
        print("Rendering agent results...")
        
        # Make videos of the best temperature field trajecotry and cure field trajectories as function of time
        if self.render:
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