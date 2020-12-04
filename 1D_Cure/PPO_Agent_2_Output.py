# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:41:07 2020

@author: Grayson Schaer, gschaer2
"""

import numpy as np
import torch
import NN_Stdev_2_Output
import NN

class PPO_Agent:
    
    def __init__(self, num_states, steps_per_trajectory, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, alpha):

        # Policy and value estimation network
        # Input is the state
        # Output is the mean of the gaussian distribution from which actions are sampled
        self.actor = NN_Stdev_2_Output.Neural_Network(num_inputs=num_states, num_outputs=2, num_hidden_layers=2, num_neurons_in_layer=128)
        self.actor_optimizer =  torch.optim.Adam(self.actor.parameters() , lr=alpha)
        
        # Old policy and value estimation network used to calculate clipped surrogate objective
        # Input is the state
        # Output is the mean of the gaussian distribution from which actions are sampled
        self.old_actor = NN_Stdev_2_Output.Neural_Network(num_inputs=num_states, num_outputs=2, num_hidden_layers=2, num_neurons_in_layer=128)
        self.old_actor.load_state_dict(self.actor.state_dict())
                
        # Critic NN that estimates the value function
        # Input is the state
        # Output is the estimated value of that state
        self.critic =  NN.Neural_Network(num_inputs=num_states, num_outputs=1, num_hidden_layers=2, num_neurons_in_layer=128)
        self.critic_optimizer =  torch.optim.Adam(self.critic.parameters() , lr=alpha)
        
        # Trajectory memory
        self.num_states = num_states
        self.steps_per_trajectory = steps_per_trajectory
        self.trajectory_index  = 0
        self.trajectories_per_batch = trajectories_per_batch
        self.trajectory_states = np.zeros((trajectories_per_batch*steps_per_trajectory, num_states))
        self.trajectory_actions = np.zeros((2, trajectories_per_batch*steps_per_trajectory))
        self.trajectory_rewards = np.zeros(trajectories_per_batch*steps_per_trajectory)
        
        # Hyperparameters
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        self.gamma = gamma
        self.lamb = lamb
        self.epsilon = epsilon
        self.gamma_lamb_reduction_array= np.zeros(steps_per_trajectory)
        for curr_step in range(steps_per_trajectory):
            self.gamma_lamb_reduction_array[curr_step] = (self.gamma * self.lamb) ** (curr_step)
        
        # Training memory
        self.value_estimation_error = []
        
    # Clips an input float to the range [min_val, max_val]
    # @ param num - the float to be clipped
    # @ param min_val - the minimal inclusive value num can take
    # @ param max_val - the maximum inclusive value num can take
    # @ return - the clipped float
    def clip_float(self, num, min_val, max_val):
        return max(min(num, max_val), min_val)
    
    
    # Calcuates action given state and policy.
    # @ param state - The state in which the policy is applied to calculate the action
    # @ return action - The calculated action based on the state and policy
    # @ return stdev - The calculated stdev based on the policy
    def get_action(self, state):
        
        # Get the gaussian distribution parameters used to sample the action for the old and new policy
        means, stdev_1, stdev_2 = self.actor.forward(torch.tensor(state, dtype=torch.float))
        
        # Sample the first action
        dist_1 = torch.distributions.normal.Normal(means[0], stdev_1)
        action_1 = dist_1.sample().item()
        stdev_1 = stdev_1.detach().item()
        
        # Sample the second action
        dist_2 = torch.distributions.normal.Normal(means[1], stdev_2)
        action_2 = dist_2.sample().item()
        stdev_2 = stdev_2.detach().item()
        
        return action_1, stdev_1, action_2, stdev_2
  
    
    # Gets the autodifferentiable probability ratios for the given trajectory
    # @ param minibatch_indices - list of indices that represent set of minibatches over which SGD will occur
    # @ return - the differentiable probability ratios
    def get_action_prob_ratio_minibatches(self, minibatch_indices):
        
        probability_ratio_minibatches = []
        
        for curr_epoch in range(self.num_epochs):
            # Get the current and old action distribution parameters
            state_minibatch = torch.tensor(self.trajectory_states[minibatch_indices[curr_epoch,:], :], dtype=torch.float)
            means_minibatch, stdev_1, stdev_2 = self.actor.forward(state_minibatch)
            with torch.no_grad():
                old_means_minibatch, old_stdev_1, old_stdev_2 = self.old_actor.forward(state_minibatch)
            
            # Set the distributions based on the parameters above
            dist_1 = torch.distributions.normal.Normal(means_minibatch[:,0], stdev_1)
            dist_2 = torch.distributions.normal.Normal(means_minibatch[:,1], stdev_2)
            old_dist_1 = torch.distributions.normal.Normal(old_means_minibatch[:,0], old_stdev_1)
            old_dist_2 = torch.distributions.normal.Normal(old_means_minibatch[:,1], old_stdev_2)
            
            # Get the probability ratios
            action_minibatch = torch.tensor(self.trajectory_actions[:,minibatch_indices[curr_epoch,:]], dtype=torch.float)
            numerator_1 = torch.exp(dist_1.log_prob(action_minibatch[0,:])).double()
            numerator_2 = torch.exp(dist_2.log_prob(action_minibatch[1,:])).double()
            denominator_1 =  torch.exp(old_dist_1.log_prob(action_minibatch[0,:])).double()
            denominator_2 =  torch.exp(old_dist_2.log_prob(action_minibatch[1,:])).double()
            probability_ratio = (numerator_1 * numerator_2) / (denominator_1 * denominator_2)
            
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
            value_target_minibatch = reward_minibatch + self.gamma * next_value_estimate_minibatch
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
    # @ param action - action to be added to trajectory memory
    # @ param reward - reward to be added to trajectory memory
    def update_agent(self, state, action, reward):
        
        # Calculate the location in the memory matrices to insert new data
        current_trajectory = self.trajectory_index // self.steps_per_trajectory
        current_step = self.trajectory_index - current_trajectory * self.steps_per_trajectory
        
        # Update the state, action, and reward memory
        self.trajectory_states[self.trajectory_index,:] = state 
        self.trajectory_actions[:, self.trajectory_index] = action
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
            self.trajectory_actions = np.zeros((2, self.trajectories_per_batch*self.steps_per_trajectory))
            self.trajectory_rewards = np.zeros(self.trajectories_per_batch*self.steps_per_trajectory)
        
        
    # Updates the actor's NN weights based on a batch of trajectories
    # @ param advantage_estimate_minibatches - non differentiable minibatch set of advantage estimates
    # @ param probability_ratio_minibatches - differentiable minibatch set of action probability ratios
    # @ param value_estimate_minibatches - differentiable minibatch set of value estimates
    # @ param value_target_minibatches - non differentiable minibatch set of value targets
    def learn(self, advantage_estimate_minibatches, probability_ratio_minibatches, value_estimate_minibatches, value_target_minibatches):
        
        actor_loss = 0.0
        critic_loss = 0.0
        total_target = 0.0
        total_value = 0.0
                                
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
            
            total_target = total_target + value_target_minibatches[curr_epoch].mean().item()
            total_value = total_value + value_estimate_minibatches[curr_epoch].mean().item()
            
        self.value_estimation_error.append(100.0*(total_target - total_value)/total_target)
            
        # Conduct minibatch stochastic gradient descent on actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Conduct minibatch stochastic gradient descent on critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        