# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 10:41:07 2020

@author: Grayson Schaer
"""

import numpy as np
import torch
import NN_Stdev_3_Output
import NN
import pickle
import os

class PPO_Agent:

    def __init__(self, num_states, steps_per_trajectory, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, alpha, decay_rate, load_agent, reset_stdev):

        # Policy and value estimation network
        # Input is the state
        # Output is the mean of the gaussian distribution from which actions are sampled
        self.actor = NN_Stdev_3_Output.Neural_Network(num_inputs=num_states, num_outputs=3, num_hidden_layers=2, num_neurons_in_layer=160)
        self.actor_optimizer =  torch.optim.Adam(self.actor.parameters() , lr=alpha)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.actor_optimizer, gamma=decay_rate)

        # Old policy and value estimation network used to calculate clipped surrogate objective
        # Input is the state
        # Output is the mean of the gaussian distribution from which actions are sampled
        self.old_actor = NN_Stdev_3_Output.Neural_Network(num_inputs=num_states, num_outputs=3, num_hidden_layers=2, num_neurons_in_layer=160)
        self.old_actor.load_state_dict(self.actor.state_dict())

        # Critic NN that estimates the value function
        # Input is the state
        # Output is the estimated value of that state
        self.critic =  NN.Neural_Network(num_inputs=num_states, num_outputs=1, num_hidden_layers=2, num_neurons_in_layer=160)
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
        
        if load_agent == 1:
            # Find load path
            print("Loading previous agent...")
            done = False
            curr_folder = 1
            while not done:
                path = "results/PPO_"+str(curr_folder)
                if not os.path.isdir(path):
                    done = True
                else:
                    curr_folder = curr_folder + 1
            load_path = "results/PPO_"+str(curr_folder-1)
            if not os.path.isdir(load_path):
                raise RuntimeError("Could not find previous folder: " + load_path)
            load_path = load_path + "/output"
            if not os.path.exists(load_path):
                raise RuntimeError("Could not find previous output file: " + load_path)
            with open(load_path, 'rb') as file:
                input_data = pickle.load(file)
            if 'agent' in input_data:
                old_agent = input_data['agent']
            else:
                old_agent = input_data['logbook']['agents'][0]
                
            # Copy previous agent
            self.copy(old_agent, reset_stdev)

    # Copies the actor and critic NNs from another agent to this agent
    # @param agent - the agent from which the NNs are copied
    # @param reset_stdev - boolean that determines if the stdev of the agent is reset or not
    def copy(self, agent, reset_stdev):
        # Copy the actor NN
        self.actor.load_state_dict(agent.actor.state_dict())

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
        self.critic.load_state_dict(agent.critic.state_dict())
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
        return action_1, action_2, action_3

    # Calcuates determinisitc action given state and policy, but action just a ping pong back and forth weeeeee
    # @ param state - The state in which the policy is applied to calculate the action
    # @ param y_loc_rate_action - The y rate of the input
    # @ return action - The calculated deterministic action based on the state and policy
    def get_greedy_pingpong(self, state, y_loc_rate_action):

        # Get the gaussian distribution parameters used to sample the action for the old and new policy
        with torch.no_grad():
            means, stdev_1, stdev_2, stdev_3 = self.actor.forward(torch.tensor(state, dtype=torch.float))

        # Return the means
        action_1 = float(means[0])
        action_3 = float(means[2])

        # Ping pong weeeee
        if state[-2] >= 0.95:
            action_2 = -50.0
        elif state[-2] <= 0.05:
            action_2 = 50.0
        else:
            if y_loc_rate_action == 0.0:
                action_2 = -50.0
            else:
                action_2 = y_loc_rate_action

        return action_1, action_2, action_3

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
