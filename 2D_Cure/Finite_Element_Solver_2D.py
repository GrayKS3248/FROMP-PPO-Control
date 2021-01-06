# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:39:40 2020

@author: Grayson Schaer
"""

import numpy as np

class FES():
    
    def __init__(self, for_pd=False, random_target=False, target_switch=False, control=False):
        
        # Environment spatial parameters 
        self.num_panels_length = 400 # Must be multiple of 10
        self.num_panels_width = 60 # Must be multiple of 10
        self.num_panels = self.num_panels_length * self.num_panels_width
        self.length = 0.060
        self.width = 0.010
        
        # Environment time parameters
        self.sim_duration = 360.0
        self.current_time = 0.0
        self.time_step = 0.05
        self.current_index = 0
        
        # Initial conditions
        self.initial_temperature = 278.15
        self.initial_temp_delta = 0.01 * self.initial_temperature
        self.initial_cure = 0.10
        self.initial_cure_delta = 0.01 * self.initial_cure
        
        # Boundary conditions
        self.bc_tc= 0.152
        self.bc_htc = 0.025
        self.ambient_temperature = self.initial_temperature
        
        # Problem definition constants
        self.temperature_limit = 563.15
        self.target = 0.00015
        self.purturbation_scale = 0.000025
        self.random_target = random_target
        self.target_switch = target_switch
        self.control = control
        
        # Calculate the target vectors
        self.target_front_vel = np.ones(int(self.sim_duration / self.time_step))*self.target
        if self.random_target:
            self.target_front_vel = np.ones(int(self.sim_duration / self.time_step))*self.target - 2.0*(np.random.rand()-0.5)*self.purturbation_scale
        if self.target_switch:
            switch_location = int((0.20*np.random.rand()+0.40) * (len(self.target_front_vel)-1))
            switch_vel = self.target_front_vel[switch_location] + 2.0*(np.random.rand()-0.5)*self.purturbation_scale
            self.target_front_vel[switch_location:]=switch_vel
        self.current_target_front_vel = self.target_front_vel[self.current_index]
        
        # Trigger conditions
        self.trigger_temperature = 458.15
        self.trigger_time = 0.0
        self.trigger_duration = 35.0
        
        # Monomer physical parameters
        self.thermal_conductivity = 0.152
        self.density = 980.0
        self.enthalpy_of_reaction = 352100.0
        self.specific_heat = 1440.0
        self.pre_exponential = 190985.325
        self.activiation_energy = 51100.0
        self.gas_const = 8.3144
        self.model_fit_order = 1.927
        self.autocatalysis_const = 0.365
        
        # Spatial panels
        self.panels_x, self.panels_y = np.meshgrid(np.linspace(0.0,self.length,self.num_panels_length), np.linspace(0.0,self.width,self.num_panels_width))
        self.step_size_x = self.panels_x[0][1]-self.panels_x[0][0]
        self.step_size_y = self.panels_y[0][1]-self.panels_y[0][0]
        
        # Temperature panels
        self.temp_panels = np.ones((self.num_panels_length,self.num_panels_width))*self.initial_temperature
        self.temp_panels = self.temp_panels + self.get_perturbation(self.temp_panels, self.initial_temp_delta)
        
        # Cure panels
        self.cure_panels = np.ones((self.num_panels_length,self.num_panels_width))*self.initial_cure
        self.cure_panels = self.cure_panels + self.get_perturbation(self.cure_panels, self.initial_cure_delta)
        
        # Front parameters
        self.front_loc = np.zeros(self.num_panels_width)
        self.front_vel = np.zeros(self.num_panels_width)
        self.time_front_last_moved = np.zeros(self.num_panels_width)
        self.front_has_started = np.zeros(self.num_panels_width)
        
        # Input magnitude parameters
        if self.control:
            self.max_input_mag = 0.0
        else:
            self.max_input_mag = 10.0
        self.max_input_mag_rate = self.time_step
        self.input_magnitude = np.random.rand()
        self.mag_scale = 0.083333333
        self.mag_offset = 0.5
        
        # Input distribution parameters
        self.radius_of_input = self.length / 10.0
        K = (self.max_input_mag * self.radius_of_input * np.sqrt(2.0*np.pi)) / (2.0*np.sqrt(np.log(10.0)))
        sigma = self.radius_of_input / (2.0*np.sqrt(np.log(10.0)))
        self.front_const = K / (sigma * np.sqrt(2.0 * np.pi))
        self.exp_const = -1.0 / (2.0 * sigma * sigma)
        
        # Input location parameters
        self.min_input_x_loc = 0.0
        self.max_input_x_loc = self.length
        self.max_input_x_loc_rate = self.length * self.time_step
        self.min_input_y_loc = 0.0
        self.max_input_y_loc = self.width
        self.max_input_y_loc_rate = self.length * self.time_step
        self.input_location = np.array([np.random.choice(self.panels_x[0]), np.random.choice(self.panels_y[:,0])])
        self.loc_rate_scale = 0.0006
        self.loc_rate_offset = 0.0
        
        # Input panels
        self.input_panels = self.input_magnitude * self.front_const * np.exp(((self.panels_x - self.input_location[0])**2 * self.exp_const) + 
                                                                              (self.panels_y - self.input_location[1])**2 * self.exp_const)
        self.input_panels[self.input_panels<0.01*self.max_input_mag] = 0.0
        
        # Reward constants
        self.max_reward = 2.0
        self.input_punishment_const = 0.10
        self.overage_punishment_const = 0.25
        self.integral_punishment_const = 0.10
        self.max_integral = self.length * self.width * self.temperature_limit
        self.integral_delta = self.max_integral - self.length * self.width * self.initial_temperature
        
        # Simulation limits
        self.stab_lim = 10.0 * self.temperature_limit
        self.for_pd = for_pd

    # Get smooth 2D perturbation for temperature and cure fields
    def get_perturbation(self, size_array, delta):
        # Get magnitude and biases
        mag_1 = 2.0 * np.random.rand() - 1.0
        mag_2 = 2.0 * np.random.rand() - 1.0
        mag_3 = 2.0 * np.random.rand() - 1.0
        bias_1 = 4.0 * np.pi * np.random.rand() - 2.0 * np.pi
        bias_2 = 4.0 * np.pi * np.random.rand() - 2.0 * np.pi
        bias_3 = 4.0 * np.pi * np.random.rand() - 2.0 * np.pi
        min_mag = np.random.rand()
        max_mag = np.random.rand()
        min_x_bias = 2.0*np.random.rand()-1.0
        max_x_bias = 2.0*np.random.rand()-1.0
        min_y_bias = 2.0*np.random.rand()-1.0
        max_y_bias = 2.0*np.random.rand()-1.0
        
        # Determine size of perturbation field
        x, y = np.meshgrid(np.linspace(-2.0*min_mag+min_x_bias, 2.0*max_mag+max_x_bias, len(size_array[0])), np.linspace(-2.0*min_mag+min_y_bias, 2.0*max_mag+max_y_bias, len(size_array)))
        xy = x * y
        
        # Calculate perturbation field
        perturbation = (mag_1*np.sin(1.0*xy+bias_1) + mag_2*np.sin(2.0*xy+bias_2) + mag_3*np.sin(3.0*xy+bias_3))
        scale = np.max(abs(perturbation))
        perturbation = (delta * perturbation) / scale
        
        return perturbation

    def step_input(self, action):
        # Check if the finite element solver is set up for a ppo or pd controller
        if self.for_pd:
            # Update input's position
            x_location_rate_command = np.clip(action[0], -self.max_input_x_loc_rate, self.max_input_x_loc_rate)
            input_x_location = np.clip(self.input_location[0] + x_location_rate_command * self.time_step, self.min_input_x_loc, self.max_input_x_loc)
            y_location_rate_command = np.clip(action[1], -self.max_input_y_loc_rate, self.max_input_y_loc_rate)
            input_y_location = np.clip(self.input_location[1] + y_location_rate_command * self.time_step, self.min_input_y_loc, self.max_input_y_loc)
            self.input_location = np.array([input_x_location, input_y_location])
            
            # Update input's magnitude
            magnitude_rate_command = np.clip(action[2], -self.max_input_mag_rate, self.max_input_mag_rate)
            self.input_magnitude = np.clip(self.input_magnitude + magnitude_rate_command * self.time_step, 0.0, 1.0)
        
        else:
            # Update the input's position
            x_location_rate_command = np.clip(self.loc_rate_offset + self.loc_rate_scale * action[0], -self.max_input_x_loc_rate, self.max_input_x_loc_rate)
            input_x_location = np.clip(self.input_location[0] + x_location_rate_command * self.time_step, self.min_input_x_loc, self.max_input_x_loc)
            y_location_rate_command = np.clip(self.loc_rate_offset + self.loc_rate_scale * action[1], -self.max_input_y_loc_rate, self.max_input_y_loc_rate)
            input_y_location = np.clip(self.input_location[1] + y_location_rate_command * self.time_step, self.min_input_y_loc, self.max_input_y_loc)
            self.input_location = np.array([input_x_location, input_y_location])
            
            # Update the input's magnitude
            magnitude_command = self.mag_offset + self.mag_scale * action[2]
            if magnitude_command > self.input_magnitude:
                self.input_magnitude = np.clip(min(self.input_magnitude + self.max_input_mag_rate, magnitude_command), 0.0, 1.0)
            elif magnitude_command < self.input_magnitude:
                self.input_magnitude = np.clip(max(self.input_magnitude - self.max_input_mag_rate, magnitude_command), 0.0, 1.0)
            else:
                self.input_magnitude = np.clip(self.input_magnitude, 0.0, self.max_input_mag)

        # Use the actions to define input thermal rate across entire spacial field
        self.input_panels = self.input_magnitude * self.front_const * np.exp(((self.panels_x - self.input_location[0])**2 * self.exp_const) + 
                                                                              (self.panels_y - self.input_location[1])**2 * self.exp_const)
        self.input_panels[self.input_panels<0.01*self.max_input_mag] = 0.0
        
    def step_cure(self):
        # Get the cure rate across the entire field based on the cure kinetics
        cure_rate = ((self.pre_exponential*np.exp(-self.activiation_energy / (self.temp_panels*self.gas_const))) * 
                     ((1 - self.cure_panels)**self.model_fit_order) * 
                     (1 + self.autocatalysis_const * self.cure_panels))
        
        # Update the cure field using forward Euler method
        self.cure_panels = self.cure_panels + cure_rate * self.time_step
        
        # Return the cure rate
        return cure_rate

    # TODO
    def step_front(self):
        # Calculate the spatial cure derivative
        cure_diff = -1.0*np.diff(self.cure_panels,axis=0)/self.step_size_x
        
        # If any point's spatial cure derivative is greater than a threshold, update the front location
        if (cure_diff>=100.0).any():
            
            # Find the furthest right points in each row that meet the spatial cure derivative threshold
            new_front_loc = np.zeros(self.num_panels_width)
            for curr_row in range(self.num_panels_width):
                new_front_loc[curr_row] = self.panels_x[curr_row,np.flatnonzero(cure_diff[curr_row,:]>=100.0)[-1]]
                
            # If the front has moved compared to the previously recorded front location, update the recorded front velocity
            if (new_front_loc != self.front_loc).any():
                
                # Update each front row
                for curr_row in range(self.num_panels_width):
                    
                    # Only update front rows that have moved
                    if new_front_loc != self.front_loc[curr_row]:
                        
                        # If the front has already began, update the recorded front velocity
                        if self.front_has_started[curr_row] != 0.0:
                            self.front_vel[curr_row] = (new_front_loc[curr_row] - self.front_loc[curr_row]) / (self.current_time - self.time_front_last_moved[curr_row])
                        
                        # If the front has not already started, mark the front as started
                        else:
                            self.front_has_started[curr_row] = 1.0
                        
                        # Update the last time the front moved
                        self.time_front_last_moved[curr_row] = self.current_time[curr_row]
        
        # If the front has not started yet, set front position and rate to 0.0
        else:
            new_front_loc = np.zeros(self.num_panels_width)
            self.front_vel = np.zeros(self.num_panels_width)
            
        # Update the front location
        self.front_loc = new_front_loc

    # TODO
    def step_temperature(self, cure_rate):
        # Get the second spacial derivative of the temperature field
        diff_x_grid = np.insert(self.panels, 0, np.array([-2.0*self.step_size, -self.step_size]))
        diff_x_grid = np.insert(diff_x_grid, len(diff_x_grid), np.array([self.panels[-1]+self.step_size, self.panels[-1]+2.0*self.step_size]))
        diff_y_grid = np.insert(self.temp_panels, 0, np.array([self.ambient_temperature, self.ambient_temperature]))
        diff_y_grid = np.insert(diff_y_grid, len(diff_y_grid), np.array([self.ambient_temperature, self.ambient_temperature]))
        first_diff_grid = np.zeros(diff_y_grid.shape)
        first_diff_grid[0:-1] = np.diff(diff_y_grid)/np.diff(diff_x_grid)
        first_diff_grid[-1] = (diff_y_grid[-1] - diff_y_grid[-2])/(diff_x_grid[-1] - diff_x_grid[-2])
        second_diff_grid = np.zeros(first_diff_grid.shape)
        second_diff_grid[0:-1] = np.diff(first_diff_grid)/np.diff(diff_x_grid)
        second_diff_grid[-1] = (first_diff_grid[-1] - first_diff_grid[-2])/(diff_x_grid[-1] - diff_x_grid[-2])
        second_diff_grid = second_diff_grid[1:-3]
        
        # Use the second spacial derivative of the temperature field, input field, and cure rate to calculate the temperature field rate
        temp_panels_rate = (((self.thermal_conductivity * second_diff_grid) + 
                                  (self.density * self.enthalpy_of_reaction * cure_rate) + 
                                  (self.density * self.specific_heat * self.input_panels)) / 
                                  (self.density * self.specific_heat))
        
        # Appy the boundary conditions
        temp_panels_rate[0] = self.bc_htc * (self.temp_panels[0] - self.ambient_temperature) / (-1.0 * self.bc_tc)
        temp_panels_rate[-1] = self.bc_htc * (self.temp_panels[-1] - self.ambient_temperature) / (-1.0 * self.bc_tc)
        
        # Update the temperature field using forward Euler method
        self.temp_panels = self.temp_panels + temp_panels_rate * self.time_step
        
        # Apply trigger thermal input
        if self.current_time >= self.trigger_time and self.current_time < self.trigger_time + self.trigger_duration:
            self.temp_panels[0] = self.trigger_temperature
            
        # Check for unstable growth
        if((self.temp_panels >= self.stab_lim).any() or (self.temp_panels <= -self.stab_lim).any()):
            raise RuntimeError('Unstable growth detected. Increase temporal precision, decrease spatial precision, or lower thermal conductivity')

        return temp_panels_rate

    def blockshaped(self, arr, nrows, ncols):
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))

    def get_state(self, temp_rate, action):
        # Check if the finite element solver is set up for a ppo or pd controller
        if self.for_pd:
            # Get the average temperature and temperature rates of self.num_panels/10 even segments across entire length
            average_temps = np.mean(self.blockshaped(self.temp_panels,self.num_panels_length//20,self.num_panels_width//20),axis=0)
            average_temps = average_temps.reshape(np.size(average_temps))
            average_temp_rates = np.mean(self.blockshaped(temp_rate,self.num_panels_length//20,self.num_panels_width//20),axis=0)
            average_temp_rates = average_temp_rates.reshape(np.size(average_temp_rates))
            
            # Compress front location and velocity data
            average_front_loc = np.mean(self.front_loc.reshape(self.num_panels_width//10,10),axis=1)
            average_front_vel = np.mean(self.front_vel.reshape(self.num_panels_width//10,10),axis=1)
            
            # Get the input location and magnitude rates
            input_x_location_rate = np.clip(action[0], -self.max_input_x_loc_rate, self.max_input_x_loc_rate)
            input_y_location_rate = np.clip(action[1], -self.max_input_y_loc_rate, self.max_input_y_loc_rate)
            
            # Normalize and concatenate all substates
            state = np.concatenate((average_temps, average_temp_rates,
                                    average_front_loc, average_front_vel, 
                                    self.input_location, [input_x_location_rate], [input_y_location_rate],
                                    [self.current_target_front_vel]))  
            
        else:
            # Get the average temperature in even areas across entire field
            average_temps = np.mean(self.blockshaped(self.temp_panels,self.num_panels_length//20,self.num_panels_width//20),axis=0)
            average_temps = average_temps.reshape(np.size(average_temps))
            
            # Find the area over which the laser can see
            x_min = np.argmin(abs(self.panels_x[0,:] - self.input_location[0] + self.radius_of_input))
            x_max = np.argmin(abs(self.panels_x[0,:] - self.input_location[0] - self.radius_of_input))
            x_max = x_max - (x_max-x_min)%5
            y_min = np.argmin(abs(self.panels_y[:,0] - self.input_location[1] + self.radius_of_input))
            y_max = np.argmin(abs(self.panels_y[:,0] - self.input_location[1] - self.radius_of_input))
            y_max = y_max - (y_max-y_min)%5
    
            # Calculate average temperature blocks (5X5) in laser view
            laser_view = np.mean(self.blockshaped(self.temp_panels[x_min:x_max,y_min:y_max],5,5),axis=0)
            laser_view = laser_view.reshape(np.size(laser_view))
            
            # Compress front location and velocity data
            average_front_loc = np.mean(self.front_loc.reshape(self.num_panels_width//10,10),axis=1)
            average_front_vel = np.mean(self.front_vel.reshape(self.num_panels_width//10,10),axis=1)
            
            # Normalize and concatenate all substates
            state = np.concatenate((average_temps/self.temperature_limit, 
                                    laser_view/self.temperature_limit,
                                    average_front_loc/self.length, 
                                    average_front_vel/self.current_target_front_vel, 
                                    [self.input_location[0]/self.length], [self.input_location[1]/self.width],
                                    [self.input_magnitude]))
            
        # Return the state
        return state

    # TODO
    def get_reward(self):
        # Calculate the punishments based on the temperature field, input strength, action, and overage
        if self.control:
            input_punishment = 0.0
        else:
            input_punishment = -self.input_punishment_const * self.max_reward * (self.input_magnitude / self.max_input_mag)
        overage_punishment =  -self.overage_punishment_const * self.max_reward * (max(self.temp_panels) >= self.temperature_limit)
        integral_punishment = -self.integral_punishment_const * self.max_reward * (1.0 - (self.max_integral - np.trapz(self.temp_panels,x=self.panels)) / (self.integral_delta))
        punishment = input_punishment + overage_punishment + integral_punishment
        
        # Calculate the reward based on the punishments and the front rate error
        if abs(self.front_vel - self.current_target_front_vel) / (self.current_target_front_vel) <= 0.075:
            front_rate_reward = self.max_reward
        elif abs(self.front_vel - self.current_target_front_vel) / (self.current_target_front_vel) <= 0.25:
            front_rate_reward = 0.10*self.max_reward
        else:
            front_rate_reward = -0.10*self.max_reward
        reward = front_rate_reward + punishment

        # Return the calculated reward
        return reward

    # TODO
    def step_time(self):
        # Update the current time and check for simulation completion
        done = (self.current_time + self.time_step >= self.sim_duration)
        if not done:
            self.current_time = self.current_time + self.time_step
            self.current_index = self.current_index + 1
            self.current_target_front_vel = self.target_front_vel[self.current_index]
        
        return done

    # TODO
    def step(self, action):
        # Step the input, cure, front, and temperature
        self.step_input(action)
        cure_rate = self.step_cure()
        self.step_front()
        temp_panels_rate = self.step_temperature(cure_rate)
        
        # Get state and reward
        state = self.get_state(temp_panels_rate, action)
        reward = self.get_reward()
        
        # Step time
        done = self.step_time()
            
        # Return next state, reward for previous action, and whether simulation is complete or not
        return state, reward, done
    
    def reset(self):
        # Reset time
        self.current_time = 0.0
        self.current_index = 0
        
        # Calculate the target vectors
        self.target_front_vel = np.ones(int(self.sim_duration / self.time_step))*self.target
        if self.random_target:
            self.target_front_vel = np.ones(int(self.sim_duration / self.time_step))*self.target - 2.0*(np.random.rand()-0.5)*self.purturbation_scale
        if self.target_switch:
            switch_location = int((0.20*np.random.rand()+0.40) * (len(self.target_front_vel)-1))
            switch_vel = self.target_front_vel[switch_location] + 2.0*(np.random.rand()-0.5)*self.purturbation_scale
            self.target_front_vel[switch_location:]=switch_vel
        self.current_target_front_vel = self.target_front_vel[self.current_index]
        
        # Reset input
        self.input_magnitude = np.random.rand()
        self.input_location = np.array([np.random.choice(self.panels_x[0]), np.random.choice(self.panels_y[:,0])])
        
        # Temperature panels
        self.temp_panels = np.ones((self.num_panels_length,self.num_panels_width))*self.initial_temperature
        self.temp_panels = self.temp_panels + self.get_perturbation(self.temp_panels, self.initial_temp_delta)
        
        # Cure panels
        self.cure_panels = np.ones((self.num_panels_length,self.num_panels_width))*self.initial_cure
        self.cure_panels = self.cure_panels + self.get_perturbation(self.cure_panels, self.initial_cure_delta)
        
        # Reset input panels
        self.input_panels = self.input_magnitude * self.front_const * np.exp(((self.panels_x - self.input_location[0])**2 * self.exp_const) + 
                                                                              (self.panels_y - self.input_location[1])**2 * self.exp_const)
        self.input_panels[self.input_panels<0.01*self.max_input_mag] = 0.0
        
        # Front parameters
        self.front_loc = np.zeros(self.num_panels_width)
        self.front_vel = np.zeros(self.num_panels_width)
        self.time_front_last_moved = np.zeros(self.num_panels_width)
        self.front_has_started = np.zeros(self.num_panels_width)
        
        # Return the initial state
        temp_rate = np.zeros((len(self.temp_panels), len(self.temp_panels[0])))
        action = np.array([0.0, 0.0, 0.0])
        return self.get_state(temp_rate, action)