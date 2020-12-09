# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:39:40 2020

@author: Grayson Schaer
"""

import numpy as np

class FES():
    
    def __init__(self):
    
        # Simulation parameters
        self.spacial_precision = 400 # Must be multiple of 10
        self.temporal_precision = 0.05
        self.field_length = 0.060
        self.simulation_time = 360.0
        self.current_time = 0.0
        
        # Initial conditions
        self.initial_temperature = 278.15
        self.initial_temp_perturbation = 0.005 * self.initial_temperature
        self.initial_cure = 0.10
        self.initial_cure_perturbation = 0.01 * self.initial_cure
        
        # Boundary conditions
        self.bc_thermal_conductivity = 0.152
        self.bc_heat_transfer_coef = 0.025
        self.ambient_temperature = self.initial_temperature
        
        # Reward and training targets
        self.maximum_temperature = 563.15
        self.desired_front_rate = 0.00015
        
        # Trigger conditions
        self.trigger_temperature = 458.15
        self.trigger_time = 0.0
        self.trigger_length = 35.0
        
        # Physical parameters
        self.thermal_conductivity = 0.152
        self.density = 980.0
        self.enthalpy_of_reaction = 352100.0
        self.specific_heat = 1440.0
        self.pre_exponential = 190985.325
        self.activiation_energy = 51100.0
        self.gas_const = 8.3144
        self.model_fit_order = 1.927
        self.autocatalysis_const = 0.365
        
        # Mesh grids
        self.spacial_grid = np.linspace(0.0,self.field_length,self.spacial_precision)
        rough_temp_perturbation = np.random.rand(self.spacial_precision)*self.initial_temp_perturbation
        smooth_temp_perturbation = self.initial_temp_perturbation * np.sin((np.random.randint(1,6) * np.pi * self.spacial_grid) / (self.field_length))
        self.temperature_grid = self.initial_temperature + rough_temp_perturbation + smooth_temp_perturbation
        rough_cure_perturbation = np.random.rand(self.spacial_precision)*self.initial_cure_perturbation
        smooth_cure_perturbation = self.initial_cure_perturbation * np.sin((np.random.randint(1,6) * np.pi * self.spacial_grid) / (self.field_length))
        self.cure_grid = self.initial_cure + rough_cure_perturbation + smooth_cure_perturbation
        self.input_grid = np.array([0.0]*self.spacial_precision)
        self.front_position = 0.0
        self.front_rate = 0.0
        self.previous_front_move = 0.0
        self.front_has_started=False
        
        # Input parameters
        self.peak_thermal_rate = 3.0
        self.radius_of_input = self.field_length / 10.0
        self.input_location = np.random.choice(self.spacial_grid)
        self.max_movement_rate = self.field_length * self.temporal_precision
        self.input_magnitude = np.random.rand() * self.peak_thermal_rate
        self.max_magnitude_rate = self.peak_thermal_rate * self.temporal_precision
        K = (self.peak_thermal_rate * self.radius_of_input * np.sqrt(2.0*np.pi)) / (2.0*np.sqrt(np.log(10.0)))
        sigma = self.radius_of_input / (2.0*np.sqrt(np.log(10.0)))
        self.front_const = K / (sigma * np.sqrt(2.0 * np.pi))
        self.exponential_const = -1.0 / (2.0 * sigma * sigma)
        
        # Reward constants
        self.max_reward = 1.5
        self.c1 =  self.max_reward / (0.75*self.desired_front_rate)
        self.c2 = self.max_reward / 25.0

        # Simulation limits
        self.stable_temperature_limit = 10.0 * self.maximum_temperature

    def step(self, action):
        
        # Clip the action and use it to update the input's position and magnitude
        ok_action = True
        next_input_location = self.input_location + np.clip(0.001*action[0], -self.max_movement_rate, self.max_movement_rate)
        next_magnitude = self.input_magnitude + np.clip(0.05*action[1], -self.max_magnitude_rate, self.max_magnitude_rate)
        if (next_input_location > self.field_length+self.radius_of_input) or (next_input_location < self.radius_of_input):
            ok_action = False
        else:
            self.input_location = next_input_location
            self.input_magnitude = np.clip(next_magnitude, 0.0, self.peak_thermal_rate)
            
        # Update the input grid to reflet the current action
        self.input_grid = self.input_magnitude * self.front_const * np.exp((self.spacial_grid - self.input_location)**2 * self.exponential_const)
        self.input_grid[self.input_grid<0.01*self.peak_thermal_rate] = 0.0
        
        # Get the second spacial derivative of the temperature field
        x_step_size = self.spacial_grid[1] - self.spacial_grid[0]
        diff_x_grid = np.insert(self.spacial_grid, 0, np.array([-2.0*x_step_size, -x_step_size]))
        diff_x_grid = np.insert(diff_x_grid, len(diff_x_grid), np.array([self.spacial_grid[-1]+x_step_size, self.spacial_grid[-1]+2.0*x_step_size]))
        diff_y_grid = np.insert(self.temperature_grid, 0, np.array([self.ambient_temperature, self.ambient_temperature]))
        diff_y_grid = np.insert(diff_y_grid, len(diff_y_grid), np.array([self.ambient_temperature, self.ambient_temperature]))
        first_diff_grid = np.zeros(diff_y_grid.shape)
        first_diff_grid[0:-1] = np.diff(diff_y_grid)/np.diff(diff_x_grid)
        first_diff_grid[-1] = (diff_y_grid[-1] - diff_y_grid[-2])/(diff_x_grid[-1] - diff_x_grid[-2])
        second_diff_grid = np.zeros(first_diff_grid.shape)
        second_diff_grid[0:-1] = np.diff(first_diff_grid)/np.diff(diff_x_grid)
        second_diff_grid[-1] = (first_diff_grid[-1] - first_diff_grid[-2])/(diff_x_grid[-1] - diff_x_grid[-2])
        second_diff_grid = second_diff_grid[1:-3]
        
        # Get the cure rate based on the cure kinetics
        cure_rate = ((self.pre_exponential*np.exp(-self.activiation_energy / (self.temperature_grid*self.gas_const))) * 
                     ((1 - self.cure_grid)**self.model_fit_order) * 
                     (1 + self.autocatalysis_const * self.cure_grid))
        
        # Update the cure field using forward Euler method
        self.cure_grid = self.cure_grid + cure_rate * self.temporal_precision
        
        # Calculate the front position and rate
        cure_diff = -1.0*np.diff(self.cure_grid)/np.diff(self.spacial_grid)
        if (cure_diff>=100.0).any():
            new_front_position = self.spacial_grid[np.flatnonzero(cure_diff>=100.0)[-1]]
            if new_front_position != self.front_position:
                if self.front_has_started:
                    self.front_rate = (new_front_position - self.front_position) / (self.current_time - self.previous_front_move)
                else:
                    self.front_has_started = True
                self.previous_front_move = self.current_time
        else:
            new_front_position = 0.0
            self.front_rate = 0.0
        self.front_position = new_front_position
        
        # Use the second spacial derivative of the temperature field, input field, and cure rate to calculate the temperature field rate
        temperature_grid_rate = (((self.thermal_conductivity * second_diff_grid) + 
                                  (self.density * self.enthalpy_of_reaction * cure_rate) + 
                                  (self.density * self.specific_heat * self.input_grid)) / 
                                  (self.density * self.specific_heat))
        
        # Appy the boundary conditions
        temperature_grid_rate[0] = self.bc_heat_transfer_coef * (self.temperature_grid[0] - self.ambient_temperature) / (-1.0 * self.bc_thermal_conductivity)
        temperature_grid_rate[-1] = self.bc_heat_transfer_coef * (self.temperature_grid[-1] - self.ambient_temperature) / (-1.0 * self.bc_thermal_conductivity)
        
        # Use the temperature field using forward Euler method
        self.temperature_grid = self.temperature_grid + temperature_grid_rate * self.temporal_precision
        
        # Apply trigger thermal input
        if self.current_time >= self.trigger_time and self.current_time < self.trigger_time + self.trigger_length:
            self.temperature_grid[0] = self.trigger_temperature
        
        # Check for unstable growth
        if((self.temperature_grid >= self.stable_temperature_limit).any() or (self.temperature_grid <= -self.stable_temperature_limit).any()):
            raise RuntimeError('Unstable growth detected. Increase temporal precision, decrease spatial precision, or lower thermal conductivity')
        
        # Return the current state (reduced temperature field, front pos, front rate, input pos, input mag), and get reward
        state = np.concatenate((self.temperature_grid[0::10], [self.temperature_grid[-1]], [self.front_position], [self.front_rate], [self.input_location], [self.input_magnitude]))
        reward = self.get_reward(ok_action)
        
        # Update the current time and check for simulation completion
        done = (self.current_time + 2.0*self.temporal_precision >= self.simulation_time)
        if not done:
            self.current_time = self.current_time + self.temporal_precision
            
        # Return next state, reward for previous action, and whether simulation is complete or not
        return state, reward, done
    
    def get_reward(self, ok_action):
        
        if not ok_action:
            return -self.max_reward
        
        else:
            
            # If the maximum temperature is below the maximum allowed temperature, return a positive reward
            if (self.temperature_grid <= self.maximum_temperature).all():
                
                # Calculate error between current state and desired state
                error = abs(self.front_rate - self.desired_front_rate)
                reward = max(self.max_reward - self.c1 * error, 0.0)
                
            # If the maximum temperature is above the maximum allowed temperature, return a negative reward
            else:
                # Calculate overage between current temperature and maximum temperature
                overage = np.max(self.temperature_grid) - self.maximum_temperature
                reward = max(-self.c2 * overage, -self.max_reward)
            
            return reward
    
    def reset(self):
        # Reset time
        self.current_time = 0.0
        
        # Reset input
        self.input_location = np.random.choice(self.spacial_grid)
        self.input_magnitude = np.random.rand() * self.peak_thermal_rate
                
        # Reset fields
        self.temperature_grid = self.initial_temperature + self.initial_temp_perturbation * np.sin((np.random.randint(1,6) * np.pi * self.spacial_grid) / (self.field_length))
        self.cure_grid = self.initial_cure + self.initial_cure_perturbation * np.sin((np.random.randint(1,6) * np.pi * self.spacial_grid) / (self.field_length))
        self.input_grid = np.array([0.0]*self.spacial_precision)
        self.front_position = 0.0
        self.front_rate = 0.0
        self.previous_front_move = 0.0
        
        # Return the temperature field
        return np.concatenate((self.temperature_grid[0::10], [self.temperature_grid[-1]], [self.front_position], [self.front_rate], [self.input_location], [self.input_magnitude]))