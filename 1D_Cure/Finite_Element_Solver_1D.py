# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:39:40 2020

@author: Grayson Schaer
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

class FES():
    
    def __init__(self):
    
        # Simulation parameters
        self.spacial_precision = 300 # Must be multiple of 10
        self.temporal_precision = 0.05
        self.field_length = 0.10
        self.simulation_time = 300.0
        self.current_time = 0.0
        
        # Initial conditions
        self.initial_temperature = 298.15
        self.initial_cure = 0.10
        self.initial_input_location = self.field_length / 2.0
        
        # Boundary conditions
        self.bc_thermal_conductivity = 0.80
        self.bc_heat_transfer_coef = 8.59
        self.ambient_temperature = 298.15
        
        # Reward and training targets
        self.maximum_temperature = 523.15
        self.desired_front_rate = 0.001
        
        # Trigger conditions
        self.trigger_temperature = 453.15
        self.trigger_time = 0.0
        self.trigger_length = 30.0
        
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
        
        # Input parameters
        self.peak_thermal_rate = 3.0
        self.radius_of_input = self.field_length / 15.0
        self.input_location = self.initial_input_location
        self.max_movement_rate = self.field_length * self.temporal_precision
        self.input_magnitude = self.peak_thermal_rate / 2.0
        self.max_magnitude_rate = self.peak_thermal_rate * self.temporal_precision
        K = (self.peak_thermal_rate * self.radius_of_input * np.sqrt(2.0*np.pi)) / (2.0*np.sqrt(np.log(10.0)))
        sigma = self.radius_of_input / (2.0*np.sqrt(np.log(10.0)))
        self.front_const = K / (sigma * np.sqrt(2.0 * np.pi))
        self.exponential_const = -1.0 / (2.0 * sigma * sigma)
        
        # Reward constants
        self.c1 =  5e6
        self.c2 = 0.20
        self.max_reward = 5.0
        
        # Mesh grids
        self.spacial_grid = np.linspace(0.0,self.field_length,self.spacial_precision)
        self.temperature_grid = np.array([self.initial_temperature]*self.spacial_precision)
        self.cure_grid = np.array([self.initial_cure]*self.spacial_precision)
        self.input_grid = np.array([0.0]*self.spacial_precision)
        self.front_position = 0.0
        self.front_rate = 0.0
        self.previous_front_move = 0.0

    def step(self, action):
        
        # Clip the action and use it to update the input's position and magnitude
        ok_action = True
        next_input_location = self.input_location + np.clip(action[0], -self.max_movement_rate, self.max_movement_rate)
        next_magnitude = self.input_magnitude + np.clip(action[1], -self.max_magnitude_rate, self.max_magnitude_rate)
        if (next_input_location > self.field_length+self.radius_of_input) or (next_input_location < self.radius_of_input):
            ok_action = False
        else:
            self.input_location = next_input_location
            self.input_magnitude = np.clip(next_magnitude, 0.0, self.peak_thermal_rate)
            
        # Update the input grid to reflet the current action
        self.input_grid = self.input_magnitude * self.front_const * np.exp((self.spacial_grid - self.input_location)**2 * self.exponential_const)
        self.input_grid[self.input_grid<0.01*self.peak_thermal_rate] = 0.0
        
        # Get the second spacial derivative of the temperature field
        temperature_grid_spline = UnivariateSpline(self.spacial_grid, self.temperature_grid, k=3, s=0.0)
        temperature_grid_second_derivative = temperature_grid_spline.derivative(n=2)
        temperature_grid_second_derivative = temperature_grid_second_derivative(self.spacial_grid)
        
        # Get the cure rate based on the cure kinetics
        cure_rate = ((self.pre_exponential*np.exp(-self.activiation_energy / (self.temperature_grid*self.gas_const))) * 
                     ((1 - self.cure_grid)**self.model_fit_order) * 
                     (1 + self.autocatalysis_const * self.cure_grid))
        
        # Update the cure field using forward Euler method
        self.cure_grid = self.cure_grid + cure_rate * self.temporal_precision
        
        # Calculate the front position and rate
        if (self.cure_grid >= 0.99).any():
            new_front_position = self.spacial_grid[np.flatnonzero(self.cure_grid>=0.99)[-1]]
            if new_front_position != self.front_position:
                self.front_rate = (new_front_position - self.front_position) / (self.current_time - self.previous_front_move)
                self.previous_front_move = self.current_time
        else:
            new_front_position = 0.0
            self.front_rate = 0.0
        self.front_position = new_front_position
        
        # Use the second spacial derivative of the temperature field, input field, and cure rate to calculate the temperature field rate
        temperature_grid_rate = (((self.thermal_conductivity * temperature_grid_second_derivative) + 
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
        
        # Return the current state (reduced temperature field, front pos, front rate, input pos, input mag), and get reward
        state = np.concatenate((self.temperature_grid[0::10], [self.front_position], [self.front_rate], [self.input_location], [self.input_magnitude]))
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
                
                # Calculate square error between current state and desired state
                sq_error = ((self.front_rate - self.desired_front_rate)**2)
                reward = max(self.max_reward - self.c1 * sq_error, 0.0)
                
            # If the maximum temperature is above the maximum allowed temperature, return a negative reward
            else:
                reward = -self.c2 * self.max_reward
            
            return reward
    
    def reset(self):
        # Reset time
        self.current_time = 0.0
        
        # Reset input
        self.input_location = self.initial_input_location
        self.input_magnitude = self.peak_thermal_rate / 2.0
                
        # Reset fields
        self.temperature_grid = np.array([self.initial_temperature]*self.spacial_precision)
        self.cure_grid = np.array([self.initial_cure]*self.spacial_precision)
        self.input_grid = np.array([0.0]*self.spacial_precision)
        self.front_position = 0.0
        
        # Return the temperature field
        return np.concatenate((self.temperature_grid[0::10], [self.front_position], [0.0], [self.input_location], [self.input_magnitude]))