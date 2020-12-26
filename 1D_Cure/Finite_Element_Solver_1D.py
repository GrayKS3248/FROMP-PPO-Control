# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:39:40 2020

@author: Grayson Schaer
"""

import numpy as np

class FES():
    
    def __init__(self, for_pd=False, random_target=False, target_switch=False, control=False):
        
        # Environment spatial parameters 
        self.num_panels = 400 # Must be multiple of 10
        self.length = 0.060
        
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
        self.panels = np.linspace(0.0,self.length,self.num_panels)
        self.step_size = self.panels[1] - self.panels[0]
        
        # Temperature panels
        perturbation = self.initial_temp_delta * np.sin((np.random.randint(1,6) * np.pi * self.panels) / (self.length))
        self.temp_panels = self.initial_temperature + perturbation
        
        # Cure panels
        perturbation = self.initial_cure_delta * np.sin((np.random.randint(1,6) * np.pi * self.panels) / (self.length))
        self.cure_panels = self.initial_cure + perturbation
        
        # Front parameters
        self.front_loc = 0.0
        self.front_vel = 0.0
        self.time_front_last_moved = 0.0
        self.front_has_started=False
        
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
        self.min_input_loc = 0.0
        self.max_input_loc = self.length
        self.max_input_loc_rate = self.length * self.time_step
        self.input_location = np.random.choice(self.panels)
        self.loc_rate_scale = 0.0006
        self.loc_rate_offset = 0.0
        
        # Input panels
        self.input_panels = self.input_magnitude * self.front_const * np.exp((self.panels - self.input_location)**2 * self.exp_const)
        self.input_panels[self.input_panels<0.01*self.max_input_mag] = 0.0
        
        # Reward constants
        self.max_reward = 2.0
        self.input_punishment_const = 0.10
        self.overage_punishment_const = 0.25
        self.integral_punishment_const = 0.10
        self.max_integral = np.trapz(self.temperature_limit*np.ones(len(self.temp_panels)),x=self.panels)
        self.integral_delta = self.max_integral - np.trapz(self.initial_temperature*np.ones(len(self.temp_panels)),x=self.panels)
        
        # Simulation limits
        self.stab_lim = 10.0 * self.temperature_limit
        self.for_pd = for_pd

    def step_input(self, action):
        # Check if the finite element solver is set up for a ppo or pd controller
        if self.for_pd:
            # Update input's position
            location_rate_command = np.clip(action[0], -self.max_input_loc_rate, self.max_input_loc_rate)
            self.input_location = np.clip(self.input_location + location_rate_command * self.time_step, self.min_input_loc, self.max_input_loc)
            
            # Update input's magnitude
            magnitude_rate_command = np.clip(action[1], -self.max_input_mag_rate, self.max_input_mag_rate)
            self.input_magnitude = np.clip(self.input_magnitude + magnitude_rate_command * self.time_step, 0.0, 1.0)
        
        else:
            # Update the input's position
            location_rate_command = np.clip(self.loc_rate_offset + self.loc_rate_scale * action[0], -self.max_input_loc_rate, self.max_input_loc_rate)
            self.input_location = np.clip(self.input_location + location_rate_command * self.time_step, self.min_input_loc, self.max_input_loc)
            
            # Update the input's magnitude
            magnitude_command = self.mag_offset + self.mag_scale * action[1]
            if magnitude_command > self.input_magnitude:
                self.input_magnitude = np.clip(min(self.input_magnitude + self.max_input_mag_rate, magnitude_command), 0.0, 1.0)
            elif magnitude_command < self.input_magnitude:
                self.input_magnitude = np.clip(max(self.input_magnitude - self.max_input_mag_rate, magnitude_command), 0.0, 1.0)
            else:
                self.input_magnitude = np.clip(self.input_magnitude, 0.0, self.max_input_mag)

        # Use the actions to define input thermal rate across entire spacial field
        self.input_panels = self.input_magnitude * self.front_const * np.exp((self.panels - self.input_location)**2 * self.exp_const)
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

    def step_front(self):
        # Calculate the front position and rate
        cure_diff = -1.0*np.diff(self.cure_panels)/np.diff(self.panels)
        if (cure_diff>=100.0).any():
            new_front_loc = self.panels[np.flatnonzero(cure_diff>=100.0)[-1]]
            if new_front_loc != self.front_loc:
                if self.front_has_started:
                    self.front_vel = (new_front_loc - self.front_loc) / (self.current_time - self.time_front_last_moved)
                else:
                    self.front_has_started = True
                self.time_front_last_moved = self.current_time
        else:
            new_front_loc = 0.0
            self.front_vel = 0.0
        self.front_loc = new_front_loc

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

    def get_state(self, temp_rate, action):
        # Check if the finite element solver is set up for a ppo or pd controller
        if self.for_pd:
            # Get the average temperature and temperature rates of self.num_panels/10 even segments across entire length
            average_temps = np.mean(np.resize(self.temp_panels,(self.num_panels//10, 10)),axis=1)
            average_temp_rates = np.mean(np.resize(temp_rate,(self.num_panels//10, 10)),axis=1)
            
            # Get the input location and magnitude rates
            input_location_rate = np.clip(action[0], -self.max_input_loc_rate, self.max_input_loc_rate)
            
            # Normalize and concatenate all substates
            state = np.concatenate((average_temps, average_temp_rates,
                                    [self.front_loc], [self.front_vel], 
                                    [self.input_location], [input_location_rate],
                                    [self.current_target_front_vel]))  
            
        else:
            # Get the average temperature of self.num_panels/10 even segments across entire length
            average_temps = np.mean(np.resize(self.temp_panels,(self.num_panels//10, 10)),axis=1)
            
            # Get the average temperature of 10 even segments across laser's area of effect
            laser_view = self.temp_panels[(np.argmin(abs(self.panels-self.input_location+self.radius_of_input))):(np.argmin(abs(self.panels-self.input_location-self.radius_of_input)) + 1)]
            laser_view = np.mean(np.resize(laser_view[0:-(len(laser_view)%10)],(10,len(laser_view)//10)),axis=1)
            
            # Normalize and concatenate all substates
            state = np.concatenate((average_temps/self.temperature_limit, 
                                    laser_view/self.temperature_limit,
                                    [self.front_loc/self.length], 
                                    [self.front_vel/self.current_target_front_vel], 
                                    [self.input_location/self.length], 
                                    [self.input_magnitude]))
            
        # Return the state
        return state

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

    def step_time(self):
        # Update the current time and check for simulation completion
        done = (self.current_time + self.time_step >= self.sim_duration)
        if not done:
            self.current_time = self.current_time + self.time_step
            self.current_index = self.current_index + 1
            self.current_target_front_vel = self.target_front_vel[self.current_index]
        
        return done

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
        
        # Reset the target definition
        self.target_front_vel = np.ones(int(self.sim_duration / self.time_step))*self.target
        if self.random_target:
            self.target_front_vel = np.ones(int(self.sim_duration / self.time_step))*self.target - 2.0*(np.random.rand()-0.5)*self.purturbation_scale
        if self.target_switch:
            switch_location = int((0.20*np.random.rand()+0.40) * (len(self.target_front_vel)-1))
            switch_vel = self.target_front_vel[switch_location] + 2.0*(np.random.rand()-0.5)*self.purturbation_scale
            self.target_front_vel[switch_location:]=switch_vel
        self.current_target_front_vel = self.target_front_vel[self.current_index]
        
        # Reset input
        self.input_location = np.random.choice(self.panels)
        self.input_magnitude = np.random.rand()
        
        # Reset temperature panels
        perturbation = self.initial_temp_delta * np.sin((np.random.randint(1,10) * np.pi * self.panels) / (self.length))
        self.temp_panels = self.initial_temperature + perturbation
        
        # Reset cure panels
        perturbation = self.initial_cure_delta * np.sin((np.random.randint(1,10) * np.pi * self.panels) / (self.length))
        self.cure_panels = self.initial_cure + perturbation
        
        # Reset input panels
        self.input_panels = self.input_magnitude * self.front_const * np.exp((self.panels - self.input_location)**2 * self.exp_const)
        self.input_panels[self.input_panels<0.01*self.max_input_mag] = 0.0
        
        # Reset front parameters
        self.front_loc = 0.0
        self.front_vel = 0.0
        self.time_front_last_moved = 0.0
        self.front_has_started=False
        
        # Return the initial state
        temp_rate = np.zeros(len(self.temp_panels))
        action = np.array([0.0, 0.0])
        return self.get_state(temp_rate, action)