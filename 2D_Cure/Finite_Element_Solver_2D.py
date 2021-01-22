# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:39:40 2020

@author: Grayson Schaer
"""

import numpy as np

class FES():
    
    def __init__(self, for_pd=False, random_target=False, target_switch=False, control=False):
        
        # Environment spatial parameters 
        self.num_vert_length = 181
        self.num_vert_width = 31
        self.length = 0.06
        self.width = 0.01
        
        # Environment time parameters
        self.sim_duration = 360.0
        self.current_time = 0.0
        self.time_step = 0.1
        self.current_index = 0
        
        # Initial conditions
        self.initial_temperature = 278.15
        self.initial_temp_delta = 0.05 * self.initial_temperature
        self.initial_cure = 0.10
        self.initial_cure_delta = 0.05 * self.initial_cure
        
        # Boundary conditions
        self.htc = 6.0
        self.ambient_temperature = 294.15
        
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
        self.trigger_heat_rate_flux = 9000.0
        self.trigger_time = 0.0
        self.trigger_duration = 40.0
        
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
        self.thermal_diffusivity = self.thermal_conductivity / (self.specific_heat * self.density)
        
        # Create vertices of mesh
        self.mesh_verts_y_coords, self.mesh_verts_x_coords = np.meshgrid(np.linspace(0.0,self.width,self.num_vert_width), np.linspace(0.0,self.length,self.num_vert_length))
        
        # Define coordinates for center of each mesh panel
        width_start = self.mesh_verts_y_coords[0,1] / 2.0
        width_end = (self.width + self.mesh_verts_y_coords[0,-2]) / 2.0
        num_cen_width = self.num_vert_width - 1
        length_start = self.mesh_verts_x_coords[1,0] / 2.0
        length_end = (self.length + self.mesh_verts_x_coords[-2,0]) / 2.0
        num_cen_length = self.num_vert_length - 1
        self.mesh_cens_y_cords, self.mesh_cens_x_cords = np.meshgrid(np.linspace(width_start, width_end, num_cen_width), np.linspace(length_start, length_end, num_cen_length))
        self.x_step = self.mesh_cens_x_cords[1][0]
        self.y_step = self.mesh_cens_y_cords[0][1]
        
        # Init and perturb temperature and cure meshes
        self.temp_mesh = np.ones(self.mesh_cens_x_cords.shape) * self.initial_temperature
        self.temp_mesh = self.temp_mesh + self.get_perturbation(self.temp_mesh, self.initial_cure_delta)
        self.cure_mesh = np.ones(self.mesh_cens_x_cords.shape) * self.initial_cure
        self.cure_mesh = self.cure_mesh + self.get_perturbation(self.cure_mesh, self.initial_cure_delta)
        
        # Front parameters
        self.front_indices_y, self.front_indices_x = np.meshgrid(np.linspace(0,self.num_vert_width-2,self.num_vert_width-1), np.linspace(0,self.num_vert_length-3,self.num_vert_length-2))
        self.front_loc = np.zeros(self.num_vert_width-1)
        self.front_vel = np.zeros(self.num_vert_width-1)
        self.time_front_last_moved = np.zeros(self.num_vert_width-1)
        self.front_has_started = np.zeros(self.num_vert_width-1)
        
        # Input magnitude parameters
        if self.control:
            self.max_input_mag = 0.0
        else:
            self.max_input_mag = 2.5e7
        self.max_input_mag_rate = self.time_step
        self.input_magnitude = np.random.rand()
        self.mag_scale = 0.0227
        self.mag_offset = 0.5
        
        # Input distribution parameters
        self.radius_of_input = self.length / 10.0
        sigma = self.radius_of_input / (2.0*np.sqrt(np.log(10.0)))
        self.exp_const = -1.0 / (2.0 * sigma * sigma)
        
        # Input location parameters
        self.min_input_x_loc = self.mesh_cens_x_cords[0,0]
        self.max_input_x_loc = self.mesh_cens_x_cords[-1,0]
        self.min_input_y_loc = self.mesh_cens_y_cords[0,0]
        self.max_input_y_loc = self.mesh_cens_y_cords[0,-1]
        self.max_input_loc_rate = self.length * self.time_step
        self.input_location = np.array([np.random.choice(self.mesh_cens_x_cords[:,0]), np.random.choice(self.mesh_cens_y_cords[0,:])])
        self.loc_rate_scale = 2.70e-4
        self.loc_rate_offset = 0.0
        
        # Input panels
        self.input_mesh = self.input_magnitude * self.max_input_mag * np.exp(((self.mesh_cens_x_cords - self.input_location[0])**2 * self.exp_const) + 
                                                                                (self.mesh_cens_y_cords - self.input_location[1])**2 * self.exp_const)
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0
        
        # Reward constants
        self.max_reward = 2.0
        self.front_rate_reward_const = 10.0*self.max_reward**(1.0/3.0)/(6.82985986)
        self.input_punishment_const = 0.10
        self.overage_punishment_const = 25.0
        self.integral_punishment_const = 0.10
        self.front_shape_const = 10.0 / self.width
        mesh_len = self.mesh_cens_x_cords[-1,0] - self.mesh_cens_x_cords[0,0]
        mesh_wid = self.mesh_cens_y_cords[0,-1] - self.mesh_cens_y_cords[0,0]
        self.max_integral = mesh_len * mesh_wid * self.temperature_limit
        self.integral_delta = self.max_integral - mesh_len * mesh_wid * self.initial_temperature
        
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
            self.input_location[0] = input_x_location
            self.input_location[1] = input_y_location
            
            # Update input's magnitude
            magnitude_rate_command = np.clip(action[2], -self.max_input_mag_rate, self.max_input_mag_rate)
            self.input_magnitude = np.clip(self.input_magnitude + magnitude_rate_command * self.time_step, 0.0, 1.0)
        
        elif not self.control:
            # Update the input's position
            cmd = self.loc_rate_offset + self.loc_rate_scale * action[0:2]
            cmd.clip(-self.max_input_loc_rate, self.max_input_loc_rate, out=cmd)
            self.input_location = self.input_location + cmd * self.time_step
            self.input_location.clip(np.array([0.0, 0.0]), np.array([self.length, self.width]), out=self.input_location)
            
            # Update the input's magnitude
            magnitude_command = self.mag_offset + self.mag_scale * action[2]
            if magnitude_command > self.input_magnitude:
                self.input_magnitude = np.clip(min(self.input_magnitude + self.max_input_mag_rate, magnitude_command), 0.0, 1.0)
            elif magnitude_command < self.input_magnitude:
                self.input_magnitude = np.clip(max(self.input_magnitude - self.max_input_mag_rate, magnitude_command), 0.0, 1.0)
                
        # Use the actions to define input thermal rate across entire spacial field
        x=(self.mesh_cens_x_cords-self.input_location[0])
        y=(self.mesh_cens_y_cords-self.input_location[1])
        self.input_mesh = self.input_magnitude*self.max_input_mag * np.exp( self.exp_const*(x*x + y*y) )
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0
        
    def step_cure(self):
        # Get the cure rate across the entire field based on the cure kinetics
        cure_rate = ((self.pre_exponential * np.exp( (-self.activiation_energy) / (self.gas_const * self.temp_mesh) )) *
                    ((1 - self.cure_mesh) ** self.model_fit_order) *
                    (1 + self.autocatalysis_const * self.cure_mesh))
        
        # Update the cure field using forward Euler method
        self.cure_mesh = self.cure_mesh + cure_rate * self.time_step
        
        # Return the cure rate
        return cure_rate
    
    def step_front(self):
        # Calculate the spatial cure derivative
        front_mesh = self.cure_mesh[0:-1,:]>=0.80
        
        # Find the furthest right points in each row that meet the spatial cure derivative threshold
        for curr_row in range(self.num_vert_width-1):
            front_indices = self.front_indices_x[:,curr_row][front_mesh[:,curr_row]]
            if len(front_indices) > 0: 
                new_front_loc = self.mesh_cens_x_cords[int(front_indices[-1]), curr_row]
                    
                # Only update front rows that have moved
                if new_front_loc != self.front_loc[curr_row]:
                    
                    # If the front has already began, update the recorded front velocity
                    if self.front_has_started[curr_row] != 0.0:
                        self.front_vel[curr_row] = (new_front_loc - self.front_loc[curr_row]) / (self.current_time - self.time_front_last_moved[curr_row])
                        self.front_loc[curr_row] = new_front_loc
                        self.time_front_last_moved[curr_row] = self.current_time
                    
                    # If the front has not already started, mark the front as started
                    else:
                        self.front_has_started[curr_row] = 1.0
                        self.front_loc[curr_row] = new_front_loc
                        self.time_front_last_moved[curr_row] = self.current_time

    def step_temperature(self, cure_rate):
       
        # Calculate the trigger boundary condition
        trigger_q = np.zeros(self.temp_mesh[0,:].shape)
        if self.current_time >= self.trigger_time and self.current_time < self.trigger_time + self.trigger_duration:
            trigger_q[:] = self.trigger_heat_rate_flux
        
        # Calculate the heat transfer boundaray condition [W/m^3]
        left_dT_dx = -(self.htc / self.thermal_conductivity) * (self.ambient_temperature - self.temp_mesh[0,:]) - (trigger_q / self.thermal_conductivity)
        right_dT_dx = (self.htc / self.thermal_conductivity) * (self.ambient_temperature - self.temp_mesh[-1,:])
        top_dT_dy = (self.htc / self.thermal_conductivity) * (self.ambient_temperature - self.temp_mesh[:,-1])
        bottom_dT_dy = -(self.htc / self.thermal_conductivity) * (self.ambient_temperature - self.temp_mesh[:,0])
        
        # Calculate the first derivative of the temperature field with respect to x and append the boundary conditions
        dT_dx = np.zeros((self.num_vert_length,self.num_vert_width-1))
        dT_dx[0,:] = left_dT_dx
        dT_dx[-1,:] = right_dT_dx
        dT_dx[1:-1,:] = np.diff(self.temp_mesh,axis=0)/self.x_step
        
        # Calculate the second derivative of the temperature field with respect to x
        dT2_dx2 = np.diff(dT_dx,axis=0)/self.x_step
        
        # Calculate the first derivative of the temperature field with respect to y and append the boundary conditions
        dT_dy = np.zeros((self.num_vert_length-1,self.num_vert_width))
        dT_dy[:,0] = bottom_dT_dy
        dT_dy[:,-1] = top_dT_dy
        dT_dy[:,1:-1] = np.diff(self.temp_mesh,axis=1)/self.y_step
        
        # Calculate the second derivative of the temperature field with respect to y
        dT2_dy2 = np.diff(dT_dy,axis=1)/self.y_step
       
        # Calculate the temperature laplacian 
        del_sq_T = dT2_dx2 + dT2_dy2

        # Calculate the total interal heat rate density [W/m^3]
        cure_q = self.enthalpy_of_reaction * self.density * cure_rate
        total_q = cure_q + self.input_mesh
        
        # Calculate the temperature rate field
        temp_rate = self.thermal_diffusivity*del_sq_T + (self.thermal_diffusivity/self.thermal_conductivity)*total_q
        
        # Update the temperature field using forward Euler method
        self.temp_mesh = self.temp_mesh + temp_rate * self.time_step
            
        # Check for unstable growth
        if((self.temp_mesh >= self.stab_lim).any() or (self.temp_mesh <= -self.stab_lim).any()):
            raise RuntimeError('Unstable growth detected.')

        return temp_rate

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
            average_temps = np.mean(self.blockshaped(self.temp_mesh,(self.num_vert_length-1)//9,(self.num_vert_width-1)//5),axis=0)
            average_temps = average_temps.reshape(np.size(average_temps))
            average_temp_rates = np.mean(self.blockshaped(temp_rate,(self.num_vert_length-1)//9,(self.num_vert_width-1)//5),axis=0)
            average_temp_rates = average_temp_rates.reshape(np.size(average_temp_rates))
            
            # Compress front location and velocity data
            average_front_loc = np.mean(self.front_loc.reshape((self.num_vert_width-1)//5,5),axis=1)
            average_front_vel = np.mean(self.front_vel.reshape((self.num_vert_width-1)//5,5),axis=1)
            
            # Get the input location and magnitude rates
            input_x_location_rate = np.clip(action[0], -self.max_input_loc_rate, self.max_input_loc_rate)
            input_y_location_rate = np.clip(action[1], -self.max_input_loc_rate, self.max_input_loc_rate)
            
            # Normalize and concatenate all substates
            state = np.concatenate((average_temps, average_temp_rates,
                                    average_front_loc, average_front_vel, 
                                    self.input_location, [input_x_location_rate], [input_y_location_rate],
                                    [self.current_target_front_vel]))  
            
        else:
            # Get the average temperature in even areas across entire field
            average_temps = np.mean(self.blockshaped(self.temp_mesh,(self.num_vert_length-1)//9,(self.num_vert_width-1)//5),axis=0)
            average_temps = average_temps.reshape(np.size(average_temps))
            
            # Find the x coords over which the laser can see
            x_loc = self.mesh_cens_x_cords[:,0] - self.input_location[0]
            x_min = np.argmin(abs(x_loc + self.radius_of_input))
            x_max = np.argmin(abs(x_loc - self.radius_of_input))
            x_max = x_max - (x_max-x_min)%5
            if x_max == x_min:
                if x_max - 5 >= 0 :
                    x_min = x_max - 5
                else:
                    x_max = x_min + 5
                    
            # Find the x coords over which the laser can see
            y_loc = self.mesh_cens_y_cords[0,:] - self.input_location[1]
            y_min = np.argmin(abs(y_loc + self.radius_of_input))
            y_max = np.argmin(abs(y_loc - self.radius_of_input))
            y_max = y_max - (y_max-y_min)%5
            if y_max == y_min:
                if y_max - 5 >= 0 :
                    y_min = y_max - 5
                else:
                    y_max = y_min + 5
                    
            # Calculate average temperature blocks (5X5) in laser view
            laser_view = np.mean(self.blockshaped(self.temp_mesh[x_min:x_max,y_min:y_max],5,5),axis=0)
            laser_view = laser_view.reshape(np.size(laser_view))
            
            # Compress front location and velocity data
            average_front_loc = np.mean(self.front_loc.reshape((self.num_vert_width-1)//5,5),axis=1)
            average_front_vel = np.mean(self.front_vel.reshape((self.num_vert_width-1)//5,5),axis=1)
            
            # Normalize and concatenate all substates
            state = np.concatenate((average_temps/self.temperature_limit, 
                                    laser_view/self.temperature_limit,
                                    average_front_loc/self.length, 
                                    average_front_vel/self.current_target_front_vel, 
                                    [self.input_location[0]/self.length], [self.input_location[1]/self.width],
                                    [self.input_magnitude]))
            
        # Return the state
        return state

    def get_reward(self):
        # Calculate the punishments based on the temperature field, input strength, action, and overage
        if self.control:
            input_punishment = 0.0
        else:
            input_punishment = -self.input_punishment_const * self.max_reward * self.input_magnitude
        overage_punishment =  -self.overage_punishment_const * self.max_reward * (np.max(self.temp_mesh) >= self.temperature_limit)
        integral = np.trapz(self.temp_mesh, x=self.mesh_cens_x_cords, axis=0)
        integral = np.trapz(integral, x=self.mesh_cens_y_cords[0,:])
        integral_punishment = -self.integral_punishment_const * self.max_reward * (1.0 - (self.max_integral - integral) / (self.integral_delta))
        front_shape_punishment = -self.front_shape_const * np.mean(abs(self.front_loc-np.mean(self.front_loc)))
        punishment = max(input_punishment + overage_punishment + integral_punishment + front_shape_punishment, -self.max_reward)
        
        # Calculate the reward
        mean_front_vel_error = min(np.mean(abs(self.front_vel - self.current_target_front_vel) / (self.current_target_front_vel)), 1.0)
        front_rate_reward = ((0.682985986 - mean_front_vel_error) * self.front_rate_reward_const)**3.0
        
        # Sum reward and punishment
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
        temp_rate = self.step_temperature(cure_rate)
        
        # Get state and reward
        state = self.get_state(temp_rate, action)
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
        self.input_location = np.array([np.random.choice(self.mesh_cens_x_cords[:,0]), np.random.choice(self.mesh_cens_y_cords[0,:])])
        
        # Temperature and cure mesh
        self.temp_mesh = np.ones(self.mesh_cens_x_cords.shape) * self.initial_temperature
        self.temp_mesh = self.temp_mesh + self.get_perturbation(self.temp_mesh, self.initial_cure_delta)
        self.cure_mesh = np.ones(self.mesh_cens_x_cords.shape) * self.initial_cure
        self.cure_mesh = self.cure_mesh + self.get_perturbation(self.cure_mesh, self.initial_cure_delta)
        
        # Reset input panels
        self.input_mesh = self.input_magnitude * self.max_input_mag * np.exp(((self.mesh_cens_x_cords - self.input_location[0])**2 * self.exp_const) + 
                                                                                (self.mesh_cens_y_cords - self.input_location[1])**2 * self.exp_const)
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0
        
        # Front parameters
        self.front_loc = np.zeros(self.num_vert_width-1)
        self.front_vel = np.zeros(self.num_vert_width-1)
        self.time_front_last_moved = np.zeros(self.num_vert_width-1)
        self.front_has_started = np.zeros(self.num_vert_width-1)
        
        # Return the initial state
        temp_rate = np.zeros((len(self.temp_mesh), len(self.temp_mesh[0])))
        action = np.array([0.0, 0.0, 0.0])
        return self.get_state(temp_rate, action)