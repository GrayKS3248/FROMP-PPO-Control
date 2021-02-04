# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:39:40 2020

@author: Grayson Schaer
"""

import numpy as np

class FES():
    
    def __init__(self):
        
        # Environment spatial parameters 
        self.num_vert_length = 241
        self.num_vert_width = 41
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
        
        # Boundary conditions
        self.htc = 9.0
        self.ambient_temperature = 294.15
        
        # Monomer physical parameters
        self.thermal_conductivity = 0.152
        self.density = 980.0
        self.specific_heat = 1440.0
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
        self.temp_mesh = self.temp_mesh + self.get_perturbation(self.temp_mesh, self.initial_temp_delta)
        
        # Problem definition constants
        self.target_ref = 340.0
        self.target_temp = (self.target_ref + (2.0*(np.random.rand()) - 1.0) * 25.0)
        self.target_temp_mesh = self.target_temp * np.ones(self.temp_mesh.shape)
        self.temp_error = 0.0
        self.temp_error_max = 0.0
        self.temp_error_min = 0.0
        
        # Input magnitude parameters
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
        self.input_punishment_const = 0.01
        self.overage_punishment_const = 0.10
        
        # Simulation limits
        self.stab_lim = 10.0 * self.ambient_temperature

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

    def step_temperature(self):
        # Calculate the heat transfer boundaray condition [W/m^3]
        left_dT_dx = -(self.htc / self.thermal_conductivity) * (self.ambient_temperature - self.temp_mesh[0,:])
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
        
        # Calculate the temperature rate field
        temp_rate = self.thermal_diffusivity*del_sq_T + (self.thermal_diffusivity/self.thermal_conductivity)*self.input_mesh
        
        # Update the temperature field using forward Euler method
        self.temp_mesh = self.temp_mesh + temp_rate * self.time_step
            
        # Check for unstable growth
        if((self.temp_mesh >= self.stab_lim).any() or (self.temp_mesh <= -self.stab_lim).any()):
            raise RuntimeError('Unstable growth detected.')

    def blockshaped(self, arr, nrows, ncols):
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))

    def get_state(self):
        # Get the average temperature in even areas across entire field
        average_temps = np.mean(self.blockshaped(self.temp_mesh,(self.num_vert_length-1)//8,(self.num_vert_width-1)//8),axis=0)
        average_temps = average_temps.reshape(np.size(average_temps))
        
        # Find the x coords over which the laser can see
        x_loc = self.mesh_cens_x_cords[:,0] - self.input_location[0]
        x_min = np.argmin(abs(x_loc + self.radius_of_input))
        x_max = np.argmin(abs(x_loc - self.radius_of_input))
        x_max = x_max - (x_max-x_min)%7
        if x_max == x_min:
            if x_max - 7 >= 0 :
                x_min = x_max - 7
            else:
                x_max = x_min + 7
                
        # Find the x coords over which the laser can see
        y_loc = self.mesh_cens_y_cords[0,:] - self.input_location[1]
        y_min = np.argmin(abs(y_loc + self.radius_of_input))
        y_max = np.argmin(abs(y_loc - self.radius_of_input))
        y_max = y_max - (y_max-y_min)%7
        if y_max == y_min:
            if y_max - 7 >= 0 :
                y_min = y_max - 7
            else:
                y_max = y_min + 7
                
        # Calculate average temperature blocks (5X5) in laser view
        laser_view = np.mean(self.blockshaped(self.temp_mesh[x_min:x_max,y_min:y_max],7,7),axis=0)
        laser_view = laser_view.reshape(np.size(laser_view))
        
        # Normalize and concatenate all substates
        state = np.concatenate((average_temps/self.target_temp, 
                                laser_view/self.target_temp,
                                [self.input_location[0]/self.length], [self.input_location[1]/self.width],
                                [self.input_magnitude]))
            
        # Return the state
        return state

    def get_reward(self):
        # Calculate the input punishment
        input_punishment = -self.input_punishment_const * self.max_reward * self.input_magnitude
        
        # Calculate the temperature overage punishment
        overage = (np.max(self.temp_mesh) / self.target_temp)
        overage_punishment = 0.0
        if overage >= 1.15:
            overage_punishment = -self.overage_punishment_const * self.max_reward * (overage-0.15)
        
        # Calculate the temperature field reward and punishment
        size = (self.num_vert_length-1)*(self.num_vert_width-1)
        on_target = np.sum(((((self.temp_mesh > 0.99 * self.target_temp) * 1.0 + (self.temp_mesh < 1.01 * self.target_temp) * 1.0) == 2.0) * 1.0))
        close_target = np.sum(((((self.temp_mesh > 0.95 * self.target_temp) * 1.0 + (self.temp_mesh < 1.05 * self.target_temp) * 1.0) == 2.0) * 1.0)) - on_target
        near_target = np.sum(((((self.temp_mesh > 0.90 * self.target_temp) * 1.0 + (self.temp_mesh < 1.10 * self.target_temp) * 1.0) == 2.0) * 1.0)) - close_target - on_target
        off_target = size - near_target - close_target - on_target
        on_target_reward = self.max_reward * (on_target / size)
        close_target_reward = 0.75 * self.max_reward * (close_target / size)
        near_target_reward = 0.25 * self.max_reward * (near_target / size)
        off_target_punishment = -0.25 * self.max_reward * (off_target / size)
        
        # Sum reward and punishment
        reward = on_target_reward + close_target_reward + near_target_reward + off_target_punishment + overage_punishment + input_punishment

        # Calculate the errors
        difference = self.temp_mesh/self.target_temp - 1.0
        self.temp_error = np.mean(difference)
        self.temp_error_max = np.max(difference)
        self.temp_error_min = np.min(difference)
        
        # Return the calculated reward
        return reward

    def step_time(self):
        # Update the current time and check for simulation completion
        done = (self.current_time + self.time_step >= self.sim_duration)
        if not done:
            self.current_time = self.current_time + self.time_step
            self.current_index = self.current_index + 1
        
        return done

    def step(self, action):
        # Step the input, cure, front, and temperature
        self.step_input(action)
        self.step_temperature()
        
        # Get state and reward
        state = self.get_state()
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
        self.target_temp = (self.target_ref + (2.0*(np.random.rand()) - 1.0) * 25.0)
        self.target_temp_mesh = self.target_temp * np.ones(self.temp_mesh.shape)
        self.temp_error = 0.0
        self.temp_error_max = 0.0
        self.temp_error_min = 0.0
        
        # Init and perturb temperature and cure meshes
        self.temp_mesh = np.ones(self.mesh_cens_x_cords.shape) * self.initial_temperature
        self.temp_mesh = self.temp_mesh + self.get_perturbation(self.temp_mesh, self.initial_temp_delta)
        
        # Reset input
        self.input_magnitude = np.random.rand()
        self.input_location = np.array([np.random.choice(self.mesh_cens_x_cords[:,0]), np.random.choice(self.mesh_cens_y_cords[0,:])])
        
        # Reset input panels
        self.input_mesh = self.input_magnitude * self.max_input_mag * np.exp(((self.mesh_cens_x_cords - self.input_location[0])**2 * self.exp_const) + 
                                                                                (self.mesh_cens_y_cords - self.input_location[1])**2 * self.exp_const)
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0
        
        # Return the initial state
        return self.get_state()