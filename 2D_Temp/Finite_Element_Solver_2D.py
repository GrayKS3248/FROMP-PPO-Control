# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:39:40 2020

@author: Grayson Schaer
"""

import numpy as np

class FES():
    
    def __init__(self, loc_multiplier):
        
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
        self.initial_temperature = 294.15
        self.initial_temp_delta = 0.02 * self.initial_temperature
        
        # Boundary conditions
        self.htc = 9.0
        self.ambient_temperature = 294.15
        
        # Substrate physical parameters
        self.thermal_conductivity = 0.6
        self.density = 997.0
        self.specific_heat = 4182.0
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
        self.target_ref = 313.15
        self.target_temp = (self.target_ref + (2.0*(np.random.rand()) - 1.0) * 5.0)
        self.target_temp_mesh = self.target_temp * np.ones(self.temp_mesh.shape)
        self.temp_error = 0.0
        self.temp_error_max = 0.0
        self.temp_error_min = 0.0
        
        # Input magnitude parameters
        self.max_input_mag = 2.5e7
        self.max_input_mag_rate = self.time_step
        self.input_magnitude = np.random.rand()
        
        # Input distribution parameters
        self.radius_of_input = self.length / 10.0
        sigma = self.radius_of_input / (2.0*np.sqrt(np.log(10.0)))
        self.exp_const = -1.0 / (2.0 * sigma * sigma)
        self.coarseness = 1
        
        # Input location parameters
        self.movement_dirn = np.array([0.0, 1.0])
        self.x_dirn_movement = 1.0
        self.length_wise_dist = 0.0
        self.max_input_loc_rate = self.length * self.time_step
        self.input_location = np.array([np.random.choice(self.mesh_cens_x_cords[:,0]), np.random.choice(self.mesh_cens_y_cords[0,:])])
        self.loc_multiplier = loc_multiplier
        
        # Input panels
        self.input_mesh = self.input_magnitude * self.max_input_mag * np.exp(((self.mesh_cens_x_cords - self.input_location[0])**2 * self.exp_const) + 
                                                                                (self.mesh_cens_y_cords - self.input_location[1])**2 * self.exp_const)
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0
        
        # Reward constants
        self.max_reward = 2.0
        self.input_punishment_const = 0.05
        self.overage_punishment_const = 0.25
        
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
        # Determine the movement state
        at_top_edge = self.input_location[1] >= 0.98*self.width
        at_bottom_edge = self.input_location[1] <= 0.02*self.width
        at_right_edge = self.input_location[0] >= 0.98*self.length
        at_left_edge = self.input_location[0] <= 0.02*self.length
        going_up = self.movement_dirn[1] == 1.0
        going_down = self.movement_dirn[1] == -1.0
        going_right = self.movement_dirn[0] == 1.0
        going_left = self.movement_dirn[0] == -1.0
        time_to_switch = False
        
        # Update how long the input has been traversing lengthwise
        if going_right or going_left:
            self.length_wise_dist = self.length_wise_dist + self.max_input_loc_rate*self.time_step
            if self.length_wise_dist >= self.loc_multiplier * 2.0 * self.radius_of_input:
                time_to_switch = True
        else :
            self.length_wise_dist = 0.0
        
        # Determine the movement direction
        if (going_up and at_top_edge) or (going_down and at_bottom_edge):
            self.movement_dirn = np.array([self.x_dirn_movement, 0.0])
        elif (going_right and at_right_edge and at_top_edge) or (going_left and at_left_edge and at_top_edge):
            self.movement_dirn = np.array([0.0, -1.0])
            self.x_dirn_movement = -1.0 * self.x_dirn_movement
        elif (going_right and at_right_edge and at_bottom_edge) or (going_left and at_left_edge and at_bottom_edge):
            self.movement_dirn = np.array([0.0, 1.0])
            self.x_dirn_movement = -1.0 * self.x_dirn_movement
        elif (going_right and at_top_edge and time_to_switch) or (going_left and at_top_edge and time_to_switch):
            self.movement_dirn = np.array([0.0, -1.0])
        elif (going_right and at_bottom_edge and time_to_switch) or (going_left and at_bottom_edge and time_to_switch):
            self.movement_dirn = np.array([0.0, 1.0])
        
        # Update the input location based on the movement direction
        self.input_location = self.input_location + self.movement_dirn*self.max_input_loc_rate*self.time_step
        self.input_location.clip(np.array([0.0, 0.0]), np.array([self.length, self.width]), out=self.input_location)
        
        # Update the input's magnitude
        magnitude_command = action[0]
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

    def get_state(self):
        # Calculate average temperature of the laser's view
        if np.max(self.input_mesh) > 0.0:
            laser_view = np.mean(self.temp_mesh[self.input_mesh != 0.0])
        else:
            mesh = 0.02 * self.max_input_mag * np.exp(((self.mesh_cens_x_cords - self.input_location[0])**2 * self.exp_const) + 
                                                            (self.mesh_cens_y_cords - self.input_location[1])**2 * self.exp_const)
            mesh[mesh<0.01*self.max_input_mag] = 0.0
            laser_view = np.mean(self.temp_mesh[mesh != 0.0])
        
        # Normalize and concatenate all substates
        state = np.array([laser_view/self.target_temp, self.input_magnitude])
            
        # Return the state
        return state

    def get_reward(self):
        # Calculate the input punishment
        input_punishment = -self.input_punishment_const * self.max_reward * self.input_magnitude
        
        # Calculate the temperature overage punishment
        overage = (np.max(self.temp_mesh) / self.target_temp)
        overage_punishment = 0.0
        if overage >= 1.10:
            overage_punishment = -self.overage_punishment_const * self.max_reward * (overage-0.10)
        
        # Calculate the temperature field reward and punishment
        size = (self.num_vert_length-1)*(self.num_vert_width-1)
        ratio = self.temp_mesh / self.target_temp
        on_target = np.sum(np.logical_and((ratio>=0.995),(ratio<=1.005)))
        close_target = np.sum(np.logical_and((ratio>=0.99),(ratio<=1.01))) - on_target
        near_target = np.sum(np.logical_and((ratio>=0.975),(ratio<=1.025))) - close_target - on_target
        off_target = size - near_target - close_target - on_target
        on_target_reward = self.max_reward * (on_target / size)
        close_target_reward = 0.67 * self.max_reward * (close_target / size)
        near_target_reward = 0.25 * self.max_reward * (near_target / size)
        off_target_punishment = -0.50 * self.max_reward * (off_target / size)
        
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
        self.target_temp = (self.target_ref + (2.0*(np.random.rand()) - 1.0) * 5.0)
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
        self.movement_dirn = np.array([0.0, 1.0])
        self.x_dirn_movement = 1.0
        self.length_wise_dist = 0.0
        
        # Reset input panels
        self.input_mesh = self.input_magnitude * self.max_input_mag * np.exp(((self.mesh_cens_x_cords - self.input_location[0])**2 * self.exp_const) + 
                                                                                (self.mesh_cens_y_cords - self.input_location[1])**2 * self.exp_const)
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0
        
        # Return the initial state
        return self.get_state()