# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:39:40 2020

@author: Grayson Schaer
"""

import numpy as np

class FES():
    
    def __init__(self, control=False, loc_multiplier=0.40, uniform_target=True, split_target=False, random_target=False):
        
        # Ensure inputs are ok
        if loc_multiplier <= 0.0:
            raise RuntimeError('Location multiplier must be greater than or equal to 0.0')
        if uniform_target and (split_target or random_target):
            raise RuntimeError('Split and random target mode must be false when uniform target mode is true.')
        if split_target and (uniform_target or random_target):
            raise RuntimeError('Uniform and random target mode must be false when split target mode is true.')
        if random_target and (split_target or uniform_target):
            raise RuntimeError('Uniform and split target mode must be false when random target mode is true.')
            
        # Environment spatial parameters 
        self.num_vert_length = 181
        self.num_vert_width = 181
        self.length = 0.03
        self.width = 0.03
        
        # Environment time parameters
        self.sim_duration = 30.0
        self.current_time = 0.0
        self.time_step = 0.01
        self.current_index = 0
        
        # Initial conditions
        self.initial_temperature = 294.15
        self.initial_temp_delta = 0.0025 * self.initial_temperature
        
        # Boundary conditions
        self.htc = 7.0
        self.ambient_temperature = 294.15
        
        # Substrate physical parameters
        self.thermal_conductivity = 1.0
        self.density = 1000.0
        self.specific_heat = 2000.0
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
        
        # Constants set to divide mesh
        self.x_cens = np.round(np.linspace(0,self.num_vert_length-2,21)).astype(int)
        self.x_cens = self.x_cens[1:-1:2]
        self.y_cens = np.round(np.linspace(0,self.num_vert_width-2,21)).astype(int)
        self.y_cens = self.y_cens[1:-1:2]
        self.x_coords = np.round(np.linspace(0,self.num_vert_length-2,11)).astype(int)
        self.y_coords = np.round(np.linspace(0,self.num_vert_width-2,11)).astype(int)
        self.avg_error = np.zeros((10,10))
        
        # Init and perturb temperature and cure meshes
        self.temp_mesh = np.ones(self.mesh_cens_x_cords.shape) * self.initial_temperature
        self.temp_mesh = self.temp_mesh + self.get_perturbation(self.temp_mesh, self.initial_temp_delta)
        
        # Problem definition constants
        self.target_ref = 304.15
        self.uniform_target = uniform_target
        self.split_target = split_target
        self.random_target = random_target
        if self.uniform_target:
            target_temp = (self.target_ref + np.random.rand() * 3.0)
            self.target_temp_mesh = target_temp * np.ones(self.temp_mesh.shape)
        elif self.split_target:
            target_temp_1 = self.target_ref + np.random.rand()*3.0
            target_temp_2 = self.target_ref + np.random.rand()*3.0
            split_point = round((0.25 * np.random.rand() + 0.375)*(self.num_vert_length-1))
            mesh_1 = np.ones(self.temp_mesh.shape)
            mesh_1[split_point:len(mesh_1)] = 0.0
            mesh_2 = np.ones(self.temp_mesh.shape)
            mesh_2[0:split_point] = 0.0
            self.target_temp_mesh = target_temp_1 * mesh_1 + target_temp_2 * mesh_2
        elif self.random_target:
            self.target_temp_mesh = self.target_ref * np.ones(self.temp_mesh.shape)
            self.target_temp_mesh = self.target_temp_mesh + self.get_perturbation(self.target_temp_mesh, 3.0)
        self.temp_error = 0.0
        self.temp_error_max = 0.0
        self.temp_error_min = 0.0
        
        # Input magnitude parameters
        self.control = control
        self.max_input_mag = 2.0e7
        if self.control:
            self.max_input_mag_rate = 0.0
            self.input_magnitude = 1.0
        else:
            self.max_input_mag_rate = 2.0
            self.input_magnitude = np.random.rand()
        
        # Input distribution parameters
        self.radius_of_input = 0.013
        sigma = self.radius_of_input / (2.0*np.sqrt(np.log(10.0)))
        self.exp_const = -1.0 / (2.0 * sigma * sigma)
        
        # Input location parameters
        self.half_dist = False
        self.movement_dirn = np.array([0.0, 1.0])
        self.x_dirn_movement = 1.0
        self.length_wise_dist = 0.0
        self.loc_multiplier = loc_multiplier
        if self.control:
            self.max_input_loc_rate = 0.0
            self.input_location = np.array([self.mesh_cens_x_cords[(self.num_vert_length-1)//2,0], self.mesh_cens_y_cords[0,(self.num_vert_width-1)//2]])
        else:
            self.max_input_loc_rate = 0.05
            self.input_location = np.array([np.random.choice(self.mesh_cens_x_cords[:,0]), np.random.choice(self.mesh_cens_y_cords[0,:])])
        
        # Input panels
        self.input_mesh = self.input_magnitude * self.max_input_mag * np.exp(((self.mesh_cens_x_cords - self.input_location[0])**2 * self.exp_const) + 
                                                                                (self.mesh_cens_y_cords - self.input_location[1])**2 * self.exp_const)
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0
        
        # Determine target position
        error = abs(self.temp_mesh - self.target_temp_mesh)
        for i in range(10):
            for j in range(10):
                self.avg_error[i][j] = np.mean(error[self.x_coords[i]:self.x_coords[i+1], self.y_coords[j]:self.y_coords[j+1]])
        index = np.unravel_index(np.argmax(self.avg_error), self.avg_error.shape)
        self.target_pos = np.array([self.mesh_cens_x_cords[self.x_cens[index[0]],self.y_cens[index[1]]], self.mesh_cens_y_cords[self.x_cens[index[0]],self.y_cens[index[1]]]])
        
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
        if True:
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
                if (self.length_wise_dist>=self.loc_multiplier*2.0*self.radius_of_input) or (self.half_dist and self.length_wise_dist>=self.loc_multiplier*self.radius_of_input):
                    time_to_switch = True
                    if self.half_dist:
                        self.half_dist = False
            else :
                self.length_wise_dist = 0.0
            
            # Determine the movement direction
            if (going_up and at_top_edge) or (going_down and at_bottom_edge):
                self.movement_dirn = np.array([self.x_dirn_movement, 0.0])
            elif (going_right and at_right_edge and at_top_edge) or (going_left and at_left_edge and at_top_edge):
                self.movement_dirn = np.array([0.0, -1.0])
                self.x_dirn_movement = -1.0 * self.x_dirn_movement
                if going_right:
                    self.half_dist = True
            elif (going_right and at_right_edge and at_bottom_edge) or (going_left and at_left_edge and at_bottom_edge):
                self.movement_dirn = np.array([0.0, 1.0])
                self.x_dirn_movement = -1.0 * self.x_dirn_movement
                if going_right:
                    self.half_dist = True
            elif (going_right and at_top_edge and time_to_switch) or (going_left and at_top_edge and time_to_switch):
                self.movement_dirn = np.array([0.0, -1.0])
            elif (going_right and at_bottom_edge and time_to_switch) or (going_left and at_bottom_edge and time_to_switch):
                self.movement_dirn = np.array([0.0, 1.0])
            
            # Update the input location based on the movement direction
            self.input_location = self.input_location + self.movement_dirn*self.max_input_loc_rate*self.time_step
            self.input_location.clip(np.array([0.0, 0.0]), np.array([self.length, self.width]), out=self.input_location)
        
        else:
            # Average the temperature field into 10x10 field
            error = abs(self.temp_mesh - self.target_temp_mesh)
            for i in range(10):
                for j in range(10):
                    self.avg_error[i][j] = np.mean(error[self.x_coords[i]:self.x_coords[i+1], self.y_coords[j]:self.y_coords[j+1]])
            index = np.unravel_index(np.argmax(self.avg_error), self.avg_error.shape)
            if np.linalg.norm(self.target_pos - self.input_location) <= self.radius_of_input/4.0:
                self.target_pos = np.array([self.mesh_cens_x_cords[self.x_cens[index[0]],self.y_cens[index[1]]], self.mesh_cens_y_cords[self.x_cens[index[0]],self.y_cens[index[1]]]])
            target_dirn = self.target_pos - self.input_location
            target_dirn = target_dirn / np.linalg.norm(target_dirn)
            target_dirn[np.argmax(abs(target_dirn))] = 1.0 if target_dirn[np.argmax(abs(target_dirn))]>0.0 else -1.0
            target_dirn[np.argmin(abs(target_dirn))] = 0.0
            
             # Update the input location based on the lowest temperature region
            self.input_location = self.input_location + target_dirn*self.max_input_loc_rate*self.time_step
            self.input_location.clip(np.array([0.0, 0.0]), np.array([self.length, self.width]), out=self.input_location)
        
        # Update the input's magnitude
        magnitude_command = action[0]
        if magnitude_command > self.input_magnitude:
            self.input_magnitude = np.clip(min(self.input_magnitude + self.max_input_mag_rate*self.time_step, magnitude_command), 0.0, 1.0)
        elif magnitude_command < self.input_magnitude:
            self.input_magnitude = np.clip(max(self.input_magnitude - self.max_input_mag_rate*self.time_step, magnitude_command), 0.0, 1.0)
                
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
        
        # Calculate the heat loss in the z direction
        bulk_loss = ((self.htc / self.thermal_conductivity) * (self.ambient_temperature - self.temp_mesh)) / 5.0e-4
        bulk_loss[0:2,:] = 0.0
        bulk_loss[-2:len(bulk_loss),:] = 0.0
        bulk_loss[:,0:2] = 0.0
        bulk_loss[:,-2:len(bulk_loss[0,:])] = 0.0
        
        # Calculate the temperature rate field
        temp_rate = self.thermal_diffusivity*del_sq_T + (self.thermal_diffusivity/self.thermal_conductivity) * (self.input_mesh + bulk_loss)
        
        # Update the temperature field using forward Euler method
        self.temp_mesh = self.temp_mesh + temp_rate * self.time_step
            
        # Check for unstable growth
        if((self.temp_mesh >= self.stab_lim).any() or (self.temp_mesh <= -self.stab_lim).any()):
            raise RuntimeError('Unstable growth detected.')

    def get_state(self):
        # Calculate average temperature of the laser's view
        normalized_temp_mesh = self.temp_mesh / self.target_temp_mesh
        if np.max(self.input_mesh) > 0.0:
            laser_view = np.mean(normalized_temp_mesh[self.input_mesh != 0.0])
        else:
            mesh = 0.02 * self.max_input_mag * np.exp(((self.mesh_cens_x_cords - self.input_location[0])**2 * self.exp_const) + 
                                                            (self.mesh_cens_y_cords - self.input_location[1])**2 * self.exp_const)
            mesh[mesh<0.01*self.max_input_mag] = 0.0
            laser_view = np.mean(normalized_temp_mesh[mesh != 0.0])
        
        # Normalize and concatenate all substates
        state = np.array([laser_view, self.input_magnitude])
            
        # Return the state
        return state

    def get_reward(self):
        # Calculate the input punishment
        input_punishment = -self.input_punishment_const * self.max_reward * self.input_magnitude
        
        # Calculate the temperature overage punishment
        overage = np.max(self.temp_mesh / self.target_temp_mesh)
        overage_punishment = 0.0
        if overage >= 1.10:
            overage_punishment = -self.overage_punishment_const * self.max_reward * (overage-0.10)
        
        # Calculate the temperature field reward and punishment
        size = (self.num_vert_length-1)*(self.num_vert_width-1)
        average_percent_difference = np.sum(abs((self.target_temp_mesh-self.temp_mesh)/(self.target_temp_mesh-self.initial_temperature)))/size
        target_reward = self.max_reward * ((-1.666667 * average_percent_difference + 1)**3.0)
        
        # Sum reward and punishment
        reward = target_reward + overage_punishment + input_punishment

        # Calculate the errors
        difference = self.temp_mesh/self.target_temp_mesh - 1.0
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
        if self.uniform_target:
            target_temp = self.target_ref + np.random.rand() * 3.0
            self.target_temp_mesh = target_temp * np.ones(self.temp_mesh.shape)
        elif self.split_target:
            target_temp_1 = self.target_ref + np.random.rand()*3.0
            target_temp_2 = self.target_ref + np.random.rand()*3.0
            split_point = round((0.25 * np.random.rand() + 0.375)*(self.num_vert_length-1))
            mesh_1 = np.ones(self.temp_mesh.shape)
            mesh_1[split_point:len(mesh_1)] = 0.0
            mesh_2 = np.ones(self.temp_mesh.shape)
            mesh_2[0:split_point] = 0.0
            self.target_temp_mesh = target_temp_1 * mesh_1 + target_temp_2 * mesh_2
        elif self.random_target:
            self.target_temp_mesh = self.target_ref * np.ones(self.temp_mesh.shape)
            self.target_temp_mesh = self.target_temp_mesh + self.get_perturbation(self.target_temp_mesh, 3.0)
        self.temp_error = 0.0
        self.temp_error_max = 0.0
        self.temp_error_min = 0.0
        
        # Init and perturb temperature and cure meshes
        self.temp_mesh = np.ones(self.mesh_cens_x_cords.shape) * self.initial_temperature
        self.temp_mesh = self.temp_mesh + self.get_perturbation(self.temp_mesh, self.initial_temp_delta)
        
        # Reset input
        self.half_dist = False
        if self.control:
            self.input_magnitude = 1.0
            self.input_location = np.array([self.mesh_cens_x_cords[(self.num_vert_length-1)//2,0], self.mesh_cens_y_cords[0,(self.num_vert_width-1)//2]])
        else:
            self.input_magnitude = np.random.rand()
            self.input_location = np.array([np.random.choice(self.mesh_cens_x_cords[:,0]), np.random.choice(self.mesh_cens_y_cords[0,:])])
        self.movement_dirn = np.array([0.0, 1.0])
        self.x_dirn_movement = 1.0
        self.length_wise_dist = 0.0
        self.avg_error = np.zeros((10,10))
        
        # Reset input panels
        self.input_mesh = self.input_magnitude * self.max_input_mag * np.exp(((self.mesh_cens_x_cords - self.input_location[0])**2 * self.exp_const) + 
                                                                                (self.mesh_cens_y_cords - self.input_location[1])**2 * self.exp_const)
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0
        
        # Determine target position
        error = abs(self.temp_mesh - self.target_temp_mesh)
        for i in range(10):
            for j in range(10):
                self.avg_error[i][j] = np.mean(error[self.x_coords[i]:self.x_coords[i+1], self.y_coords[j]:self.y_coords[j+1]])
        index = np.unravel_index(np.argmax(self.avg_error), self.avg_error.shape)
        self.target_pos = np.array([self.mesh_cens_x_cords[self.x_cens[index[0]],self.y_cens[index[1]]], self.mesh_cens_y_cords[self.x_cens[index[0]],self.y_cens[index[1]]]])
        
        # Return the initial state
        return self.get_state()