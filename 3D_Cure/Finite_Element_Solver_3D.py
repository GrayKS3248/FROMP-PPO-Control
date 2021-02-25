# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 09:39:40 2020

@author: Grayson Schaer
"""

import numpy as np

class FES():
    
    def __init__(self):
        ################################ USER SET PARAMETERS ################################
        # Simulation parameters
        self.random_target=False
        self.target_switch=False
        self.control=False
        self.trigger=True
        
        # Mesh parameters
        self.num_vert_length = 120  # Unitless
        self.num_vert_width = 24    # Unitless
        self.num_vert_depth = 12    # Unitless
        
        # Spatial parameters
        self.length = 0.05  # Meters
        self.width = 0.01   # Meters
        self.depth = 0.005  # Meters
        
        # Temporal parameters
        self.sim_duration = 240.0  # Seconds
        self.time_step = 0.1       # Seconds
        
        # Initial conditions
        self.initial_temperature = 278.15  # Kelvin
        self.initial_cure = 0.05           # Decimal Percent
        
        # Boundary conditions
        self.htc = 10.0                    # Watts / (Meter^2 * Kelvin)
        self.ambient_temperature = 294.15  # Kelvin
        
        # Problem definition
        self.temperature_limit = 523.15  # Kelvin
        self.target = 0.00015            # Meters / Second
        
        # Monomer physical parameters
        self.thermal_conductivity = 0.152     # Watts / Meter * Kelvin
        self.density = 980.0                  # Kilograms / Meter^3
        self.enthalpy_of_reaction = 352100.0  # Joules / Kilogram
        self.specific_heat = 1440.0           # Joules / Kilogram * Kelvin
        self.pre_exponential = 10**5.281      # 1 / Seconds
        self.activiation_energy = 51100.0     # Joules / Mol
        self.gas_const = 8.3144               # Joules / Mol * Kelvin
        self.model_fit_order = 1.927          # Unitless
        self.autocatalysis_const = 0.365      # Unitless
        
        # Input distribution parameters
        self.radius_of_input = 0.005       # Meters
        self.laser_power = 0.2             # Watts
        self.input_mag_percent_rate = 0.5  # Decimal Percent / Second
        self.max_input_loc_rate = 0.025    # Meters / Second
        
        # Set trigger conditions
        self.trigger_flux = 25500.0   # Watts / Meter^2
        self.trigger_time = 0.0       # Seconds
        self.trigger_duration = 10.0  # Seconds
        
        # NN Input conversion factors
        self.mag_scale = 0.0227        # Unitless
        self.mag_offset = 0.5          # Unitless
        self.loc_rate_scale = 2.70e-4  # Unitless
        self.loc_rate_offset = 0.0     # Unitless
        
        ################################ CALCULATED PARAMETERS ################################
        # Initiate simulation time and target velocity index
        self.current_time = 0.0
        self.current_index = 0
        
        # Define initial condition deltas
        self.initial_temp_delta = 0.01 * self.initial_temperature
        self.initial_cure_delta = 0.025 * self.initial_cure
        
        # Define randomizing scaling and problem type
        self.randomizing_scale = self.target/6.0
        
        # Calculate the target velocity temporal vector and define the current target
        self.target_front_vel = np.ones(int(self.sim_duration / self.time_step))*self.target
        if self.random_target:
            self.target_front_vel = np.ones(int(self.sim_duration / self.time_step))*self.target - 2.0*(np.random.rand()-0.5)*self.randomizing_scale
        if self.target_switch:
            switch_location = int((0.20*np.random.rand()+0.40) * (len(self.target_front_vel)-1))
            switch_vel = self.target_front_vel[switch_location] + 2.0*(np.random.rand()-0.5)*self.randomizing_scale
            self.target_front_vel[switch_location:]=switch_vel
        self.current_target_front_vel = self.target_front_vel[self.current_index]
        
        # Set trigger conditions
        if not self.trigger:
            self.trigger_flux = 0.0
            self.trigger_time = 0.0
            self.trigger_duration = 0.0
            
        # Calculate monomer physical parameters
        self.thermal_diffusivity = self.thermal_conductivity / (self.specific_heat * self.density)
        
        # Create mesh and calculate step size
        x_range = np.linspace(0.0,self.length,self.num_vert_length)
        y_range = np.linspace(0.0,self.width,self.num_vert_width)
        z_range = np.linspace(0.0,self.depth,self.num_vert_depth)
        self.mesh_x, self.mesh_y, self.mesh_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        self.x_step = self.mesh_x[1,0,0]
        self.y_step = self.mesh_y[0,1,0]
        self.z_step = self.mesh_z[0,0,1]
        
        # Init and perturb temperature mesh
        self.temp_mesh = np.ones(self.mesh_x.shape) * self.initial_temperature
        self.temp_mesh = self.temp_mesh + self.get_perturbation(self.temp_mesh, self.initial_temp_delta)
        
        # Init and perturb cure mesh
        self.cure_mesh = np.ones(self.mesh_x.shape) * self.initial_cure
        self.cure_mesh = self.cure_mesh + self.get_perturbation(self.cure_mesh, self.initial_cure_delta)
        
        # Init front mesh
        self.front_mesh = self.cure_mesh>=0.80
        
        # Front parameters
        self.front_loc = np.zeros((self.num_vert_width, self.num_vert_depth))
        self.front_vel = np.zeros((self.num_vert_width, self.num_vert_depth))
        self.time_front_last_moved = np.zeros((self.num_vert_width, self.num_vert_depth))
        self.front_has_started = np.zeros((self.num_vert_width, self.num_vert_depth))
        
        # Input magnitude parameters
        sigma = 0.329505114491*self.radius_of_input
        self.exp_const = -1.0 / (2.0 * sigma * sigma)
        x = np.linspace(0,self.radius_of_input,1000)
        y = 0.01**((x**2.0)/(self.radius_of_input**2.0))
        self.max_input_mag = self.laser_power/(4.0*np.trapz(y,x=x)**2.0)
        
        # Input location parameters
        self.min_input_x_loc = 0.0
        self.max_input_x_loc = self.length
        self.min_input_y_loc = 0.0
        self.max_input_y_loc = self.width
        
        # Reward constants
        self.max_reward = 2.0
        self.dist_punishment_const = 0.15
        self.front_rate_reward_const = 10.0*self.max_reward**(1.0/3.0)/(10.0)
        self.input_punishment_const = 0.10
        self.overage_punishment_const = 0.40
        self.integral_punishment_const = 0.10
        self.front_shape_const = 10.0 / self.width
        self.max_integral = self.length * self.width * self.depth * self.temperature_limit
        self.integral_delta = self.max_integral - self.length * self.width * self.depth * self.initial_temperature

        # Initiate input
        self.input_percent = np.random.rand()
        self.input_location = np.array([np.random.choice(self.mesh_x[:,0,0]), np.random.choice(self.mesh_y[0,:,0])])
        if self.control:
            self.max_input_mag = 0.0
            self.input_mag_percent_rate = 0.0
            self.input_percent = 0.0
            self.max_input_loc_rate = 0.0
            self.input_location = np.array([self.mesh_x[self.num_vert_length//2,0,0], self.mesh_y[0,self.num_vert_width//2,0]])
        
        # Initiate input wattage mesh
        self.input_mesh = self.input_percent*self.max_input_mag*np.exp(((self.mesh_x[:,:,0]-self.input_location[0])**2*self.exp_const) + 
                                                                        (self.mesh_y[:,:,0]-self.input_location[1])**2*self.exp_const)
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0
        
        # Simulation stability limit
        self.stab_lim = 10.0 * self.temperature_limit
        
    # Get smooth 3D perturbation over input fields
    # @ param - size_array: array used to determine size of output mesh
    # @ param - delta: maximum magnitude of perturbation
    # @ return - perturbation: array of size size_array with smooth continuous perturbation of magnitude delta
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
        min_z_bias = 2.0*np.random.rand()-1.0
        max_z_bias = 2.0*np.random.rand()-1.0
        
        # Determine size of perturbation field
        x_range = np.linspace(-2.0*min_mag+min_x_bias, 2.0*max_mag+max_x_bias, len(size_array[:,0,0]))
        y_range = np.linspace(-2.0*min_mag+min_y_bias, 2.0*max_mag+max_y_bias, len(size_array[0,:,0]))
        z_range = np.linspace(-2.0*min_mag+min_z_bias, 2.0*max_mag+max_z_bias, len(size_array[0,0,:]))
        x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        xyz = x * y * z
        
        # Calculate perturbation field
        perturbation = (mag_1*np.sin(1.0*xyz+bias_1) + mag_2*np.sin(2.0*xyz+bias_2) + mag_3*np.sin(3.0*xyz+bias_3))
        scale = np.max(abs(perturbation))
        perturbation = (delta * perturbation) / scale
        
        return perturbation

    # step_input: function used to update the input's location and magnitude based on the given action
    # @param - action: action in form [location_rate_x, location_rate_y, input_percent] from PPO agent
    def step_input(self, action):
        # Convert the raw PPO command to usable, clipped location rate commands
        cmd = self.loc_rate_offset + self.loc_rate_scale * action[0:2]
        cmd.clip(-self.max_input_loc_rate, self.max_input_loc_rate, out=cmd)
        
        # Update the input's location from the converted location rate commands
        self.input_location = self.input_location + cmd * self.time_step
        self.input_location.clip(np.array([0.0, 0.0]), np.array([self.length, self.width]), out=self.input_location)
        
        # Convert the raw PPO command to a usable, clipped input percent command
        input_percent_command = self.mag_offset + self.mag_scale * action[2]
        
        # Update the input's magnitude from the converted input percent command
        if input_percent_command > self.input_percent:
            self.input_percent = np.clip(min(self.input_percent + self.input_mag_percent_rate * self.time_step, input_percent_command), 0.0, 1.0)
        elif input_percent_command < self.input_percent:
            self.input_percent = np.clip(max(self.input_percent - self.input_mag_percent_rate * self.time_step, input_percent_command), 0.0, 1.0)
                
        # Update the input wattage mesh
        self.input_mesh = self.input_percent*self.max_input_mag*np.exp(((self.mesh_x[:,:,0]-self.input_location[0])**2*self.exp_const) + 
                                                                        (self.mesh_y[:,:,0]-self.input_location[1])**2*self.exp_const)
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0
        
    # step_cure: function that calculates the cure rate at every point in the 3D mesh and uses this data to update the cure
    # @return - cure_rate: The calcualted cure rate (percent / second) across entire 3D mesh
    def step_cure(self):
        # Get the cure rate across the entire field based on the cure kinetics
        cure_rate = ((self.pre_exponential * np.exp((-1.0 * self.activiation_energy) / (self.gas_const * self.temp_mesh))) *
                    ((1.0 - self.cure_mesh) ** self.model_fit_order) *
                    (1.0 + self.autocatalysis_const * self.cure_mesh))
        
        # Update the cure field using forward Euler method
        self.cure_mesh = self.cure_mesh + cure_rate * self.time_step
        self.cure_mesh[self.cure_mesh>1.0] = 1.0
        
        # Return the cure rate
        return cure_rate
    
    # step_front: Update the recorded front position and velocity. These measures span the y and z directions
    def step_front(self):
        # Calculate the new front mesh
        cure_mesh_xor = np.logical_xor(self.cure_mesh>=0.80, self.front_mesh)
        self.front_mesh = self.cure_mesh>=0.80
        
        # Find the furthest right points in each widthwise and depthwise row that meet the front criteria
        for i in range(self.num_vert_width):
            for j in range(self.num_vert_depth):
                # Record the previous front location and current front location to calculate front rate
                prev_loc = self.front_loc[i,j]
                new_loc = self.mesh_x[cure_mesh_xor[:,i,j],i,j]
                if len(new_loc) >= 1:
                    new_loc = new_loc[-1]
                else:
                    new_loc = prev_loc
                
                # If the front has moved, update the front start status, the front rate, time it last moved, and the front location
                if prev_loc != new_loc:
                    if not self.front_has_started[i,j]:
                        self.front_has_started[i,j] = True
                        self.time_front_last_moved[i,j] = self.current_time
                        self.front_loc[i,j] = new_loc
                    else:    
                        self.front_vel[i,j] = (new_loc - prev_loc) / (self.current_time - self.time_front_last_moved[i,j])
                        self.time_front_last_moved[i,j] = self.current_time
                        self.front_loc[i,j] = new_loc
                    
    # step_temperature: steps the temperature field based on the internal heat generation, conduction, trigger, input, and boundary conditions
    # @param - cure_rate: The rate of cure mesh in decimal percent per second
    def step_temperature(self, cure_rate):
        # Calculate the second derivatives of the temperature field with respect to x, y, and z
        T_t2 = 2.0*self.temp_mesh
        dT2_dx2 = (np.roll(self.temp_mesh,-1,axis=0) - T_t2 + np.roll(self.temp_mesh,1,axis=0)) / (self.x_step*self.x_step)
        dT2_dy2 = (np.roll(self.temp_mesh,-1,axis=1) - T_t2 + np.roll(self.temp_mesh,1,axis=1)) / (self.y_step*self.y_step)
        dT2_dz2 = (np.roll(self.temp_mesh,-1,axis=2) - T_t2 + np.roll(self.temp_mesh,1,axis=2)) / (self.z_step*self.z_step)
        
        # Calculate x total heat flux
        if self.current_time >= self.trigger_time and self.current_time < self.trigger_time + self.trigger_duration:
            left_flux = self.htc*(self.temp_mesh[0,:,:]-self.ambient_temperature) - self.trigger_flux
        else:
            left_flux = self.htc*(self.temp_mesh[0,:,:]-self.ambient_temperature)
        right_flux = self.htc*(self.temp_mesh[-1,:,:]-self.ambient_temperature)
        
        # Calculate y total heat flux
        front_flux = self.htc*(self.temp_mesh[:,0,:]-self.ambient_temperature)
        back_flux = self.htc*(self.temp_mesh[:,-1,:]-self.ambient_temperature)
        
        # Calculate z total heat flux
        top_flux = self.htc*(self.temp_mesh[:,:,0]-self.ambient_temperature) - self.input_mesh
        bottom_flux = self.htc*(self.temp_mesh[:,:,-1]-self.ambient_temperature)
        
        # Calculate boundary conditions
        dT2_dx2[ 0,:,:] = 2.0*(self.temp_mesh[ 1,:,:]-self.temp_mesh[ 0,:,:]-(self.x_step*left_flux  /self.thermal_conductivity))/(self.x_step*self.x_step)
        dT2_dx2[-1,:,:] = 2.0*(self.temp_mesh[-2,:,:]-self.temp_mesh[-1,:,:]-(self.x_step*right_flux /self.thermal_conductivity))/(self.x_step*self.x_step)
        dT2_dy2[:, 0,:] = 2.0*(self.temp_mesh[:, 1,:]-self.temp_mesh[:, 0,:]-(self.y_step*front_flux /self.thermal_conductivity))/(self.y_step*self.y_step)
        dT2_dy2[:,-1,:] = 2.0*(self.temp_mesh[:,-2,:]-self.temp_mesh[:,-1,:]-(self.y_step*back_flux  /self.thermal_conductivity))/(self.y_step*self.y_step)
        dT2_dz2[:,:, 0] = 2.0*(self.temp_mesh[:,:, 1]-self.temp_mesh[:,:, 0]-(self.z_step*top_flux   /self.thermal_conductivity))/(self.z_step*self.z_step)
        dT2_dz2[:,:,-1] = 2.0*(self.temp_mesh[:,:,-2]-self.temp_mesh[:,:,-1]-(self.z_step*bottom_flux/self.thermal_conductivity))/(self.z_step*self.z_step)
        
        # Calculate the temperature rate field
        temp_rate = self.thermal_diffusivity*(dT2_dx2+dT2_dy2+dT2_dz2)+(self.enthalpy_of_reaction*cure_rate)/self.specific_heat
        
        # Update the temperature field using forward Euler method
        self.temp_mesh = self.temp_mesh + temp_rate * self.time_step
            
        # Check for unstable growth
        if((self.temp_mesh >= self.stab_lim).any() or (self.temp_mesh <= 0.0).any()):
            raise RuntimeError('Unstable growth detected.')

    # blockshaped: splits a 2D input array into a set of evenly sized 2D blocks
    # @param - arr: The array to be blockshaped
    # @param - nrows: the total number of rows into which the input will be blockshaped
    # @param - ncols: the total number of columns into which the input will be blockshaped
    # @return - the blockshaped array
    def blockshaped(self, arr, nrows, ncols):
        h, w = arr.shape
        assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
        assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                   .swapaxes(1,2)
                   .reshape(-1, nrows, ncols))

    # get_state: Gets the state fed to PPO agent based on temperature, front location, front velocity, and the input
    # @return - state: The normalized state array
    def get_state(self):
        # Get the average temperature in even areas across entire field
        average_temps = np.mean(self.blockshaped(self.temp_mesh[:,:,0],(self.num_vert_length)//6,(self.num_vert_width)//4),axis=0)
        average_temps = average_temps.reshape(np.size(average_temps))
        
        # Find the x coords over which the laser can see
        x_loc = self.mesh_x[:,0,0] - self.input_location[0]
        x_min = np.argmin(abs(x_loc + self.radius_of_input))
        x_max = np.argmin(abs(x_loc - self.radius_of_input))
        x_max = x_max - (x_max-x_min)%5
        if x_max == x_min:
            if x_max - 5 >= 0 :
                x_min = x_max - 5
            else:
                x_max = x_min + 5
                
        # Find the x coords over which the laser can see
        y_loc = self.mesh_y[0,:,0] - self.input_location[1]
        y_min = np.argmin(abs(y_loc + self.radius_of_input))
        y_max = np.argmin(abs(y_loc - self.radius_of_input))
        y_max = y_max - (y_max-y_min)%5
        if y_max == y_min:
            if y_max - 5 >= 0 :
                y_min = y_max - 5
            else:
                y_max = y_min + 5
                
        # Calculate average temperature blocks (5X5) in laser view
        laser_view = np.mean(self.blockshaped(self.temp_mesh[x_min:x_max,y_min:y_max,0],5,5),axis=0)
        laser_view = laser_view.reshape(np.size(laser_view))
        
        # Compress front location and velocity data
        average_front_loc = np.mean(self.front_loc[:,0].reshape(self.num_vert_width//4,4),axis=1)
        average_front_vel = np.mean(self.front_vel[:,0].reshape(self.num_vert_width//4,4),axis=1)
        
        # Normalize and concatenate all substates
        state = np.concatenate((average_temps/self.temperature_limit, 
                                laser_view/self.temperature_limit,
                                average_front_loc/self.length, 
                                average_front_vel/self.current_target_front_vel, 
                                [self.input_location[0]/self.length], [self.input_location[1]/self.width],
                                [self.input_percent]))
        
        # Return the state
        return state

    # get_reward - Solves for the reward fed to the PPO agent based on the reward function parameters, temperature, and front velocity
    # @return - reward: the calculated reward
    def get_reward(self):
        # Calculate the punishments based on the temperature field, input strength, action, and overage
        if self.control:
            input_punishment = 0.0
            dist_punishment = 0.0
        else:
            input_punishment = -self.input_punishment_const * self.max_reward * self.input_percent
            dist_from_front = abs(np.mean(self.front_loc) - self.input_location[0])
            if dist_from_front <= 1.25*self.radius_of_input:
                dist_from_front = 0.0 
            else:
                dist_from_front = dist_from_front/self.length
            dist_punishment = -self.dist_punishment_const * self.max_reward * dist_from_front
        overage_punishment =  -self.overage_punishment_const * self.max_reward * (np.max(self.temp_mesh) >= self.temperature_limit)
        integral = np.trapz(self.temp_mesh,x=self.mesh_x,axis=0)
        integral = np.trapz(integral,x=self.mesh_y[0,:,:],axis=0)
        integral = np.trapz(integral,x=self.mesh_z[0,0,:],axis=0)
        integral_punishment = -self.integral_punishment_const * self.max_reward * (1.0 - (self.max_integral - integral) / (self.integral_delta))
        front_shape_punishment = -self.front_shape_const * np.mean(abs(self.front_loc-np.mean(self.front_loc)))
        punishment = input_punishment + integral_punishment + front_shape_punishment + dist_punishment + overage_punishment
        
        # Calculate the reward
        mean_front_vel_error = min(np.mean(abs(self.front_vel - self.current_target_front_vel) / (self.current_target_front_vel)), 1.0)
        front_rate_reward = ((1.0 - mean_front_vel_error) * self.front_rate_reward_const)**3.0
        
        # Sum reward and punishment
        reward = front_rate_reward + punishment

        # Return the calculated reward
        return reward

    # step_time: Steps the environments time and updates the target velocity
    # @return - done: Boolean that determines whether simulation is complete or not
    def step_time(self):
        # Update the current time and check for simulation completion
        done = self.current_index == len(self.target_front_vel) - 1
        if not done:
            self.current_time = self.current_time + self.time_step
            self.current_index = self.current_index + 1
            self.current_target_front_vel = self.target_front_vel[self.current_index]
        
        return done

    # step: Steps input, cure mesh, front, and temperature mesh through time. Gets resultant state and reward
    # @params - action: PPO calcualted action to be taken
    # @return - state: The normalized state array
    # @return - reward: the calculated reward
    def step(self, action):
        # Step the input, cure, front, and temperature
        self.step_input(action)
        cure_rate = self.step_cure()
        self.step_front()
        self.step_temperature(cure_rate)
        
        # Get state and reward
        state = self.get_state()
        reward = self.get_reward()
        
        # Step time
        done = self.step_time()
            
        # Return next state, reward for previous action, and whether simulation is complete or not
        return state, reward, done
    
    # reset: resets the environment for another trajectory
    # @return - state: The normalized state array of the reset environment
    def reset(self):
        # Initiate simulation time and target velocity index
        self.current_time = 0.0
        self.current_index = 0
        
        # Calculate the target velocity temporal vector and define the current target
        self.target_front_vel = np.ones(int(self.sim_duration / self.time_step))*self.target
        if self.random_target:
            self.target_front_vel = np.ones(int(self.sim_duration / self.time_step))*self.target - 2.0*(np.random.rand()-0.5)*self.randomizing_scale
        if self.target_switch:
            switch_location = int((0.20*np.random.rand()+0.40) * (len(self.target_front_vel)-1))
            switch_vel = self.target_front_vel[switch_location] + 2.0*(np.random.rand()-0.5)*self.randomizing_scale
            self.target_front_vel[switch_location:]=switch_vel
        self.current_target_front_vel = self.target_front_vel[self.current_index]
        
        # Init and perturb temperature mesh
        self.temp_mesh = np.ones(self.mesh_x.shape) * self.initial_temperature
        self.temp_mesh = self.temp_mesh + self.get_perturbation(self.temp_mesh, self.initial_temp_delta)
        
        # Init and perturb cure mesh
        self.cure_mesh = np.ones(self.mesh_x.shape) * self.initial_cure
        self.cure_mesh = self.cure_mesh + self.get_perturbation(self.cure_mesh, self.initial_cure_delta)
        
        # Init front mesh
        self.front_mesh = self.cure_mesh>=0.80
        
        # Front parameters
        self.front_loc = np.zeros((self.num_vert_width, self.num_vert_depth))
        self.front_vel = np.zeros((self.num_vert_width, self.num_vert_depth))
        self.time_front_last_moved = np.zeros((self.num_vert_width, self.num_vert_depth))
        self.front_has_started = np.zeros((self.num_vert_width, self.num_vert_depth))
        
        # Initiate input
        self.input_percent = np.random.rand()
        self.input_location = np.array([np.random.choice(self.mesh_x[:,0,0]), np.random.choice(self.mesh_y[0,:,0])])
        if self.control:
            self.max_input_mag = 0.0
            self.input_mag_percent_rate = 0.0
            self.input_percent = 0.0
            self.max_input_loc_rate = 0.0
            self.input_location = np.array([self.mesh_x[self.num_vert_length//2,0,0], self.mesh_y[0,self.num_vert_width//2,0]])
        
        # Initiate wattage input mesh
        self.input_mesh = self.input_percent*self.max_input_mag*np.exp(((self.mesh_x[:,:,0]-self.input_location[0])**2*self.exp_const) + 
                                                                        (self.mesh_y[:,:,0]-self.input_location[1])**2*self.exp_const)
        self.input_mesh[self.input_mesh<0.01*self.max_input_mag] = 0.0

        return self.get_state()