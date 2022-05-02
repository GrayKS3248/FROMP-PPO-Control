# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:36:17 2022

@author: GKSch
"""
import numpy as np
import control
import scipy.ndimage.interpolation as inp
import scipy.ndimage as nd
from scipy import signal

class Controller:
    
    # Initializes the linear optimal controller by solving the LQR minimization problem for the heat equation
    # @param Thermal conductivity of the material in [Watts / Meter * Kelvin]
    # @param Density of the material in [Kilograms / Meter ^ 3]
    # @param Specific heat capacity of the material in [Joules / Kilogram * Kelvin]
    # @param Material's heat transfer coefficient in [Watts / Meter^2 * Kelvin]
    # @param The transverse size of the temperature grid in [Meter]
    # @param The longitudinal size of the temperature grid in [Meter]
    # @param Thickness of the material in [Meter]
    # @param Total number of nodes in the temperature grid
    # @param Relative importance of minimizing input movement compared to minimizing local input vs global optimal input
    # @param Stdev of gaussian kernal used to weight Q matrix towards edges
    def __init__(self, thermal_conductivity, density, specific_heat, htc, tamb, width, length, thickness, state_size, movement_const, sigma):
        
        # Calculate number of transverse and longitudinal nodes
        temporary = np.sqrt(width*width + 2.0*length*(2.0*state_size-1)*width + length*length)
        self.num_vert_y = int(round((temporary - width + length) / (2.0*length)))
        self.num_vert_x = int(round((temporary + width - length) / (2.0*width)))
        
        # Define size of temperature grid
        self.width = width
        self.length = length
        
        # Define input parameters used by local controller
        self.movement_const = movement_const
        self.temp_step_size_y = -10.0
        self.temp_step_size_x = -10.0
        self.radius = -10.0
        self.power = -10.0
        self.max_input = -10.0
        self.kernal = np.array([])
        self.grid_x = np.array([[]])
        self.grid_y = np.array([[]])
        
        # Calculate step sizes of the controller field
        delta_y = width / (self.num_vert_y - 1)
        delta_x = length / (self.num_vert_x - 1)
        delta_z = thickness
        
        # Generate the laplcian matrix
        self.laplacian_matrix = np.zeros((self.num_vert_y*self.num_vert_x,self.num_vert_y*self.num_vert_x))
        self.laplacian_vector = np.zeros((self.num_vert_y*self.num_vert_x,1))
        
        # Populate the laplacian matrix
        for i in np.arange(self.num_vert_y*self.num_vert_x):
            
            # Get coordinate of current point on which laplacian is being calculated
            y_coord = i % self.num_vert_y
            x_coord = i // self.num_vert_y
            
            # Get the stencil indices
            ind_00 = i
            ind_0m = i - self.num_vert_y
            ind_0p = i + self.num_vert_y
            ind_m0 = i - 1
            ind_p0 = i + 1
            
            # Top edge
            if y_coord==0 and x_coord!=0 and x_coord!=self.num_vert_x-1:
                # Second derivative in the y direction
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_y*delta_y))*(1.0-(htc*delta_y)/thermal_conductivity)
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_p0] = self.laplacian_matrix[i][ind_p0] + (1.0/(delta_y*delta_y))
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_y)
                
                # Second derivative in the x direction
                self.laplacian_matrix[i][ind_0m] = self.laplacian_matrix[i][ind_0m] + (1.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_0p] = self.laplacian_matrix[i][ind_0p] + (1.0/(delta_x*delta_x))
                
            # Bottom edge
            elif y_coord==self.num_vert_y-1 and x_coord!=0 and x_coord!=self.num_vert_x-1:
                # Second derivative in the y direction
                self.laplacian_matrix[i][ind_m0] = self.laplacian_matrix[i][ind_m0] + (1.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_y*delta_y))*(1.0-(htc*delta_y)/thermal_conductivity)
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_y)
                
                # Second derivative in the x direction
                self.laplacian_matrix[i][ind_0m] = self.laplacian_matrix[i][ind_0m] + (1.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_0p] = self.laplacian_matrix[i][ind_0p] + (1.0/(delta_x*delta_x))
                
            # Left edge
            elif x_coord==0 and y_coord!=0 and y_coord!=self.num_vert_y-1:
                # Second derivative in the y direction
                self.laplacian_matrix[i][ind_m0] = self.laplacian_matrix[i][ind_m0] + (1.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_p0] = self.laplacian_matrix[i][ind_p0] + (1.0/(delta_y*delta_y))
                
                # Second derivative in the x direction
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_x*delta_x))*(1.0-(htc*delta_x)/thermal_conductivity)
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_0p] = self.laplacian_matrix[i][ind_0p] + (1.0/(delta_x*delta_x))
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_x)
                
            # Right edge
            elif x_coord==self.num_vert_x-1 and y_coord!=0 and y_coord!=self.num_vert_y-1:
                # Second derivative in the y direction
                self.laplacian_matrix[i][ind_m0] = self.laplacian_matrix[i][ind_m0] + (1.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_p0] = self.laplacian_matrix[i][ind_p0] + (1.0/(delta_y*delta_y))
                
                # Second derivative in the x direction
                self.laplacian_matrix[i][ind_0m] = self.laplacian_matrix[i][ind_0m] + (1.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_x*delta_x))*(1.0-(htc*delta_x)/thermal_conductivity)
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_x)
                
            # Top left corner
            elif y_coord==0 and x_coord==0:
                # Second derivative in the y direction
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_y*delta_y))*(1.0-(htc*delta_y)/thermal_conductivity)
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_p0] = self.laplacian_matrix[i][ind_p0] + (1.0/(delta_y*delta_y))
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_y)
                
                # Second derivative in the x direction
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_x*delta_x))*(1.0-(htc*delta_x)/thermal_conductivity)
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_0p] = self.laplacian_matrix[i][ind_0p] + (1.0/(delta_x*delta_x))
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_x)
                
            # Top right corner
            elif y_coord==0 and x_coord==self.num_vert_x-1:
                # Second derivative in the y direction
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_y*delta_y))*(1.0-(htc*delta_y)/thermal_conductivity)
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_p0] = self.laplacian_matrix[i][ind_p0] + (1.0/(delta_y*delta_y))
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_y)
                
                # Second derivative in the x direction
                self.laplacian_matrix[i][ind_0m] = self.laplacian_matrix[i][ind_0m] + (1.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_x*delta_x))*(1.0-(htc*delta_x)/thermal_conductivity)
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_x)
                
            # Bottom left corner
            elif y_coord==self.num_vert_y-1 and x_coord==0:
                # Second derivative in the y direction
                self.laplacian_matrix[i][ind_m0] = self.laplacian_matrix[i][ind_m0] + (1.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_y*delta_y))*(1.0-(htc*delta_y)/thermal_conductivity)
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_y)
                
                # Second derivative in the x direction
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_x*delta_x))*(1.0-(htc*delta_x)/thermal_conductivity)
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_0p] = self.laplacian_matrix[i][ind_0p] + (1.0/(delta_x*delta_x))
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_x)
                
            # Bottom right corner
            elif y_coord==self.num_vert_y-1 and x_coord==self.num_vert_x-1:
                # Second derivative in the y direction
                self.laplacian_matrix[i][ind_m0] = self.laplacian_matrix[i][ind_m0] + (1.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_y*delta_y))*(1.0-(htc*delta_y)/thermal_conductivity)
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_y)
                
                # Second derivative in the x direction
                self.laplacian_matrix[i][ind_0m] = self.laplacian_matrix[i][ind_0m] + (1.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_x*delta_x))*(1.0-(htc*delta_x)/thermal_conductivity)
                self.laplacian_vector[i] = self.laplacian_vector[i] + (tamb*htc)/(thermal_conductivity*delta_x)
            
            # Bulk material
            else:
                # Second derivative in the y direction
                self.laplacian_matrix[i][ind_m0] = self.laplacian_matrix[i][ind_m0] + (1.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_y*delta_y))
                self.laplacian_matrix[i][ind_p0] = self.laplacian_matrix[i][ind_p0] + (1.0/(delta_y*delta_y))
                
                # Second derivative in the x direction
                self.laplacian_matrix[i][ind_0m] = self.laplacian_matrix[i][ind_0m] + (1.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_x*delta_x))
                self.laplacian_matrix[i][ind_0p] = self.laplacian_matrix[i][ind_0p] + (1.0/(delta_x*delta_x))
                
            # Second derivative in the z direction
            self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_z*delta_z))*(1.0-(htc*delta_z)/thermal_conductivity)
            self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (-2.0/(delta_z*delta_z))
            self.laplacian_matrix[i][ind_00] = self.laplacian_matrix[i][ind_00] + (1.0/(delta_z*delta_z))*(1.0-(htc*delta_z)/thermal_conductivity)
            self.laplacian_vector[i] = self.laplacian_vector[i] + (2.0*tamb*htc)/(thermal_conductivity*delta_z)
                    
        # Generate state space system for temperature controller
        A = (thermal_conductivity/(density*specific_heat)) * self.laplacian_matrix
        B = (1.0 / (density*specific_heat*delta_z)) * np.eye(len(A))
        
        # Solve LQR control problem for temperature controller
        Q = np.zeros((max(self.num_vert_y,self.num_vert_x), max(self.num_vert_y,self.num_vert_x)))
        Q[0:int(round(max(self.num_vert_y,self.num_vert_x)*0.1)),:]=1
        Q[-int(round(max(self.num_vert_y,self.num_vert_x)*0.1)):,:]=1
        Q[:,0:int(round(max(self.num_vert_y,self.num_vert_x)*0.1))]=1
        Q[:,-int(round(max(self.num_vert_y,self.num_vert_x)*0.1)):]=1
        Q = nd.gaussian_filter(Q, sigma, mode='constant', cval=1.0)
        Q = Q / np.min(Q[:,0])
        Q = inp.zoom(Q, (self.num_vert_y/len(Q), self.num_vert_x/len(Q[0])))
        Q[Q>1.0]=1.0
        Q[Q<0.25]=0.25
        Q=Q/np.mean(Q)
        Q=Q.flatten(order='F')
        Q = 1e8*np.diag(Q)
        R = np.eye(len(A))
        self.K_temperature,_,_ = control.lqr(A,B,Q,R)
        
        # Construct the 3rd order finite difference derivative matrix
        fd_derivative = np.diag((1.0/12.0)*np.ones(self.num_vert_y-2),k=-2)+np.diag((-1.0/12.0)*np.ones(self.num_vert_y-2),k=2)+np.diag((-2.0/3.0)*np.ones(self.num_vert_y-1),k=-1)+np.diag((2.0/3.0)*np.ones(self.num_vert_y-1),k=1)
        fd_derivative[0,:]=0.0
        fd_derivative[0,0]=-1
        fd_derivative[0,1]=1
        fd_derivative[1,:]=0.0
        fd_derivative[1,0]=-0.5
        fd_derivative[1,2]=0.5
        fd_derivative[self.num_vert_y-2,:]=0.0
        fd_derivative[self.num_vert_y-2,self.num_vert_y-3]=-0.5
        fd_derivative[self.num_vert_y-2,self.num_vert_y-1]=0.5
        fd_derivative[self.num_vert_y-1,:]=0.0
        fd_derivative[self.num_vert_y-1,self.num_vert_y-2]=-1
        fd_derivative[self.num_vert_y-1,self.num_vert_y-1]=1
        
        # Generate state space system for shape controller
        A = np.zeros((self.num_vert_y,self.num_vert_y))
        B = np.eye(self.num_vert_y) - (1.0/self.num_vert_y)*np.ones((self.num_vert_y,self.num_vert_y))
        
        # Solve LQR control problem for shape controller
        Q = 1e5*np.eye(self.num_vert_y)
        R = 5e6*np.matmul(np.transpose(fd_derivative), fd_derivative) + np.eye(self.num_vert_y)
        self.K_shape,_,_ = control.lqr(A,B,Q,R)
    
    
    # Gets a target temperature given a degree of cure and target front speed
    # @param The degree of cure
    # @param The target front speed [Meters/Second]
    # @return The target temperature [Kelvin]
    def get_tag_temp(self, alpha, v_tag):
        
        a = 0.000808146-0.00295865*alpha
        b = 0.0202144-0.1646*alpha+0.463609*alpha*alpha
        c = (0.813723-5.12462*alpha+9.54447*alpha*alpha) - 1.0e3*(v_tag)
        t_tag = 273.15 + (-b + np.sqrt(b*b - 4.0*a*c)) / (2.0*a)
        return t_tag
    
    
    # Calculates optimal input 
    # @param 2D array-like object. Temperature grid in [Kelvin]
    # @param 2D array-like object. Target temperature in [Kelvin]
    # @return Optimal input in [Watts / Meter^2]
    def get_input(self, temperature, target):
        
        # Scale the tempeature grid and target temperature (grid) to the proper dimension
        down_scale_temperature = inp.zoom(temperature, (self.num_vert_y/len(temperature), self.num_vert_x/len(temperature[0])))
        try:
            target = inp.zoom(target, (self.num_vert_y/len(target), self.num_vert_x/len(target[0])))
        except:
                pass
        
        # Calculate optimal energy flux
        error = (down_scale_temperature - target)
        error = error.flatten(order='F')
        optimal_input = np.matmul(-self.K, error)
        optimal_input = optimal_input.reshape(self.num_vert_y, self.num_vert_x, order='F')
        
        # Resize the optimal_input back to the original size of the temperature_grid
        optimal_input = inp.zoom(optimal_input, (len(temperature)/self.num_vert_y, len(temperature[0])/self.num_vert_x))
        
        # Return the optimal input
        return optimal_input
    
    
    # Calculates optimal local input given input parameters 
    # @param 2D array-like object. Temperature grid in [Kelvin]
    # @param 2D array-like object. Target temperature in [Kelvin]
    # @param Radius of input in [Meters]
    # @param Power of the input in [Watts]
    # @param The x coordinate of the input in the temperature image reference frame [Meters]
    # @param The y coordinate of the input in the temperature image reference frame [Meters]
    # @param Flag that indicates whether the optimal location will be calcualted. If false, optimal location is returned as (0,0)
    # @return The optimal input magnitude and location
    def get_local_input(self, temperature, target, radius_in, power_in, input_loc_x, input_loc_y, get_loc, get_mask=False):
        
        # Calculate the step sizes of the temperature field given as an input
        step_size_y = self.width / (len(temperature) - 1)
        step_size_x = self.length / (len(temperature[0]) - 1)
        
        ## UPDATE PARAMETERS IF NEEEDED ##
        ## ===================================================================================================================== ##
        # Check to see if input kernal is already generated. If it is not, calculate it
        if step_size_y!=self.temp_step_size_y or step_size_x!=self.temp_step_size_x or radius_in!=self.radius or power_in!=self.power:
            # Assign current step size and radius values
            self.temp_step_size_y = step_size_y
            self.temp_step_size_x = step_size_x
            self.radius = radius_in
            self.power=power_in
            
            # Calculate the peak input power
            self.max_input = self.power / (np.pi * 0.2171472409514 * self.radius * self.radius)
            
            # Calculate the grids
            y_linspace = np.linspace(0.0, self.width, len(temperature))
            x_linspace = np.linspace(0.0, self.length, len(temperature[0]))
            self.grid_x, self.grid_y = np.meshgrid(x_linspace, y_linspace)
            
            # Calculate the input's Gaussian kernal
            input_const = -1.0 / (0.2171472409514 * self.radius * self.radius)
            length_x = int(np.round(2.0*self.radius/self.temp_step_size_x))
            length_y = int(np.round(2.0*self.radius/self.temp_step_size_y))
            left_x = int(-0.5*length_x)
            right_x = int(0.5*length_x)
            left_y = int(-0.5*length_y)
            right_y = int(0.5*length_y)
            self.kernal = []
            curr_y = -1
            for i in np.arange(left_y, right_y+1):
                y_loc = i * self.temp_step_size_y
                curr_y = curr_y + 1
                self.kernal.append([])
                for j in np.arange(left_x, right_x+1):
                    x_loc = j * self.temp_step_size_x
                    self.kernal[curr_y].append(np.exp((x_loc)**2*input_const + (y_loc)**2*input_const))
            self.kernal = np.array(self.kernal)
            self.kernal[self.kernal<0.01]=0.0
        
        ## GET THE OPTIMAL GLOBAL INPUT ##
        ## ===================================================================================================================== ##
        opt_global_input = self.get_input(temperature, target)
        opt_global_input[opt_global_input<0.0]=0.0
        opt_global_input[opt_global_input>self.max_input]=self.max_input
        
        ## GET THE OPTIMAL LOCAL INPUT MAGNITUDE ##
        ## ===================================================================================================================== ##
        # Determine the closest indices of the input
        closest_y_ind = np.argmin(abs(self.grid_y[:,0] - input_loc_y))
        closest_x_ind = np.argmin(abs(self.grid_x[0,:] - input_loc_x))
        local_input_loc_inds = (closest_y_ind, closest_x_ind)
        
        # Calculate the local input global kernal based on current location
        curr_local_input_mask = np.zeros((len(opt_global_input)+2*int((len(self.kernal)-1)/2), len(opt_global_input[0])+2*int((len(self.kernal[0])-1)/2)))
        curr_local_input_mask[local_input_loc_inds[0]:local_input_loc_inds[0]+len(self.kernal), local_input_loc_inds[1]:local_input_loc_inds[1]+len(self.kernal[0])] = self.kernal
        curr_local_input_mask = curr_local_input_mask[int((len(self.kernal)-1)/2):len(curr_local_input_mask)-int((len(self.kernal)-1)/2), int((len(self.kernal[0])-1)/2):len(curr_local_input_mask[0])-int((len(self.kernal[0])-1)/2)]
        
        # Determine optimal local input magnitude
        opt_mag = ((np.trace(np.matmul(np.transpose(curr_local_input_mask), opt_global_input))) / (np.trace(np.matmul(np.transpose(curr_local_input_mask), curr_local_input_mask)))) / self.max_input
        if opt_mag > 1.0:
            opt_mag = 1.0
        elif opt_mag < 0.0:
            opt_mag = 0.0
        
        ## GET THE OPTIMAL LOCAL INPUT POSITION ##
        ## ===================================================================================================================== ##
        if get_loc:
            # Calculate the sqaure distance of each point from the current input location
            dist_from_local_input = ((self.grid_x-input_loc_x)*(self.grid_x-input_loc_x)+(self.grid_y-input_loc_y)*(self.grid_y-input_loc_y)) / (self.length*self.length + self.width*self.width)
            
            # Determine the optimal next position for the input
            convolved_opt_global_input = signal.convolve2d(opt_global_input, self.kernal, mode='same', boundary='symm', fillvalue=0.0)
            convolved_opt_global_input = (convolved_opt_global_input) / (np.sum(self.kernal)*(self.max_input))
            opt_loc = np.unravel_index(np.argmax(self.movement_const * (1-dist_from_local_input) + convolved_opt_global_input), convolved_opt_global_input.shape)
            opt_loc = np.array([self.grid_x[opt_loc], self.grid_y[opt_loc]])
        else:
            opt_loc = np.zeros(2)
            
        # Return the optimal magnitude and location
        ret_list = []
        ret_list.append(opt_mag)
        ret_list.append(opt_loc[0])
        ret_list.append(opt_loc[1])
        if get_mask:
            ret_list.append(opt_mag*self.max_input*curr_local_input_mask)
        return tuple(ret_list)