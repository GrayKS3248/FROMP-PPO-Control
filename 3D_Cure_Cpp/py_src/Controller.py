# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:36:17 2022

@author: GKSch
"""
import numpy as np
import control

class Controller:
    
    # Initializes the linear optimal controller by solving the LQR minimization problem for the heat equation
    # @param Thermal conductivity of the material in [Watts / Meter * Kelvin]
    # @param Density of the material in [Kilograms / Meter ^ 3]
    # @param Specific heat capacity of the material in [Joules / Kilogram * Kelvin]
    # @param Number of nodes in the transverse direction of the temperature grid
    # @param Number of nodes in the longitudinal direction of the temperature grid
    # @param Step size of the transverse temperature grid in [Meter]
    # @param Step size of the longitudinal temperature grid in [Meter]
    # @param Thickness of the material in [Meter]
    def __init__(self, thermal_conductivity, density, specific_heat, num_vert_y, num_vert_x, step_size_y, step_size_x, thickness):
        
        # Define size of temperature grid
        self.num_vert_y=num_vert_y
        self.num_vert_x=num_vert_x
        
        # Define laplacial stencil to be used
        laplacian_stencils = np.array([[35/12, -26/3, 19/2, -14/3, 11/12],
                          [11/12, -5/3, 1/2, 1/3, -1/12],
                          [-1/12, 4/3, -5/2, 4/3, -1/12],
                          [-1/12, 1/3, 1/2, -5/3, 11/12],
                          [11/12, -14/3, 19/2, -26/3, 35/12]])
        
        # Calculate the laplacian stencil indicies
        left_ind = (len(laplacian_stencils[0])//2 - len(laplacian_stencils[0]) + 1)
        right_ind = (len(laplacian_stencils[0]) - len(laplacian_stencils[0])//2 - 1)
        stencil_inds = []
        for row in range(len(laplacian_stencils)):
            stencil_inds.append(np.arange(left_ind+(right_ind-row), right_ind+1+(right_ind-row)))
        stencil_inds = np.array(stencil_inds)
        
        # Generate the laplcian matrix
        self.laplacian_matrix = np.zeros((num_vert_y*num_vert_x,num_vert_y*num_vert_x))
        
        # Populate the laplacian matrix
        for j in np.arange(num_vert_x):
            
            # Determine which stencil to used based on j distance from boundary
            j_boundary_code = 2
            while any((j+stencil_inds[j_boundary_code])>num_vert_x-1) or any((j+stencil_inds[j_boundary_code])<0):
                if any((j+stencil_inds[j_boundary_code])>num_vert_x-1):
                    j_boundary_code = j_boundary_code + 1
                elif any((j+stencil_inds[j_boundary_code])<0):
                    j_boundary_code = j_boundary_code - 1
                    
            for i in np.arange(num_vert_y):
                
                # Determine which stencil to used based on j distance from boundary
                i_boundary_code = 2
                while any((i+stencil_inds[i_boundary_code])>num_vert_y-1) or any((i+stencil_inds[i_boundary_code])<0):
                    if any((i+stencil_inds[i_boundary_code])>num_vert_y-1):
                        i_boundary_code = i_boundary_code + 1
                    elif any((i+stencil_inds[i_boundary_code])<0):
                        i_boundary_code = i_boundary_code - 1
                    
                # Y direction second derivative
                for p in stencil_inds[i_boundary_code]:
                    self.laplacian_matrix[i+num_vert_y*j][(i+p) + num_vert_y*j] = self.laplacian_matrix[i+num_vert_y*j][(i+p) + num_vert_y*j] + laplacian_stencils[i_boundary_code][np.argmin(abs(stencil_inds[i_boundary_code] - p))]/(step_size_y*step_size_y)
                    
                # X direction second derivative
                for p in stencil_inds[j_boundary_code]:
                    self.laplacian_matrix[i+num_vert_y*j][i + num_vert_y*(j+p)] = self.laplacian_matrix[i+num_vert_y*j][i + num_vert_y*(j+p)] + laplacian_stencils[j_boundary_code][np.argmin(abs(stencil_inds[j_boundary_code] - p))]/(step_size_x*step_size_x)
                    
        # Generate state space system
        A = (thermal_conductivity/(density*specific_heat)) * self.laplacian_matrix
        B = (2.0 / (density*specific_heat*thickness)) * np.eye(len(A))
        
        # Solve LQR control problem
        Q = np.eye(len(A))
        R = 0.0000004*np.eye(len(A))
        self.K,_,_ = control.lqr(A,B,Q,R)
        
    # Calculates optimal input 
    # @param num_vert_y X num_vert_x array-like object. Temperature grid in [Kelvin]
    # @param Uniform target temperature in [Kelvin]
    # @return Optimal input in [Watts / Meter^2]
    def get_input(self, temperature_grid, target_temperature):
        
        # Check inputs
        if (len(temperature_grid) != self.num_vert_y):
            raise Exception("Temperature grid must be num_vert_y X num_vert_x array-like object.")
        if (len(temperature_grid[0]) != self.num_vert_x):
            raise Exception("Temperature grid must be num_vert_y X num_vert_x array-like object.")
        if (target_temperature < 0.0):
            raise Exception("Target temperature must be greater than 0 K.")
        
        # Calculate optimal energy flux
        error = (temperature_grid - target_temperature)
        error = error.flatten(order='F')
        optimal_input = np.matmul(-self.K, error)
        optimal_input = optimal_input.reshape(self.num_vert_y, self.num_vert_x, order='F')
        
        # Return the optimal input
        return optimal_input
            
            