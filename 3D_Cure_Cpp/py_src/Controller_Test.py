# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:09:31 2022

@author: GKSch
"""

import Controller
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import opensimplex

if __name__ == "__main__":

    ## PARAMETERS ##
    ## ===================================================================================================================== ##
    # Material params
    thermal_conductivity = 0.152  ## [Watts / Meter - Kelvin]
    density = 980.0               ## [Kilograms / Meter^3]
    specific_heat = 1600.0        ## [Joules / Kilogram - Kelvin]
    
    # Domain size
    width = 0.135        ## Y-dimension size [Meters]
    length = 0.135       ## X-dimension size [Meters]
    thickness = 0.001    ## Z-dimension size [Meters]
    sim_num_vert_y = 64  ## [-]
    sim_num_vert_x = 64  ## [-]
    sim_num_vert_z = 8   ## [-]
    
    # Controller params
    num_vert_y = 20      ## Number of vertices in Y-dimension of downsampled temperature field sent to controller [-]
    num_vert_x = 20      ## Number of vertices in X-dimension of downsampled temperature field sent to controller [-]
    target_temp = 300.15  ## [Kelvin]
    
    # Simulation params
    total_time = 20.0        ## [Seconds]
    final_input_time = 20.0  ## Time at which the input is turned off [Seconds]
    dt = 0.01                ## Time step size [Seconds]
    
    # Initial temperature params
    initial_temp = 290.15         ## [Kelvin]
    initial_temp_deviation = 1.0  ## initial_temp_field = initial_temp +- initial_temp_deviation [Kelvin]
    feature_size = 0.50           ## Simplex noise feature size in percent of major dimension [Decimal Percent]
    seed = 1000                   ## Simplex noise seed
    
    # Input params
    radius = 0.03        ## [Meters]
    min_input = 0.0      ## [Watts / Meter^2]
    max_input = 20000.0  ## [Watts / Meter^2]
    
    
    ## PRE-SIMULATION CALCULATIONS ##
    ## ===================================================================================================================== ##
    # Calculate step size of temperature field and downsampled temperature field and 
    sim_step_size_y = width / (sim_num_vert_y-1)
    sim_step_size_x = length / (sim_num_vert_x-1)
    sim_step_size_z = thickness / (sim_num_vert_z-1)
    step_size_y = width / (num_vert_y-1)
    step_size_x = length / (num_vert_x-1)
    
    # Calculate spatial grid used for plotting
    y_linspace = np.linspace(0.0, width, sim_num_vert_y)
    x_linspace = np.linspace(0.0, length, sim_num_vert_x)
    grid_y, grid_x = np.meshgrid(y_linspace, x_linspace)
    
    # Initialize and randomize intial temperature field so that the mean is initial_temp and the maximum absolution deviation is initial_temp_deviation
    major_dim_step_size = [sim_step_size_y, sim_step_size_x, sim_step_size_z][np.argmax([width, length, thickness])]
    y_feature_size = (sim_num_vert_y * feature_size) * (major_dim_step_size / sim_step_size_y)
    x_feature_size = (sim_num_vert_x * feature_size) * (major_dim_step_size / sim_step_size_x)
    z_feature_size = (sim_num_vert_z * feature_size) * (major_dim_step_size / sim_step_size_z)
    initial_temp_field = np.zeros((sim_num_vert_y, sim_num_vert_x, sim_num_vert_z))
    opensimplex.seed(seed)
    for i in range(0, sim_num_vert_y):
        for j in range(0, sim_num_vert_x):
            for k in range(0, sim_num_vert_z):
                initial_temp_field[i,j,k] = opensimplex.noise3(i/y_feature_size, j/x_feature_size, k/z_feature_size)
    initial_temp_field = initial_temp_field - np.mean(initial_temp_field)
    initial_temp_field = initial_temp_field * (1.0/np.max(abs(initial_temp_field)))
    initial_temp_field = initial_temp_field * initial_temp_deviation + initial_temp
    temperature = initial_temp_field
    
    # Calculate the input's Gaussian kernal
    input_const = -1.0 / (0.2171472409514 * radius * radius)
    length_x = int(np.round(radius/step_size_x))
    length_y = int(np.round(radius/step_size_y))
    left_x = int(-0.5*length_x)
    right_x = int(0.5*length_x)
    left_y = int(-0.5*length_y)
    right_y = int(0.5*length_y)
    kernal = []
    curr_y = -1
    for i in np.arange(left_y, right_y+1):
        y_loc = i * step_size_y
        curr_y = curr_y + 1
        kernal.append([])
        for j in np.arange(left_x, right_x+1):
            x_loc = j * step_size_x
            kernal[curr_y].append(np.exp((x_loc)**2*input_const + (y_loc)**2*input_const))
    kernal = np.array(kernal)
    
    
    ## GET LAPLACIAN FUNCTION ##
    ## ===================================================================================================================== ##
    # Laplacian calculation consts for 6th order 7-stencil
    laplacian_stencil = np.array([[ 137.0/180.0, -49.0/60.0, -17.0/12.0,  47.0/18.0, -19.0/12.0,  31.0/60.0, -13.0/180.0 ], 
			                      [ -13.0/180.0,  19.0/15.0,   -7.0/3.0,   10.0/9.0,   1.0/12.0,  -1.0/15.0,    1.0/90.0 ], 
			                      [    1.0/90.0,  -3.0/20.0,    3.0/2.0, -49.0/18.0,    3.0/2.0,  -3.0/20.0,    1.0/90.0 ],
			                      [    1.0/90.0,  -1.0/15.0,   1.0/12.0,   10.0/9.0,   -7.0/3.0,  19.0/15.0, -13.0/180.0 ], 
			                      [ -13.0/180.0,  31.0/60.0, -19.0/12.0,  47.0/18.0, -17.0/12.0, -49.0/60.0, 137.0/180.0 ]])
    
    def get_laplacian(temperature):
        left_bc_temps[j][k] = np.zeros((len(temperature), len(tempearture[0][0])))
        right_bc_temps[j][k] = np.zeros((len(temperature), len(tempearture[0][0])))
        for i in range(len(temperature)):
            for k in range(len(tempearture[0][0])):
                left_bc_temps[j][k] = -4.0*((fine_x_step*htc/thermal_conductivity)*(fine_temp_mesh[get_ind(0)][j][k]-amb_temp) + (5.0/6.0)*fine_temp_mesh[get_ind(0)][j][k] + (-3.0/2.0)*fine_temp_mesh[get_ind(1)][j][k] + (1.0/2.0)*fine_temp_mesh[get_ind(2)][j][k] + (-1.0/12.0)*fine_temp_mesh[get_ind(3)][j][k]);
                right_bc_temps[j][k] = -4.0*((fine_x_step*htc/thermal_conductivity)*(fine_temp_mesh[get_ind(num_fine_vert_x-1)][j][k]-amb_temp) + (5.0/6.0)*fine_temp_mesh[get_ind(num_fine_vert_x-1)][j][k] + (-3.0/2.0)*fine_temp_mesh[get_ind(num_fine_vert_x-2)][j][k] + (1.0/2.0)*fine_temp_mesh[get_ind(num_fine_vert_x-3)][j][k] + (-1.0/12.0)*fine_temp_mesh[get_ind(num_fine_vert_x-4)][j][k]);
                
                
    ## RUN THE SIMULATION ##
    ## ===================================================================================================================== ##
    # Initialize controller
    controller = Controller.Controller(thermal_conductivity,density,specific_heat,num_vert_y,num_vert_x,step_size_y,step_size_x,thickness)
     
    for i in range(num_steps):
        # Step time
        t = dt*i
        
        if i < final_input_step:
            # Calculate and clip optimal input
            opt_in = controller.get_input(temperature, target_temp)
            opt_in[opt_in < min_input] = min_input
            opt_in[opt_in > max_input] = max_input
            
            # Determine location of real input (real input is localized, optimal input is global)
            opt_in_loc = np.unravel_index(np.argmax(signal.convolve2d(opt_in, kernal, mode='same', boundary='fill', fillvalue=0.0)), opt_in.shape)
            real_in = np.zeros((len(opt_in)+2*int((len(kernal)-1)/2), len(opt_in[0])+2*int((len(kernal[0])-1)/2)))
            real_in[opt_in_loc[0]:opt_in_loc[0]+len(kernal), opt_in_loc[1]:opt_in_loc[1]+len(kernal[0])] = kernal
            real_in = real_in[int((len(kernal)-1)/2):len(real_in)-int((len(kernal)-1)/2), int((len(kernal[0])-1)/2):len(real_in[0])-int((len(kernal[0])-1)/2)]
            
            # Determine optimal real input magnitude
            opt_in_mag = ((np.trace(np.matmul(np.transpose(real_in), opt_in))) / (np.trace(np.matmul(np.transpose(real_in), real_in)))) / max_input
            if opt_in_mag < 0.0:
                opt_in_mag = 0.0
            elif opt_in_mag > 1.0:
                opt_in_mag = 1.0
            real_in = opt_in_mag * max_input * real_in
        else:
            real_in = 0.0*mesh_x
        
        # Step the temperature field
        
        
        
        laplaician = 

        temperature_rate = (thermal_conductivity/(density*specific_heat))*laplaician + (2.0/(density*specific_heat*thickness))* real_in
        temperature = temperature + temperature_rate * dt
        
        # Make fig for temperature and input
        plt.cla()
        plt.clf()
        fig, (ax0, ax1) = plt.subplots(2, 1)
       
        # Plot temperature
        c0 = ax0.pcolormesh(1000.0*mesh_x, 1000.0*mesh_y, -100.0*(temperature - target_temp)/(initial_temp_field-target_temp), shading='nearest', cmap='jet', vmin=-100, vmax=10)
        cbar0 = fig.colorbar(c0, ax=ax0)
        cbar0.set_label("Temperature Error [%]",labelpad=10,fontsize='large')
        cbar0.ax.tick_params(labelsize=12)
        ax0.set_xlabel('X Position [mm]',fontsize='large')
        ax0.set_ylabel('Y Position [mm]',fontsize='large')
        ax0.tick_params(axis='x',labelsize=12)
        ax0.tick_params(axis='y',labelsize=12)
        ax0.set_aspect('equal', adjustable='box')
        
        # Plot input
        c1 = ax1.pcolormesh(1000.0*mesh_x, 1000.0*mesh_y, 1e-3*real_in, shading='nearest', cmap='jet', vmin=1e-3*min_input, vmax=1e-3*max_input)
        cbar1 = fig.colorbar(c1, ax=ax1)
        cbar1.set_label("Input [kW / m^2]",labelpad=10,fontsize='large')
        cbar1.ax.tick_params(labelsize=12)
        ax1.set_xlabel('X Position [mm]',fontsize='large')
        ax1.set_ylabel('Y Position [mm]',fontsize='large')
        ax1.tick_params(axis='x',labelsize=12)
        ax1.tick_params(axis='y',labelsize=12)
        ax1.set_aspect('equal', adjustable='box')
          
        # Save and close figure
        title_str = "Time From Trigger: "+'{:.2f}'.format(t)+'s'
        fig.suptitle(title_str,fontsize='xx-large', fontname = 'monospace')
        plt.gcf().set_size_inches(5.5, 9.0)
        plt.savefig('../results/' + str(i).zfill(4)+'.png', dpi=100)
        plt.close()
        
    error = 100.0*(temperature - target_temp)/(initial_temp_field-target_temp)