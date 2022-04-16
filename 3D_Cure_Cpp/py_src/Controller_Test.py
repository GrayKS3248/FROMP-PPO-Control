# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:09:31 2022

@author: GKSch
"""

import Controller
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.ndimage.interpolation as inp
import imageio
import opensimplex

if __name__ == "__main__":

    ## PARAMETERS ##
    ## ===================================================================================================================== ##
    # Material params
    thermal_conductivity = 0.152  ## [Watts / Meter - Kelvin]
    density = 980.0               ## [Kilograms / Meter^3]
    specific_heat = 1600.0        ## [Joules / Kilogram - Kelvin]
    heat_transfer_coeff = 20.0    ## [Watts / Meter^2 - Kelvin]
    ambient_temp = 293.15         ## [Kelvin]
    
    # Domain size
    width = 0.008        ## Y-dimension size [Meters]
    length = 0.00575     ## X-dimension size [Meters]
    thickness = 0.001    ## Z-dimension size [Meters]
    sim_num_vert_y = 33  ## Number of simiulation vertices in domain [-]
    sim_num_vert_x = 24  ## Number of simiulation vertices in domain [-]
    sim_num_vert_z = 5   ## Number of simiulation vertices in domain [-]
    
    # Controller params
    state_size = 800      ## Number temperature points in controller's state [-]
    movement_const = 0.0  ## [Relative importance of minimizing input movement compared to minimizing local input vs global optimal input]
    sigma = 1.0           ## Stdev of gaussian kernal used to weight Q matrix towards edges [-]
    target_temp = 305.80  ## [Kelvin]
    use_image = True      ## Determines if the target temperature uses an image or is flat
    con_dt = 0.16         ## Controller location update period [Seconds]  
    
    # Simulation params
    total_time = 60.0        ## [Seconds]
    final_input_time = 60.0  ## Time at which the input is turned off [Seconds]
    dt = 0.01                ## Time step size [Seconds]
    fr = 30              ## Frame rate [Frames / Second]
    scroll = True            ## Determine if the simulation domain scrolls into new material or remains stationary
    scroll_speed = 0.0015    ## Speed at which domain scrolls [Meters / Second]
    
    # Initial temperature params
    initial_temp = 297.15         ## [Kelvin]
    initial_temp_deviation = 2.0  ## initial_temp_field = initial_temp +- initial_temp_deviation [Kelvin]
    feature_size = 0.50           ## Simplex noise feature size in percent of major dimension [Decimal Percent]
    seed = 1000                   ## Simplex noise seed
    
    # Input params
    local = False          ## Determines if input is local or global
    radius = 0.003        ## [Meters]
    power = 0.5           ## [Watts]
    speed = 0.025         ## [Meters / Second]
    
    
    ## PRE-SIMULATION CALCULATIONS ##
    ## ===================================================================================================================== ##
    # Calculate the target temperature field
    if use_image:
        image = np.flip(imageio.imread('Half_Grad.jpg')[:,:,0],axis=0)
        image = inp.zoom(image, (sim_num_vert_y/len(image), sim_num_vert_x/len(image[0])))/255.0
        target_temp_1 = (target_temp - initial_temp) * image + initial_temp
        
        image = np.flip(imageio.imread('Half_Grad.jpg')[:,:,0],axis=0)
        image = inp.zoom(image, (sim_num_vert_y/len(image), sim_num_vert_x/len(image[0])))/255.0
        target_temp_2 = (target_temp - initial_temp) * image + initial_temp
        
        image = np.flip(imageio.imread('Half_Grad.jpg')[:,:,0],axis=0)
        image = inp.zoom(image, (sim_num_vert_y/len(image), sim_num_vert_x/len(image[0])))/255.0
        target_temp_3 = (target_temp - initial_temp) * image + initial_temp
        
        image = np.flip(imageio.imread('Half_Grad.jpg')[:,:,0],axis=0)
        image = inp.zoom(image, (sim_num_vert_y/len(image), sim_num_vert_x/len(image[0])))/255.0
        target_temp_4 = (target_temp - initial_temp) * image + initial_temp
    else:
        target_temp = target_temp * np.ones((sim_num_vert_y, sim_num_vert_x))   
        
    # Calculate step size of temperature field and downsampled temperature field and 
    sim_step_size_y = width / (sim_num_vert_y-1)
    sim_step_size_x = length / (sim_num_vert_x-1)
    sim_step_size_z = thickness / (sim_num_vert_z-1)
    
    # Calculate spatial grid used for plotting
    y_linspace = np.linspace(0.0, width, sim_num_vert_y)
    x_linspace = np.linspace(0.0, length, sim_num_vert_x)
    sim_grid_x, sim_grid_y = np.meshgrid(x_linspace, y_linspace)
    
    # Initialize and randomize intial temperature field so that the mean is initial_temp and the maximum absolution deviation is initial_temp_deviation
    major_dim_step_size = [sim_step_size_y, sim_step_size_x, sim_step_size_z][np.argmax([width, length, thickness])]
    y_feature_size = (sim_num_vert_y * feature_size) * (major_dim_step_size / sim_step_size_y)
    x_feature_size = (sim_num_vert_x * feature_size) * (major_dim_step_size / sim_step_size_x)
    z_feature_size = (sim_num_vert_z * feature_size) * (major_dim_step_size / sim_step_size_z)
    if scroll:
        ext_num_vert_x = int(np.ceil((((length + total_time*scroll_speed)*(sim_num_vert_x-1)) / length) + 1))
        scroll_dt = total_time / (ext_num_vert_x - sim_num_vert_x)
        curr_ext_ind = 0
        initial_temp_field = np.zeros((sim_num_vert_y, ext_num_vert_x, sim_num_vert_z))
        max_cum_temp_field = np.zeros((sim_num_vert_y, ext_num_vert_x))
        max_cum_target_temp_field = np.zeros((sim_num_vert_y, ext_num_vert_x))
        opensimplex.seed(seed)
        for i in range(0, sim_num_vert_y):
            for j in range(0, ext_num_vert_x):
                for k in range(0, sim_num_vert_z):
                    initial_temp_field[i,j,k] = opensimplex.noise3(i/y_feature_size, j/x_feature_size, k/z_feature_size)
    else:
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
    
    
    ## GET LAPLACIAN FUNCTION ##
    ## ===================================================================================================================== ##
    # The 9 point laplacian kernal based on nonuniform grid size
    l_kernal = np.zeros((3,3,3))
    l_kernal[1,1,1] = -2.0 * (1.0/(sim_step_size_y*sim_step_size_y) + 1.0/(sim_step_size_x*sim_step_size_x) + 1.0/(sim_step_size_z*sim_step_size_z))
    l_kernal[0,1,1] = 1.0/(sim_step_size_y*sim_step_size_y)
    l_kernal[2,1,1] = 1.0/(sim_step_size_y*sim_step_size_y)
    l_kernal[1,0,1] = 1.0/(sim_step_size_x*sim_step_size_x)
    l_kernal[1,2,1] = 1.0/(sim_step_size_x*sim_step_size_x)
    l_kernal[1,1,0] = 1.0/(sim_step_size_z*sim_step_size_z)
    l_kernal[1,1,2] = 1.0/(sim_step_size_z*sim_step_size_z)
        
    # Calculates the laplacian of a 3-dimensional field with Neumann boundary conditions
    def get_laplacian(temp, heat, htc, amb, kappa, step_y, step_x, step_z):
        
        # Determine number of vertices in temperature field
        y_vert = len(temp)
        x_vert = len(temp[0])
        z_vert = len(temp[0][0])
                
        # Append the virtual boundary temperatures to the original temperature field
        padded_temp = np.zeros((y_vert+2, x_vert+2, z_vert+2))
        padded_temp[1:y_vert+1,1:x_vert+1,1:z_vert+1] = temp
        padded_temp[1:y_vert+1,0,1:z_vert+1] = -4.0*((step_x*htc/kappa)*(temp[:,0,:]-amb) + (5.0/6.0)*temp[:,0,:] + (-3.0/2.0)*temp[:,1,:] + (1.0/2.0)*temp[:,2,:] + (-1.0/12.0)*temp[:,3,:])
        padded_temp[1:y_vert+1,x_vert+1,1:z_vert+1] = -4.0*((step_x*htc/kappa)*(temp[:,x_vert-1,:]-amb) + (5.0/6.0)*temp[:,x_vert-1,:] + (-3.0/2.0)*temp[:,x_vert-2,:] + (1.0/2.0)*temp[:,x_vert-3,:] + (-1.0/12.0)*temp[:,x_vert-4,:])
        padded_temp[0,1:x_vert+1,1:z_vert+1] = -4.0*((step_y*htc/kappa)*(temp[0,:,:]-amb) + (5.0/6.0)*temp[0,:,:] + (-3.0/2.0)*temp[1,:,:] + (1.0/2.0)*temp[2,:,:] + (-1.0/12.0)*temp[3,:,:])
        padded_temp[y_vert+1,1:x_vert+1,1:z_vert+1] = -4.0*((step_y*htc/kappa)*(temp[y_vert-1,:,:]-amb) + (5.0/6.0)*temp[y_vert-1,:,:] + (-3.0/2.0)*temp[y_vert-2,:,:] + (1.0/2.0)*temp[y_vert-3,:,:] + (-1.0/12.0)*temp[y_vert-4,:,:])
        padded_temp[1:y_vert+1,1:x_vert+1,0] = temp[:,:,0] - (step_z/kappa)*(htc*(temp[:,:,0]-amb)-heat)
        padded_temp[1:y_vert+1,1:x_vert+1,z_vert+1] = temp[:,:,z_vert-1] - (step_z*htc/kappa)*(temp[:,:,z_vert-1]-amb)
        
        # Get the 9 point stencil laplacian
        laplacian = signal.convolve(padded_temp, l_kernal, mode='valid')
        return laplacian
             
    
    ## RUN THE SIMULATION ##
    ## ===================================================================================================================== ##
    # Initialize controller
    controller = Controller.Controller(thermal_conductivity,density,specific_heat,width,length,thickness,state_size,movement_const,sigma)
    
    # Initialize the time and an iterator
    t = 0.0
    itr = 0
    
    # Initial the input masks
    if not local:
        opt_global_input_mask = np.zeros((sim_num_vert_y, sim_num_vert_x))
    else:
        opt_local_input_mask = np.zeros((sim_num_vert_y, sim_num_vert_x))
        local_input_loc = np.array([0.0, 0.0])
        
    # Update the inputs and temperature fields
    while t <= total_time:

        # Update the target temperature
        if use_image:
            if t >=0.0 and t < total_time/4:
                target_temp = target_temp_1
            elif t >= total_time/4 and t < 2*total_time/4:
                target_temp = target_temp_2
            elif t >= 2*total_time/4 and t < 3*total_time/4:
                target_temp = target_temp_3
            else:
                target_temp = target_temp_4

        # Update the input only if the input is still turned on
        if t <= final_input_time:
        
            # Take an image of the temperature field
            if scroll:
                temperature_image = temperature[:,curr_ext_ind:curr_ext_ind+sim_num_vert_x,:][:,:,0]
            else:
                temperature_image = temperature[:,:,0]
                
            # Get the global optimal input
            if not local:
                opt_global_input_mask = controller.get_input(temperature_image, target_temp)
                opt_global_input_mask[opt_global_input_mask<0.0]=0.0
                opt_global_input_mask[opt_global_input_mask>(power/(length*width))]=(power/(length*width))
                
            # Get the local optimal input magnitude and position
            else:
                opt_local_input = controller.get_local_input(temperature_image, target_temp, radius, power, local_input_loc[0], local_input_loc[1], get_loc=(t % con_dt < dt), get_mask=True)
                
                # Determine the optimal direction of travel for the input and update its location
                if (t % con_dt < dt):
                    opt_local_input_loc =  np.array([opt_local_input[1], opt_local_input[2]])
                opt_dirn =  opt_local_input_loc - local_input_loc
                dist = np.linalg.norm(opt_dirn)
                
                # Update the input location
                if dist != 0.0:
                    if dist < speed*dt:
                        local_input_loc = opt_local_input_loc
                    else:
                        local_input_loc = local_input_loc + speed * dt * (opt_dirn / dist)
                
                # Update the input's mask
                opt_local_input_mask = np.array(opt_local_input[-1])
           
        # If the input is off, send 0 input command
        else:
            if not local:
                opt_global_input_mask = np.zeros((sim_num_vert_y, sim_num_vert_x))
            else: 
                opt_local_input_mask = np.zeros((sim_num_vert_y, sim_num_vert_x))   
        
        # Extend the input mask if scroll option is enabled
        if scroll and not local:
            temporary = np.zeros((sim_num_vert_y, ext_num_vert_x))   
            temporary[:,curr_ext_ind:curr_ext_ind+sim_num_vert_x] = opt_global_input_mask
            opt_global_input_mask = temporary
        elif scroll and local: 
            temporary = np.zeros((sim_num_vert_y, ext_num_vert_x))   
            temporary[:,curr_ext_ind:curr_ext_ind+sim_num_vert_x] = opt_local_input_mask
            opt_local_input_mask = temporary
        
        # Step the temperature field
        if not local:
            laplacian = get_laplacian(temperature, opt_global_input_mask, heat_transfer_coeff, ambient_temp, thermal_conductivity, sim_step_size_y, sim_step_size_x, sim_step_size_z)
        else:
            laplacian = get_laplacian(temperature, opt_local_input_mask, heat_transfer_coeff, ambient_temp, thermal_conductivity, sim_step_size_y, sim_step_size_x, sim_step_size_z)
        temperature = temperature + ((thermal_conductivity/(density*specific_heat))*laplacian) * dt
        
        # Update the maximum cumulative temperature field
        if scroll:
            temperature_image = temperature[:,:,0]
            max_cum_temp_field[temperature_image > max_cum_temp_field] = temperature_image[temperature_image > max_cum_temp_field]
            temporary = np.zeros((sim_num_vert_y, ext_num_vert_x))
            temporary[:,curr_ext_ind:curr_ext_ind+sim_num_vert_x] = target_temp
            max_cum_target_temp_field[temporary > max_cum_target_temp_field] = temporary[temporary > max_cum_target_temp_field]
        
        # Make fig for temperature and input
        if (fr!=0.0) and (t % (1.0/fr) < dt):
            plt.cla()
            plt.clf()
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True)
            fig.set_size_inches(12,5)
           
            # Plot temperature
            if scroll:
                temperature_image = temperature[:,curr_ext_ind:curr_ext_ind+sim_num_vert_x,0]
            else:
                temperature_image = temperature[:,:,0]
            c0 = ax0.pcolormesh(1000.0*sim_grid_x, 1000.0*sim_grid_y, temperature_image-273.15, shading='gouraud', cmap='jet', vmin=(initial_temp-initial_temp_deviation)-273.15, vmax=(np.max(target_temp) + 0.1*(np.max(target_temp) - initial_temp))-273.15)
            cbar0 = fig.colorbar(c0, ax=ax0,shrink=0.75,fraction=0.3)
            cbar0.set_label("Temperature [C]",labelpad=8,fontsize=14)
            cbar0.ax.tick_params(labelsize=14)
            ax0.set_xlabel('X Position [mm]',fontsize=16,labelpad=12)
            ax0.set_ylabel('Y Position [mm]',fontsize=16,labelpad=12)
            ax0.tick_params(axis='x',labelsize=16, length=6, width=1.25)
            ax0.tick_params(axis='y',labelsize=16, length=6, width=1.25)
            ax0.set_title("$μ_{err}=$"+str(round(np.mean(temperature_image[:,:]-target_temp),1)) + " °C\n" +"$σ_{err}=$" + str(round(np.std(temperature_image[:,:]-target_temp),1)) + " °C",fontsize=16)
            ax0.set_aspect('equal', adjustable='box', anchor='W')
            
            # Plot target
            c1 = ax1.pcolormesh(1000.0*sim_grid_x, 1000.0*sim_grid_y, target_temp-273.15, shading='gouraud', cmap='jet', vmin=(initial_temp-initial_temp_deviation)-273.15, vmax=(np.max(target_temp) + 0.1*(np.max(target_temp) - initial_temp))-273.15)
            cbar1 = fig.colorbar(c1, ax=ax1,shrink=0.75,fraction=0.3)
            cbar1.set_label("Temperature [C]",labelpad=8,fontsize=14)
            cbar1.ax.tick_params(labelsize=14)
            ax1.set_xlabel('X Position [mm]',fontsize=16,labelpad=12)
            ax1.tick_params(axis='x',labelsize=16, length=6, width=1.25)
            ax1.tick_params(axis='y',labelsize=16, length=6, width=1.25)
            ax1.set_title("Target",fontsize=16)
            ax1.set_aspect('equal', adjustable='box', anchor='W')
            
            # Plot input
            if not local:
                plot_input = 1e-3*opt_global_input_mask
                max_input = power / (length*width)
            else:
                plot_input = 1e-3*opt_local_input_mask
                max_input = power / (np.pi * 0.2171472409514 * radius * radius)
            if scroll:
                plot_input = plot_input[:,curr_ext_ind:curr_ext_ind+sim_num_vert_x]
            c2 = ax2.pcolormesh(1000.0*sim_grid_x, 1000.0*sim_grid_y, plot_input, shading='gouraud', cmap='coolwarm', vmin=0.0, vmax=1e-3*max_input)
            cbar2 = fig.colorbar(c2, ax=ax2,shrink=0.75,fraction=0.3)
            cbar2.set_label("Input [kW / m^2]",labelpad=8,fontsize=14)
            cbar2.ax.tick_params(labelsize=14)
            ax2.set_xlabel('X Position [mm]',fontsize=16,labelpad=12)
            ax2.tick_params(axis='x',labelsize=16, length=6, width=1.25)
            ax2.tick_params(axis='y',labelsize=16, length=6, width=1.25)
            ax2.set_title("Input",fontsize=16)
            ax2.set_aspect('equal', adjustable='box', anchor='E')
            
            # Save and close figure
            title_str = "Time From Trigger: "+'{:.2f}'.format(t)+'s'
            fig.suptitle(title_str,fontsize='xx-large', fontname = 'monospace')
            plt.savefig('../results/' + str(itr).zfill(4)+'.png', dpi=100)
            plt.close()
            itr = itr + 1
        
        # Step time
        t = t + dt
        
        # Scroll grid
        if scroll and (t % (scroll_dt) < dt):
            curr_ext_ind = curr_ext_ind + 1
    
    # Plot maximum cumulative temperature field
    if scroll:
        y_linspace = np.linspace(0.0, width, sim_num_vert_y)
        x_linspace = np.linspace(0.0, (length/(sim_num_vert_x-1))*(ext_num_vert_x-1), ext_num_vert_x)
        ext_grid_x, ext_grid_y = np.meshgrid(x_linspace, y_linspace)
        
        plt.cla()
        plt.clf()
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
        fig.set_size_inches(8,((2/3)*(total_time+20))/(((length/(sim_num_vert_x-1))*(ext_num_vert_x-1))/width))
        
        c0 = ax0.pcolormesh(1000.0*ext_grid_x, 1000.0*ext_grid_y, max_cum_temp_field-273.15, shading='gouraud', cmap='jet', vmin=(initial_temp-initial_temp_deviation)-273.15, vmax=(np.max(target_temp) + 0.1*(np.max(target_temp) - initial_temp))-273.15)
        cbar0 = fig.colorbar(c0, ax=ax0,shrink=0.65,fraction=0.1)
        cbar0.set_label("Temperature [C]",labelpad=8,fontsize=14)
        cbar0.ax.tick_params(labelsize=14)
        ax0.set_ylabel('Y Position [mm]',fontsize=16,labelpad=14)
        ax0.tick_params(axis='x',labelsize=16, length=6, width=1.25)
        ax0.tick_params(axis='y',labelsize=16, length=6, width=1.25)
        ax0.set_aspect('equal', adjustable='box', anchor='W')
        ax0.set_title("Actual",fontsize=16)
        
        c1 = ax1.pcolormesh(1000.0*ext_grid_x, 1000.0*ext_grid_y, max_cum_target_temp_field-273.15, shading='gouraud', cmap='jet', vmin=(initial_temp-initial_temp_deviation)-273.15, vmax=(np.max(target_temp) + 0.1*(np.max(target_temp) - initial_temp))-273.15)
        cbar1 = fig.colorbar(c1, ax=ax1,shrink=0.65,fraction=0.1)
        cbar1.set_label("Temperature [C]",labelpad=8,fontsize=14)
        cbar1.ax.tick_params(labelsize=14)
        ax1.set_xlabel('X Position [mm]',fontsize=16,labelpad=14)
        ax1.set_ylabel('Y Position [mm]',fontsize=16,labelpad=14)
        ax1.tick_params(axis='x',labelsize=16, length=6, width=1.25)
        ax1.tick_params(axis='y',labelsize=16, length=6, width=1.25)
        ax1.set_aspect('equal', adjustable='box', anchor='W')
        ax1.set_title("Target",fontsize=16)

        fig.suptitle("Maximum Cumulative Temperature °C",fontsize=16)
        plt.savefig('../results/max_cum_temp.png', dpi=1000)
        plt.close()
        
        print(str(np.mean(abs(max_cum_target_temp_field-max_cum_temp_field))))