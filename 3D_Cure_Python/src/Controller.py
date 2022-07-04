# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:36:17 2022

@author: Grayson Schaer
"""
import numpy as np
import control
from scipy import ndimage
from scipy import signal

#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#

class Con:
    def __init__(self, env, state_size=128, boundary_temp_bias=0., QR_ratio=1.0e7, radius=0.004, power=0.25, slew_speed=0.025):
        """
        Initializes a temperature controller that either applies a heat flux over an entire field or a local, Gaussian heat flux over a portion of a field.
        If applying local fluxes, the controller also controls local input location.

        Parameters
        ----------
        env : Environment.Env
            The simulation environment in which the controller acts.
        state_size : int, optional
            Total number of nodes in the temperature grid used as state. The default is 128.
        boundary_temp_bias : float [0.0,1.0], optional
            Q matrix bias of near boundary temperatures to near center temperatures. 0 is no bias, 1 is max bias. The default is 0..
        QR_ratio : float > 0.0, optional
            The ratio of mean(Q)/mean(R) used in determination of LQR control problem. The default is 1.0e7.
        radius : float > 0.0, optional
            The radius of the heat source in meters. The default is 0.004.
        power : float > 0.0, optional
            The total power of the heat source in watts. The default is 0.25.
        slew_speed : float > 0.0, optional
            The maximum slew speed of the heat input in m/s. The default is 0.025.

        Returns
        -------
        None.

        """
        
        # Define thermal properties
        self.kappa = env.kappa    ## Watts / Meter * Kelvin
        self.cp = env.cp          ## Joules / Kilogram * Kelvin
        self.rho = env.rho        ## Kilograms / Meter ^ 3
        self.htc = env.htc        ## Joules / Meter^2 * Kelvin
        self.amb_T = env.amb_T    ## Kelvin
        
        # Define input parameters used by local controller
        self.dy = None
        self.dx = None
        self.peak_flux = None
        self.prev_time = None
        self.opt_loc=np.array([0.0,0.0])
        self.radius = radius
        self.power = power
        self.speed=slew_speed
        self.input_loc_y=0.0
        self.input_loc_x=0.0
        self.kernal = np.array([[]])
        self.grid_x = np.array([[]])
        self.grid_y = np.array([[]])
        
        # Define size of grid
        self.y_len = env.y_len
        self.x_len = env.x_len_F
        self.num_y_ratio = env.num_y_ratio
        self.num_x_ratio = env.num_x_ratio
        
        # Calculate number of transverse and longitudinal nodes
        temporary = np.sqrt(self.y_len*self.y_len + 2.0*self.x_len*(2.0*state_size-1)*self.y_len + self.x_len*self.x_len)
        self.num_y = int(round((temporary - self.y_len + self.x_len) / (2.0*self.x_len)))
        self.num_x = int(round((temporary + self.y_len - self.x_len) / (2.0*self.y_len)))

        # Calculate step sizes of the controller field
        self.dy = self.y_len / (self.num_y - 1)
        self.dx = self.x_len / (self.num_x - 1)
        self.dz = env.z_len

        # Build laplacian matrix
        padded_field = np.zeros((self.num_y+2, self.num_x+2, 3))
        padded_field[1:-1,1:-1,1:-1] = 1.0
        padded_field = padded_field.flatten()
        int_inds = np.arange(padded_field.shape[0])[padded_field==1]
        self.laplacian_matrix = env.build_laplacian_mat(self.num_y,self.num_x,1,self.dy,self.dx,self.dz,order=[1,1,1]).toarray()
        self.laplacian_matrix = self.laplacian_matrix[int_inds,:]
        self.laplacian_matrix = self.laplacian_matrix[:,int_inds]

        # Generate state space system for temperature controller
        A = (self.kappa/(self.rho*self.cp)) * self.laplacian_matrix
        B = (1.0 / (self.rho*self.cp*self.dz)) * np.eye(len(A))

        # Solve LQR control problem for temperature controller
        if boundary_temp_bias == 0.0:
            Q = np.ones((self.num_y*self.num_x,))
        elif boundary_temp_bias == 1.0:
            Q = np.zeros((self.num_y,self.num_x))
            Q[0,:]=1.
            Q[-1,:]=1.
            Q[:,0]=1.
            Q[:,-1]=1.
            Q=Q/np.mean(Q)
            Q=Q.flatten()
        else:
            Q = np.zeros((max(self.num_y,self.num_x), max(self.num_y,self.num_x)))
            Q[0:int(round(max(self.num_y,self.num_x)*0.1)),:]=1
            Q[-int(round(max(self.num_y,self.num_x)*0.1)):,:]=1
            Q[:,0:int(round(max(self.num_y,self.num_x)*0.1))]=1
            Q[:,-int(round(max(self.num_y,self.num_x)*0.1)):]=1
            Q = ndimage.gaussian_filter(Q, -2*max(self.num_y,self.num_x)*(boundary_temp_bias)+2*max(self.num_y,self.num_x), mode='constant', cval=1.0)
            Q = ndimage.zoom(Q, zoom=[self.num_y/len(Q), self.num_x/len(Q[0])], mode='nearest', order=5)
            Q=Q/np.mean(Q)
            Q=Q.flatten()
        Q = QR_ratio*np.diag(Q)
        R = np.eye(len(A))
        self.K_temperature,_,_ = control.lqr(A,B,Q,R)
        self.K_temperature[abs(self.K_temperature)<1.0e-4]=0.0
    

    def get_tag_temp(self, a, v):
        """
        Gets a target temperature given a degree of cure and target front speed.

        Parameters
        ----------
        a : float
            The initial degree of cure [0.07,0.15].
        v : float
            The target front speed in mm/s [0.37,1.90].

        Returns
        -------
        T : float
            The required initial temperature [281.15,309.15] in Kelvin given the initial degree of cure to achieve the target front speed.

        """
    
        # Front fit coefficients
        c0=83.237022495062
        c1=-337.117520704445
        c2=-0.625520042790
        c3=-379.964069909699
        c4=2.742904708748
        c5=0.001185123181
        c6=1.375682848902
        c7=-0.005561755952
        min_T = 281.15
        max_T = 309.15
        
        # Use a binary search to optimize the T guess based on front speed fit
        for i in range(20):
            T = 0.5*(min_T+max_T)
            err = (c0+c1*a+c2*T+c3*a*a+c4*a*T+c5*T*T+c6*a*a*T+c7*a*T*T)-v
            next_guess_bigger = err<-0.0003
            next_guess_smaller = err>0.0003
            if next_guess_bigger or next_guess_smaller:
                if next_guess_bigger:
                    min_T = T
                if next_guess_smaller:
                    max_T = T
            else:
                break    
    
        return T

    
    def get_opt_glob_input(self, temperature, target):
        """
        Calculates the optimal global input given a current temperature field and a target temperature field

        Parameters
        ----------
        temperature : array_like, shape (n,m)
            The current temperature field in Kelvin.
        target : float or array_like, shape (p,q)
            The target temperature field in Kelvin.

        Returns
        -------
        optimal_input : array_like, shape (n,m)
            The optimal global heat flux in Watts/Meter^2.

        """
        
        # Scale the tempeature grid and target temperature (grid) to the proper dimensions
        scaled_temperature = ndimage.zoom(temperature, zoom=[self.num_y/len(temperature), self.num_x/len(temperature[0])], mode='nearest', order=5)
        try:
            target = ndimage.zoom(target, zoom=[self.num_y/len(target), self.num_x/len(target[0])], mode='nearest', order=5)
        except:
            pass
        
        # Calculate optimal energy flux
        error = scaled_temperature-target
        error = error.flatten()
        optimal_input = (-self.K_temperature)@error
        optimal_input = optimal_input.reshape(self.num_y, self.num_x)
        
        # Resize the optimal_input back to the original size of the temperature_grid
        optimal_input = ndimage.zoom(optimal_input, (len(temperature)/self.num_y, len(temperature[0])/self.num_x))
        
        # Return the optimal input
        return optimal_input
    
    
    def get_opt_loc_input(self, env, target, update_input_loc=True):
        """
        Calculates the optimal local heat input to achieve target temperature profile

        Parameters
        ----------
        env : Environment.Env
            The simulation environment in which the controller acts.
        target : float or array_like, shape (p,q)
            The target temperature field in Kelvin.
        update_input_loc : bool, optional
            Boolean flag that indicates whether the location of the local input should be updated. The default is True.

        Returns
        -------
        left_local_input_flux : array_like, shape (env.num_y_CL, env.num_x_CL)
            The optimal heat flux (watts/meter^2 positive in) applied by a local heat input to the coarse left field of the environment.
        cen_local_input_flux : array_like, shape (env.num_y_F, env.num_x_F)
            The optimal heat flux (watts/meter^2 positive in) applied by a local heat input to the fine field of the environment.
        right_local_input_flux : array_like, shape (env.num_y_CR, env.num_x_CR)
            The optimal heat flux (watts/meter^2 positive in) applied by a local heat input to the coarse right field of the environment.

        """
        
        # Set the temperature field and current time
        temperature = env.temp_F[:,:,-1]
        time = env.times_C[env.time_ind_C]
        
        # If the local input location does not equal the target position, update the local input location based on the time elapsed since last update
        if self.prev_time != None and (self.opt_loc[0]!=self.input_loc_y or self.opt_loc[1]!=self.input_loc_x):
            opt_dirn_slew = self.opt_loc - np.array([self.input_loc_y, self.input_loc_x])
            opt_dirn_slew = opt_dirn_slew/(np.sqrt(opt_dirn_slew[0]**2+opt_dirn_slew[1]**2))
            mag_of_move = np.sqrt((self.opt_loc[0] - self.input_loc_y)**2 + (self.opt_loc[1] - self.input_loc_x)**2)
            if self.speed*(time - self.prev_time) > mag_of_move:
                new_position = self.opt_loc
            else:
                new_position = np.array([self.input_loc_y, self.input_loc_x]) + opt_dirn_slew*self.speed*(time - self.prev_time)
            self.input_loc_y = new_position[0]
            self.input_loc_x = new_position[1]
        self.prev_time = time
        
        # Calculate the step sizes of the temperature field given as an input
        curr_dy = self.y_len / (len(temperature) - 1)
        curr_dx = self.x_len / (len(temperature[0]) - 1)
        
        # Update the gaussian kernal if needed
        if curr_dy != self.dy or curr_dx != self.dx:
            
            # Update grid
            self.dy = curr_dy
            self.dx = curr_dx
            y_linspace = np.linspace(0.0, self.y_len, len(temperature))
            x_linspace = np.linspace(0.0, self.x_len, len(temperature[0]))
            self.grid_x, self.grid_y = np.meshgrid(x_linspace, y_linspace)
        
            # Update kernal
            input_const = -1.0 / (0.2171472409514 * self.radius * self.radius)
            nx_in_kernal = int(np.rint(2.0*self.radius/self.dx))
            ny_in_kernal = int(np.rint(2.0*self.radius/self.dy))
            left_x_ind = int(-0.5*nx_in_kernal)
            right_x_ind = int(0.5*nx_in_kernal)
            left_y_ind = int(-0.5*ny_in_kernal)
            right_y_ind = int(0.5*ny_in_kernal)
            self.kernal = []
            curr_row = 0
            for i in np.arange(left_y_ind, right_y_ind+1):
                y_loc = i * self.dy
                self.kernal.append([])
                for j in np.arange(left_x_ind, right_x_ind+1):
                    x_loc = j * self.dx
                    self.kernal[curr_row].append(np.exp((x_loc)**2*input_const + (y_loc)**2*input_const))
                curr_row = curr_row + 1
            self.kernal = np.array(self.kernal)
            self.kernal[self.kernal<0.01]=0.0
            
            # Update peak flux
            integral = np.trapz(np.trapz(self.kernal,np.linspace(0.0, nx_in_kernal*self.dx, nx_in_kernal)),np.linspace(0.0, ny_in_kernal*self.dy, ny_in_kernal))
            self.peak_flux = self.power / integral
        
        # Get the optimal global input
        opt_global_input = self.get_opt_glob_input(temperature, target)
        opt_global_input[opt_global_input<0.0]=0.0
        opt_global_input[opt_global_input>self.peak_flux]=self.peak_flux
        
        # Determine the closest indices of the input
        closest_y_ind = np.argmin(abs(self.grid_y[:,0] - self.input_loc_y))
        closest_x_ind = np.argmin(abs(self.grid_x[0,:] - self.input_loc_x))
        
        # Create a local mask of size y_len + kernal radius by x_len + kernal radius
        local_input_flux = np.zeros((len(opt_global_input)+2*int((len(self.kernal)-1)/2), len(opt_global_input[0])+2*int((len(self.kernal[0])-1)/2)))
        
        # Apply the kernal to the local mask centered about (closest_y_ind,closest_x_ind)
        local_input_flux[closest_y_ind:closest_y_ind+len(self.kernal), closest_x_ind:closest_x_ind+len(self.kernal[0])] = self.kernal
        
        # Trim the local mask so that it is size y_len by x_len
        left_local_input_flux = local_input_flux[int((len(self.kernal)-1)/2):len(local_input_flux)-int((len(self.kernal)-1)/2), 0:int((len(self.kernal[0])-1)/2)]
        right_local_input_flux = local_input_flux[int((len(self.kernal)-1)/2):len(local_input_flux)-int((len(self.kernal)-1)/2), len(local_input_flux[0])-int((len(self.kernal[0])-1)/2):]
        cen_local_input_flux = local_input_flux[int((len(self.kernal)-1)/2):len(local_input_flux)-int((len(self.kernal)-1)/2), int((len(self.kernal[0])-1)/2):len(local_input_flux[0])-int((len(self.kernal[0])-1)/2)]
        
        # Scale the left and right local fluxes so that they are in the coarse scale
        left_local_input_flux = ndimage.zoom(left_local_input_flux, zoom=[1./self.num_y_ratio, 1./self.num_x_ratio], mode='nearest', order=5)
        left_local_input_flux[left_local_input_flux<0.01]=0.0
        right_local_input_flux = ndimage.zoom(right_local_input_flux, zoom=[1./self.num_y_ratio, 1./self.num_x_ratio], mode='nearest', order=5)
        right_local_input_flux[right_local_input_flux<0.01]=0.0
        
        # Adjust left local flux so it has the same dimensions as the environment's left field
        if env.num_x_CL != 0:
            temporary = np.zeros((env.temp_CL[:,:,0]).shape)
            
            # Local flux bigger than coarse left field
            if len(left_local_input_flux[0,:]) > env.num_x_CL:
                temporary = left_local_input_flux[:,-len(temporary[0,:]):]
            
            # Local flux exactly the same size as the coarse left field
            elif len(left_local_input_flux[0,:]) == env.num_x_CL:
                temporary = left_local_input_flux
            
            # Coarse left field is bigger than local flux
            else:
                temporary[:,-len(left_local_input_flux[0,:]):] = left_local_input_flux
                
            left_local_input_flux = temporary
            
        # If the environment's left field does not exist, give 0 heat flux
        else:
            left_local_input_flux = 0.0
            
        # Adjust right local flux so it has the same dimensions as the environment's right field
        if env.num_x_CR != 0:
            temporary = np.zeros((env.temp_CR[:,:,0]).shape)
            
            # Local flux bigger than coarse right field
            if len(right_local_input_flux[0,:]) > env.num_x_CR:
                temporary = right_local_input_flux[:,0:len(temporary[0,:])]
                
            # Local flux exactly the same size as the coarse right field
            elif len(right_local_input_flux[0,:]) == env.num_x_CR:
                temporary = right_local_input_flux    
                
            # Coarse right field is bigger than local flux
            else:
                temporary[:,0:len(right_local_input_flux[0,:])] = right_local_input_flux
            
            right_local_input_flux = temporary
          
        # If the environment's right field does not exist, give 0 heat flux
        else:
            right_local_input_flux = 0.0
        
        # Determine optimal local input magnitude based on the input kernal and the current input location
        opt_mag = ((np.trace(np.matmul(np.transpose(cen_local_input_flux), opt_global_input))) / (np.trace(np.matmul(np.transpose(cen_local_input_flux), cen_local_input_flux)))) / self.peak_flux
        if opt_mag > 1.0:
            opt_mag = 1.0
        elif opt_mag < 0.0:
            opt_mag = 0.0
            
        # Convert the local input mask to a local heat flux
        left_local_input_flux = left_local_input_flux*opt_mag*self.peak_flux
        cen_local_input_flux = cen_local_input_flux*opt_mag*self.peak_flux
        right_local_input_flux = right_local_input_flux*opt_mag*self.peak_flux
        
        # Determine the optimal next position for the input
        if update_input_loc:
            convolved_opt_global_input = signal.convolve2d(opt_global_input, self.kernal, mode='same', boundary='symm', fillvalue=0.0)
            convolved_opt_global_input = (convolved_opt_global_input) / (np.sum(self.kernal)*(self.peak_flux))
            self.opt_loc = np.unravel_index(np.argmax(convolved_opt_global_input), convolved_opt_global_input.shape)
            self.opt_loc = np.array([self.grid_y[self.opt_loc], self.grid_x[self.opt_loc]])
            
        # Return the local input flux
        return left_local_input_flux, cen_local_input_flux, right_local_input_flux