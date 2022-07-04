# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:34:32 2022

@author: Grayson Schaer
"""

import numpy as np
import opensimplex
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import ndimage
from scipy import signal
from collections import deque
import pickle
import os
import copy

#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#

class Env:
    def __init__(self,thermal_conductivity=0.152,specific_heat_capacity=1600.,density=980.,
                 mean_enthalpy_rxn=350000.,max_deviation_enthalpy_rxn=0.,init_enthalpy_rxn_seed=427845,pre_exponential=8.55e15,activation_energy=110750.,PT_m=0.77,PT_n=1.72,diffusion_const=14.48,critical_conversion=0.41,
                 laplacian_order=[2,3,2],imp_cure_rate=0.5,
                 total_time=30.,time_step=0.01,time_step_fine_scale=0.20,
                 y_length=0.008,x_length=0.05,z_length=0.001,x_length_fine=0.008,
                 y_step=3.0e-4,x_step=3.0e-4,z_step=3.0e-4,y_step_fine_scale=0.5,x_step_fine_scale=0.1,z_step_fine_scale=0.5,
                 mean_init_temp=293.15,max_deviation_init_temp=0.,init_temp_seed=124556,mean_init_cure=0.10,max_deviation_init_cure=0.,init_cure_seed=789345,
                 heat_transfer_coeff=0.,ambient_temp=293.15,
                 trigger_heat_flux=30000.,
                 render=True,frame_rate=30.,save_path='../results',
                 sim_cure_post_front=False,
                 verbose=False):
        """
        Initializes FROMP simulation environment.

        Parameters
        ----------
        thermal_conductivity : float, optional
            Thermal conductivity of material in Watts/Meter-Kelvin. The default is 0.152.
        specific_heat_capacity : float, optional
            Specific heat capacity of material in Joules/Kilogram-Kelvin. The default is 1600..
        density : float, optional
            Density of material in Kilograms/Meter^3. The default is 980..
        mean_enthalpy_rxn : float, optional
            The mean enthalpy of reaction in Joules/Kilogram. The default is 350000..
        max_deviation_enthalpy_rxn : float, optional
            The maximum magnitude of deviation from the mean for the noisy enthalpy of reaction field. Joules/Kilogram. The default is 0..
        init_enthalpy_rxn_seed : int, optional
            Seed used to generate simplex noise for enthalpy of reaction field. The default is 427845.
        pre_exponential : float, optional
            Arrhenius pre-exponential factor in 1/seconds. The default is 8.55e15.
        activation_energy : float, optional
            The activation energy of the monomer in Joules/Mol. The default is 110750..
        PT_m : float, optional
            The Prout-Tompkins model mth order coefficient. The default is 0.77.
        PT_n : float, optional
            The Prout-Tompkins model nth order coefficient. The default is 1.72.
        diffusion_const : float, optional
            The diffusion corrected Prout-Tompkins model diffusion constant. The default is 14.48.
        critical_conversion : float, optional
            The diffusion corrected Prout-Tompkins model critical conversion value. The default is 0.41.
        laplacian_order : array_like, shape(3,), optional
            The order of the discrete laplacian for each coordinate direction. [y-dirn, x-dirn, z-dirn]. The default is [2,2,2].
        imp_cure_rate : float, optional
            The critical value of cure rate in 1/second at which finite difference integration switches from explicit to implicit. The default is 0.5.
        total_time : float, optional
            The total time of the simulation in seconds. The default is 30..
        time_step : float, optional
            The size of the discrete time step in seconds. The default is 0.01.
        time_step_fine_scale : float (1/int), optional
            The ratio of fine grid time step over coarse grid time step. Must be 1/integer. The default is 0.20.
        y_length : float, optional
            The length in meters of the rectagular simultion domain in the y corridinate direction. The default is 0.008.
        x_length : float, optional
            The length in meters of the rectagular simultion domain in the x corridinate direction. The default is 0.05.
        z_length : float, optional
            The length in meters of the rectagular simultion domain in the z corridinate direction. The default is 0.001.
        x_length_fine : float, optional
            The length in meters of the fine portional of the rectangular simulation domain in the x coordinate direction. Must be smaller than x_length. The default is 0.008.
        y_step : float, optional
            The size in meters of the discrete spatial grid in the y cooridinate direction. The default is 3.0e-4.
        x_step : float, optional
            The size in meters of the discrete spatial grid in the x cooridinate direction. The default is 3.0e-4.
        z_step : float, optional
            The size in meters of the discrete spatial grid in the z cooridinate direction. The default is 3.0e-4.
        y_step_fine_scale : float (1/int), optional
            The ratio of the fine spatial grid step size in the y coordinate direction over the coarse spatial grid step size in the y coordinate direction.
            Must be 1/integer. The default is 0.5.
        x_step_fine_scale : float (1/int), optional
            The ratio of the fine spatial grid step size in the x coordinate direction over the coarse spatial grid step size in the x coordinate direction.
            Must be 1/integer. The default is 0.1.
        z_step_fine_scale : float (1/int), optional
            The ratio of the fine spatial grid step size in the z coordinate direction over the coarse spatial grid step size in the z coordinate direction.
            Must be 1/integer. The default is 0.5.
        mean_init_temp : float, optional
            The mean value in Kelvin of the initial temperature field. The default is 293.15.
        max_deviation_init_temp : float, optional
            The maximum magnitude of deviation from the mean for the noisy initial temperature field. Kelvin. The default is 0..
        init_temp_seed : int, optional
            Seed used to generate simplex noise for initial temperature field. The default is 124556.
        mean_init_cure : float, optional
            The mean value of the initial degree of cure field. The default is 0.10.
        max_deviation_init_cure : float, optional
            The maximum magnitude of deviation from the mean for the noisy initial degree of cure field. The default is 0..
        init_cure_seed : int, optional
            Seed used to generate simplex noise for initial degree of cure field. The default is 789345.
        heat_transfer_coeff : float, optional
            The heat transfer coefficient of the boundary of the simulation domain in Watts/Meter^2-Kelvin. The default is 0..
        ambient_temp : float, optional
            The ambient temperature in Kelvin. The default is 293.15.
        trigger_heat_flux : float, optional
            The heat flux of the thermal trigger in Watts/Meter^2. The default is 30000..
        render : bool, optional
            A boolean flag that indicates whether frames will be saved for rendering. The default is True.
        frame_rate : float, optional
            The frame rate in 1/second at which data are saved. The default is 30..
        save_path : float, optional
            The path to the folder in which data are saved. The default is '../results'.
        sim_cure_post_front : bool, optional
            A boolean flag that indicates whether the cure field beind the front in the left coarse field is simulated. 
            If the front yields complete or nearly complete conversion, it is not necessary to simulate cure in the coarse field behind the front. The default is False.
        verbose : bool, optional
            A boolena flag that indicates whether status will be written to cout. The default is False.

        Returns
        -------
        None.

        """
        
        if verbose:
            print("Initializing environment...")
        
        # Define material thermal properties
        self.kappa = thermal_conductivity  ## Watts / Meter * Kelvin
        self.cp = specific_heat_capacity   ## Joules / Kilogram * Kelvin
        self.rho = density                 ## Kilograms / Meter ^ 3
        
        # Define material cure properties
        self.mean_hr = mean_enthalpy_rxn
        self.dev_hr = max_deviation_enthalpy_rxn
        self.A = pre_exponential
        self.E = activation_energy
        self.R = 8.31446261815324
        self.m = PT_m
        self.n = PT_n
        self.C = diffusion_const
        self.ac = critical_conversion
        
        # Define laplacian order for each dimension
        self.order_C = laplacian_order
        self.order_F = laplacian_order
        self.imp_cure_rate = imp_cure_rate
        
        # Define time
        self.tot_t = total_time  ## Seconds
        self.dt_C = time_step    ## Seconds
        self.dt_F = time_step_fine_scale * time_step  ## Seconds
        
        # Define domain 
        self.y_len = y_length            ## Meters
        self.x_len = x_length            ## Meters
        self.z_len = z_length            ## Meters
        self.x_len_F = x_length_fine     ## Meters
        self.dy_C = y_step        ## Meters
        self.dx_C = x_step        ## Meters
        self.dz_C = z_step        ## Meters
        self.dy_F = y_step_fine_scale * y_step        ## Meters
        self.dx_F = x_step_fine_scale * x_step        ## Meters
        self.dz_F = z_step_fine_scale * z_step        ## Meters
        
        # Define initial conditions
        self.mean_T0 = mean_init_temp
        self.dev_T0 = max_deviation_init_temp
        self.mean_a0 = mean_init_cure
        self.dev_a0 = max_deviation_init_cure
        
        # Define boundary conditions
        self.htc = heat_transfer_coeff
        self.amb_T = ambient_temp
        
        # Define heat inputs
        self.q_trig = trigger_heat_flux  ## Watts / Meter ^ 2
        
        # Define save options
        self.save_frames = render
        self.fr = frame_rate
        self.path = save_path
        
        if verbose:
            print("\tCalculate number of points in grid...")
        # Fine grid points and length
        self.num_y_F = int(np.rint((self.y_len/self.dy_F)+1.))
        self.num_x_F = int(np.rint((self.x_len_F/self.dx_F)+1.))
        self.num_z_F = int(np.rint((self.z_len/self.dz_F)+1.))
        self.y_len = (self.num_y_F-1)*self.dy_F
        self.x_len_F = (self.num_x_F-1)*self.dx_F
        self.z_len = (self.num_z_F-1)*self.dz_F
        
        # Right coarse grid points and length
        self.num_y_CR = int(np.rint((self.y_len/self.dy_C)+1.))
        self.num_x_CR = int(np.rint(((self.x_len-self.x_len_F)/self.dx_C)+1.))
        if self.num_x_CR < 8:
            self.num_x_CR = 8
        self.num_z_CR = int(np.rint((self.z_len/self.dz_C)+1.))
        self.x_len = (self.num_x_CR-1)*self.dx_C + self.x_len_F
        
        # Left coarse grid points and length
        self.num_y_CL = self.num_y_CR
        self.num_x_CL = 0
        self.num_z_CL = self.num_z_CR
        
        # Fine grid
        y_space_F = np.linspace(0,self.y_len,self.num_y_F)
        x_space_F = np.linspace(0,self.x_len_F,self.num_x_F)
        z_space_F = np.linspace(0,self.z_len,self.num_z_F)
        self.x_grid_F, self.y_grid_F, self.z_grid_F = np.meshgrid(x_space_F, y_space_F, z_space_F)
        
        # Right coarse grid
        y_space_CR = np.linspace(0,self.y_len,self.num_y_CR)
        x_space_CR = np.linspace(self.x_len_F,self.x_len,self.num_x_CR)
        z_space_CR = np.linspace(0,self.z_len,self.num_z_CR)
        self.x_grid_CR, self.y_grid_CR, self.z_grid_CR = np.meshgrid(x_space_CR, y_space_CR, z_space_CR)
        
        if verbose:
            print("\tGenerate initial fields...")
        # Build the global initial fields at the coarse resolution
        max_len = max(self.y_len,self.x_len,self.z_len)
        ar=[self.y_len/max_len,self.x_len/max_len,self.z_len/max_len]
        temp_global = self.build_rect_field(self.num_y_CR,int(np.rint((self.x_len/self.dx_C)+1.)),self.num_z_CR,self.mean_T0,max_dev=self.dev_T0,ar=ar,seed=init_temp_seed)
        cure_global = self.build_rect_field(self.num_y_CR,int(np.rint((self.x_len/self.dx_C)+1.)),self.num_z_CR,self.mean_a0,max_dev=self.dev_a0,ar=ar,seed=init_cure_seed)
        hr_global = self.build_rect_field(self.num_y_CR,int(np.rint((self.x_len/self.dx_C)+1.)),self.num_z_CR,self.mean_hr,max_dev=self.dev_hr,ar=ar,seed=init_enthalpy_rxn_seed)
        
        # Determine the ratio of number fine grid points to number coarse grid points over the entire domain in all three directions
        self.num_y_ratio = self.num_y_F / self.num_y_CR
        self.num_x_ratio = ((self.x_len/self.dx_F)+1.)/((self.x_len/self.dx_C)+1.)
        self.num_z_ratio = self.num_z_F / self.num_z_CR
        
        # Subdivide the initial fields into fine and coarse sections
        self.temp_F = ndimage.zoom(temp_global, zoom=[self.num_y_ratio,self.num_x_ratio,self.num_z_ratio], mode='nearest', order=5)[:,0:self.num_x_F,:]
        self.cure_F = ndimage.zoom(cure_global, zoom=[self.num_y_ratio,self.num_x_ratio,self.num_z_ratio], mode='nearest', order=5)[:,0:self.num_x_F,:]
        self.hr_F = ndimage.zoom(hr_global, zoom=[self.num_y_ratio,self.num_x_ratio,self.num_z_ratio], mode='nearest', order=5)[:,0:self.num_x_F,:]
        self.temp_CR = temp_global[:,-self.num_x_CR:,:]
        self.cure_CR = cure_global[:,-self.num_x_CR:,:]
        self.hr_CR = hr_global[:,-self.num_x_CR:,:]
        self.cure_post = sim_cure_post_front
        
        if verbose:
            print("\tBuild dynamical matrices...")
        self.laplacian_F = self.build_laplacian_mat(self.num_y_F,self.num_x_F,self.num_z_F,self.dy_F,self.dx_F,self.dz_F,order=self.order_F)
        self.laplacian_CR = self.build_laplacian_mat(self.num_y_CR,self.num_x_CR,self.num_z_CR,self.dy_C,self.dx_C,self.dz_C,order=self.order_C)
        
        if verbose:
           print("\tGenerate cure kinetics lookup tables...")   
        self.T_start, self.T_step, self.arrhenius = self.build_arrhenius(self.mean_T0,self.mean_a0,self.mean_hr,dev_T0=self.dev_T0,dev_a0=self.dev_a0,dev_hr=self.dev_hr,amb_T=self.amb_T,cp=self.cp,A=self.A,E=self.E)
        self.a_start, self.a_step, self.kinetics, self.d_kinetics = self.build_kinetics(self.mean_a0,dev_a0=self.dev_a0,m=self.m,n=self.n,C=self.C,ac=self.ac)
        
        if verbose:
            print("\tSetup front tracker...")
        self.front=[[0.0,0.0,0.0,0.0]]
        
        if verbose:
            print("\tGenerate time sequence...")
        self.time_ind_C = 0
        self.time_ind_F = 0
        self.times_C = np.linspace(0.0,self.tot_t,int(np.floor(self.tot_t/self.dt_C))+1)
        self.times_F = np.linspace(0.0,self.tot_t,int(np.floor(self.tot_t/self.dt_F))+1)
        
        if verbose:
            print("\tSetup frame buffer...")
        self.temp_images = deque([])
        self.cure_images = deque([])
        self.heat_images = deque([])
        self.qz_tot_F = np.zeros((self.num_y_F, self.num_x_F))
        self.qz_tot_CR = np.zeros((self.num_y_CR, self.num_x_CR))
        self.qz_tot_CL = np.zeros((self.num_y_CL, self.num_x_CL))
        self.frames=self.times_C[self.times_C%(1./self.fr)<self.dt_C]
        
        if verbose:
            print("Environment initialization complete!")
    

    def build_rect_field(self,num_y,num_x,num_z,mean,max_dev=0.0,feat_size=0.25,ar=[1.,1.,1.],seed=10000):
        """
        Generates a 3D, rectangular, scalar field of specified size and mean.
    
        Parameters
        ----------
        num_y : int
            Number of coordinates in y dimension (dim=0).
        num_x : int
            Number of coordinates in x dimension (dim=1).
        num_z : int
            Number of coordinates in z dimension (dim=2).
        mean : float
            Mean value of field.
        max_dev : float, optional
            The magnitude of the maximum deviation of the field from the mean value. 
            Deviations are generated via simplex noise. The default is 0.0.
        feat_size : float [0.0,1.0], optional
            The size of the simplex noise features as a percent of the largest dimension. The default is 0.25.
        ar : array_like, shape (3,), optional
            Aspect ratio of the y dimension, x dimension, and z dimension, respectively. 
            Aspect ratio of the ith dimension is defined as (length_i / max_length_xyz). The default is [1.,1.,1.].
        seed : int, optional
            Seed used to generate simplex noise. The default is 10000.
    
        Returns
        -------
        field : array_like, shape (num_y,num_x,num_z)
            The rectangular scalar field with specified dimensions, mean, and noise.
    
        """
        
        if num_y>1 and num_x>1 and num_z>1:
            # Attempt to load cached data
            string = str((num_y,num_x,num_z,feat_size,ar,seed))
            string = string.replace(", ","_")
            string = string.replace("(","")
            string = string.replace(")","")
            string = string.replace("[","")
            string = string.replace("]","")
            string = string.replace(".","-")
            string="cache/"+string+'.dat'
            
            # If cached data does not exist, create noise
            if not os.path.exists(string):
        
                # Determine feature sizes for noise in y, x, and z directions
                y_feat_size = (feat_size*(num_y-1)/ar[0]) + 1.0
                x_feat_size = (feat_size*(num_x-1)/ar[1]) + 1.0
                z_feat_size = (feat_size*(num_z-1)/ar[2]) + 1.0
            
                # Get noise shape over field given feature size
                opensimplex.seed(seed)
                i = np.arange(0,num_y)/y_feat_size
                j = np.arange(0,num_x)/x_feat_size
                k = np.arange(0,num_z)/z_feat_size
                field = opensimplex.noise3array(k, j, i)        
        
                # Cache noisy field
                with open(string, 'wb') as file:
                    pickle.dump(field, file)
            
            # Load cached noisy field
            else: 
                with open(string, 'rb') as file:
                    field = pickle.load(file)
        
            # Scale field so that mean and max deviation are satisfied
            field = field - np.mean(field)             ## Set mean to 0
            field = field * (1.0/np.max(abs(field)))   ## Set max deviation magnitude to 1
            field = field * max_dev + mean             ## Set max deviation magnitude to max_dev, set mean to mean
            
        else:
            field = mean*np.ones((num_y,num_x,num_z))
            
        return field
        
    
    def build_laplacian_mat(self,num_y,num_x,num_z,dy,dx,dz,order=[1,1,1]):
        """
        Builds a matrix that, when matrix multiplied by a row-major (C-style) collapsed 3D field, returns the row-major collapsed laplaician of that field.
    
        Parameters
        ----------
        num_y : int
            Number of coordinates in y dimension (dim=0).
        num_x : int
            Number of coordinates in x dimension (dim=1).
        num_z : int
            Number of coordinates in z dimension (dim=2).
        dy : float
            Discrete step size of the rectangular field in the y direction (dim=0).
        dx : float
            Discrete step size of the rectangular field in the x direction (dim=1).
        dz : float
            Discrete step size of the rectangular field in the z direction (dim=2).
        order : array_like, shape (3,), optional
            Order of discrete laplacian in each dimension. Must be bewtween 1 and 3. The default is [1,1,1].
    
        Returns
        -------
        Compressed Sparse Row matrix, shape((num_y+2*order[0])*(num_x+2*order[1])*(num_z+2*order[2]), (num_y+2*order[0])*(num_x+2*order[1])*(num_z+2*order[2]))
            Sparse matrix containing the laplacian matrix
    
        """
        
        # Attempt to load cached data
        string = str((num_y,num_x,num_z,dy,dx,dz,order))
        string = string.replace(", ","_")
        string = string.replace("(","")
        string = string.replace(")","")
        string = string.replace("[","")
        string = string.replace("]","")
        string = string.replace(".","-")
        string="cache/"+string+'.dat'
        
        # If cached data does not exist, create laplacian
        if not os.path.exists(string):
            
            # Initial coefficients and arrays used to store sparse matrix data
            fd_coeffs = [np.array([1.,-2.,1.]),np.array([-1./12.,4./3.,-5./2.,4./3.,-1./12.]),np.array([1./90.,-3./20.,3./2.,-49./18.,3./2.,-3./20.,1./90.])]
            padded_shape = (num_y+2*order[0])*(num_x+2*order[1])*(num_z+2*order[2])
            laplacian_rows = deque([])
            laplacian_cols = deque([])
            laplacian_dat = deque([])
            
            # Calculate the laplacian coeffs for each row
            for row in range(padded_shape):
                
                # Get global 3D coordinates of point corresponding current row's equation
                y_coord = (row) // ((num_x+2*order[1])*(num_z+2*order[2]))
                x_coord = ((row) % ((num_x+2*order[1])*(num_z+2*order[2]))) // (num_z+2*order[2])
                z_coord = ((row) % ((num_x+2*order[1])*(num_z+2*order[2]))) % (num_z+2*order[2])
                
                # Get global 3D coordinates of adjacent points
                y_coord_adj = y_coord + np.concatenate([np.arange(-order[0],order[0]+1),
                                                        np.zeros((order[1]*2+order[2]*2+2),dtype=int)])
                x_coord_adj = x_coord + np.concatenate([np.zeros((order[0]*2+1),dtype=int),
                                                        np.arange(-order[1],order[1]+1),
                                                        np.zeros((order[2]*2+1),dtype=int)])
                z_coord_adj = z_coord + np.concatenate([np.zeros((order[0]*2+order[1]*2+2),dtype=int),
                                                        np.arange(-order[2],order[2]+1)])
                
                # Convert adjecent point coordinates to indices
                cols = y_coord_adj*(num_x+2*order[1])*(num_z+2*order[2]) + x_coord_adj*(num_z+2*order[2]) + z_coord_adj
                
                # Only add valid rows to laplacian matrix
                if (cols>=0).all() and (cols<padded_shape).all():
                    
                    # Get adj and center coeffs
                    y_dat = fd_coeffs[order[0]-1]/(dy*dy)
                    x_dat = fd_coeffs[order[1]-1]/(dx*dx)
                    z_dat = fd_coeffs[order[2]-1]/(dz*dz)
                    dat = np.concatenate((y_dat, np.concatenate((x_dat,z_dat))))
                    
                    # Get adjacent coeffs
                    adj_rows = row*np.ones(cols[cols!=row].shape,dtype=int)
                    adj_cols = cols[cols!=row]
                    adj_dat = dat[cols!=row]
                    
                    # Sum center coeffs
                    cen_dat = np.sum(dat[cols==row])
                    
                    # Add data
                    laplacian_rows.extend(adj_rows)
                    laplacian_rows.extend([row])
                    laplacian_cols.extend(adj_cols)
                    laplacian_cols.extend([row])
                    laplacian_dat.extend(adj_dat)
                    laplacian_dat.extend([cen_dat])
                        
            # Build laplacian
            laplacian = sparse.csr_matrix((laplacian_dat,(laplacian_rows,laplacian_cols)),shape=(padded_shape,padded_shape))        
            
            # Cache noisy field
            with open(string, 'wb') as file:
                pickle.dump(laplacian, file)
                
        # Load cached laplacian
        else: 
            with open(string, 'rb') as file:
                laplacian = pickle.load(file)
                
        # Build laplacian matrix
        return laplacian
    
    
    def build_arrhenius(self,mean_T0,mean_a0,mean_hr,dev_T0=0.,dev_a0=0.,dev_hr=0.,num=1000000,amb_T=298.15,cp=1600.,A=8.55e15,E=110750.):
        """
        Evaluates the Arrhenius equation over a discrete, linear temperature space.
    
        Parameters
        ----------
        mean_T0 : float
            The mean of the initial temperature field in Kelvin.
        mean_a0 : float
            The mean of the initial degree of cure field. Between 0.0 and 1.0.
        mean_hr : float
            The mean of the enthalpy of reaction field in Joules / Kilogram.
        dev_T0 : float, optional
            The maximum absolute deviation from the mean of the initial temperature field in Kelvin. The default is 0..
        dev_a0 : float, optional
            The maximum absolute deviation from the mean of the initial degree of cure field. The default is 0..
        dev_hr : float, optional
            The maximum absolute deviation from the mean of the enthalpy of reaction field in Joules / Kilogram. The default is 0..
        num : int, optional
            The number of discrete temperature points in the temperature space. The default is 1E6.
        amb_T : float, optional
            The ambient temperature in Kelvin. The default is 298.15.
        cp : float, optional
            The specific heat capacity of the material in Joules / Kilogram-Kelvin. The default is 1600..
        A : float, optional
            The pre-exponential term of the Arrhenius equation in 1/second. The default is 8.55e15.
        E : float, optional
            The activation energy of the reaction in Joules / mol. The default is 110750..
    
        Returns
        -------
        T_start : float
            The minimum temperature value in Kelvin at which the Arrhenius equation is evaluated.
        T_step : float
            The discrete step size in Kelvin of the discrete, linear temperature space over which the Arrhenius equation is evaluated.
        arrhenius : array_like, shape (num,)
            The evaluation of the Arrhenius equation over the discrete, linear temperature space.
    
        """
        
        # Build linear temperature space
        T_start = 0.90*min(mean_T0-dev_T0,amb_T)
        T_end = 1.5*(((mean_hr+dev_hr)*(1.0-(mean_a0-dev_a0)))/cp+(mean_T0+dev_T0))
        T_space, T_step = np.linspace(T_start, T_end, num, retstep=True)
        
        # Calculate and return Arrhenius equation over linear temperature space
        R = 8.314462618  ## Universal gas constant in Joules / Kilogram-mol
        arrhenius = A*np.exp(-E/(R*T_space))
        
        # Return the temperature space parameters and the arrhenius evaluation
        return T_start, T_step, arrhenius
        
        
    def build_kinetics(self,mean_a0,dev_a0=0.,num=1000000,m=0.77,n=1.72,C=14.48,ac=0.41):
        """
        Evaluates the Diffusion Extended Prout-Tompkins model over a discrete, linear degree of cure space.
    
        Parameters
        ----------
        mean_a0 : float
            The mean of the initial degree of cure field. Between 0.0 and 1.0.
        dev_a0 : float, optional
            The maximum absolute deviation from the mean of the initial degree of cure field. The default is 0..
        num : int, optional
            The number of discrete degree of cure points in the degree of cure space. The default is 1E6.
        m : float, optional
            The Prout-Tompkins model mth order coefficient. The default is 0.77.
        n : float, optional
            The Prout-Tompkins model nth order coefficient. The default is 1.72.
        C : float, optional
            The diffusion corrected Prout-Tompkins model diffusion constant. The default is 14.48.
        ac : float, optional
            The diffusion corrected Prout-Tompkins model critical conversion value. The default is 0.41.
    
        Returns
        -------
        a_start : float
            The minimum degree of cure value at which the Diffusion Extended Prout-Tompkins model is evaluated.
        a_step : float
            The discrete step size of the discrete, linear degree of cure space over which the Diffusion Extended Prout-Tompkins model is evaluated.
        kinetics : array_like, shape (num,)
            The evaluation of the Diffusion Extended Prout-Tompkins model over the discrete, linear degree of cure space.
        d_kinetics : array_like, shape (num,)
            The evaluation of the first derivative of the Diffusion Extended Prout-Tompkins model
            with respect to degree of cure over the discrete, linear degree of cure space.
    
        """
        
        # Build linear degree of cure space
        a_start = 0.99*(mean_a0-dev_a0)
        a_space, a_step = np.linspace(a_start,0.999999999,num,retstep=True)
        
        # Calculate and cure kinetics over linear degree of cure space
        kinetics = (abs(a_space)/a_space)*(((abs(a_space)**m)*((1.0-abs(a_space))**n))/(1.0+np.exp(C*(abs(a_space)-ac))))
        
        # Calculate and first derivative of cure kinetics with respect to degree of cure evaluated over linear degree of cure space
        d_kinetics = (abs(a_space)/a_space)*((m*abs(a_space)**(m-1.)*(1.-abs(a_space))**n)/(1.+np.exp(C*(abs(a_space)-ac))) + 
                          (-n*abs(a_space)**m*(1.-abs(a_space))**(n-1.))/(1.+np.exp(C*(abs(a_space)-ac))) + 
                          (-C*abs(a_space)**m*(1.-abs(a_space))**n*np.exp(C*(abs(a_space)-ac)))/((1.0+np.exp(C*(abs(a_space)-ac)))**2))
        
        # Return the degree of cure space parameters and the kinetics evaluations
        return a_start, a_step, kinetics, d_kinetics
    
    
    def step(self,heat_CL=0.0, heat_F=0.0, heat_CR=0.0):
        """
        Steps the simulation environment forward one coarse time step.
        
        Parameters
        ----------
        heat_CL : float or array-like, shape (num_y_CL,num_x_CL)
            Externally applied heat flux (watts/meter^2 positive in) to the z+ surface of the coarse left grid. The default is 0.0.
        heat_F : float or array-like, shape (num_y_F,num_x_F)
            Externally applied heat flux (watts/meter^2 positive in) to the z+ surface of the fine grid. The default is 0.0.
        heat_CR : float or array-like, shape (num_y_CR,num_x_CR)
            Externally applied heat flux (watts/meter^2 positive in) to the z+ surface of the coarse right grid. The default is 0.0.
            
        Returns
        -------
        bool
            Boolean flag that indicates whether any simulation termination conditions have been met.

        """
        
        # Determine next time or if simulation is done
        if self.time_ind_C+1 < len(self.times_C):
            next_time = self.times_C[self.time_ind_C + 1]
            half_next_time = 0.5*(self.times_C[self.time_ind_C + 1] - self.times_C[self.time_ind_C]) + self.times_C[self.time_ind_C]
            self.time_ind_C = self.time_ind_C + 1
        else:
            return True
        
        # Step the fine field half way to the next time point
        while self.times_F[self.time_ind_F] < half_next_time:
            self.step_fine_field(heat=heat_F)
            if self.time_ind_F + 1 < len(self.times_F):
                self.time_ind_F = self.time_ind_F + 1
            else:
                return True
            
        # Step the left coarse field to the next time point
        if self.num_x_CL > 0:
            
            # Get the boundaries temperatures from environment and fine temperature field
            self.temp_CL[:,-1,:] = ndimage.zoom(self.temp_F[:,0,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
            padded_temp_CL,qy_tot,qx_tot,qz_tot = self.get_boundaries(self.temp_CL,self.dy_C,self.dx_C,self.dz_C,order=self.order_C,k=self.kappa,c=self.htc,bc=self.amb_T,qz=[0.0,heat_CL])
            self.qz_tot_CL = copy.deepcopy(qz_tot[1])
            
            # Enforce Neumann boundaries between fine and coarse faces
            if self.order_C[1] == 1:
                zoomed_temp_F_1 = ndimage.zoom(self.temp_F[:,1,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_right_C = (self.dx_C*zoomed_temp_F_1+self.dx_F*self.temp_CL[:,-2,:])/(self.dx_C+self.dx_F)
            if self.order_C[1] == 2:
                zoomed_temp_F_1 = ndimage.zoom(self.temp_F[:,1,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_F_2 = ndimage.zoom(self.temp_F[:,2,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_right_C = (self.dx_F*((1./12.)*self.temp_CL[:,-3,:]+(-2./3.)*self.temp_CL[:,-2,:])+self.dx_C*((1./12.)*zoomed_temp_F_2+(-2./3.)*zoomed_temp_F_1))/(((1./12.)-(2./3.))*(self.dx_C+self.dx_F))
            if self.order_C[1] >= 3:
                zoomed_temp_F_1 = ndimage.zoom(self.temp_F[:,1,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_F_2 = ndimage.zoom(self.temp_F[:,2,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_F_3 = ndimage.zoom(self.temp_F[:,3,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_right_C = (self.dx_F*((-1./60.)*self.temp_CL[:,-4,:]+(3./20.)*self.temp_CL[:,-3,:]+(-3./4.)*self.temp_CL[:,-2,:])+self.dx_C*((-1./60.)*zoomed_temp_F_3+(3./20.)*zoomed_temp_F_2+(-3./4.)*zoomed_temp_F_1))/(((-1./60.)+(3./20.)+(-3./4.))*(self.dx_C+self.dx_F))
            for i in range(self.order_C[1]):
                padded_temp_CL[self.order_C[0]:-self.order_C[0],-i-1,self.order_C[2]:-self.order_C[2]] = virtual_temps_right_C
                
            # Get the laplacian of the coarse left temperature field
            laplacian_CL = (self.laplacian_CL@padded_temp_CL.flatten()).reshape(self.num_y_CL+2*self.order_C[0],self.num_x_CL+2*self.order_C[1],self.num_z_CL+2*self.order_C[2])[self.order_C[0]:-self.order_C[0],self.order_C[1]:-self.order_C[1],self.order_C[2]:-self.order_C[2]]
            
            # Calculate the FE cure rates for the coarse left field
            if self.cure_post:
                f = self.arrhenius[np.rint((self.temp_CL-self.T_start)/(self.T_step)).astype(int)]
                g = self.kinetics[np.rint((self.cure_CL-self.a_start)/(self.a_step)).astype(int)]
                cure_rate_CL = f*g
                
                # Limit the FE cure rate so that the max cure value is exactly 0.999999999
                cure_rate_CL[((self.cure_CL + cure_rate_CL*self.dt_C)>1.0)] = (0.999999999 - self.cure_CL[((self.cure_CL + cure_rate_CL*self.dt_C)>1.0)])/self.dt_C
                self.cure_CL = self.cure_CL + cure_rate_CL*self.dt_C
                
                # Calculate temperature rate
                temp_rate_CL = (self.kappa/(self.rho*self.cp))*laplacian_CL + (self.hr_CL*cure_rate_CL)/self.cp
            else:
                temp_rate_CL = (self.kappa/(self.rho*self.cp))*laplacian_CL
                
            # Step the temperature field via the Forward Euler method
            self.temp_CL = self.temp_CL + temp_rate_CL*self.dt_C
            
        # Step the right coarse to the next time point
        if self.num_x_CR > 0:
            
            # Get the boundaries temperatures from environment and fine temperature field
            self.temp_CR[:,0,:] = ndimage.zoom(self.temp_F[:,-1,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
            padded_temp_CR,qy_tot,qx_tot,qz_tot = self.get_boundaries(self.temp_CR,self.dy_C,self.dx_C,self.dz_C,order=self.order_C,k=self.kappa,c=self.htc,bc=self.amb_T,qz=[0.0,heat_CR])
            self.qz_tot_CR = copy.deepcopy(qz_tot[1])
            
            # Enforce Neumann boundaries between fine and coarse faces
            if self.order_C[1] == 1:
                zoomed_temp_F_2 = ndimage.zoom(self.temp_F[:,-2,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_left_C = (self.dx_F*self.temp_CR[:,1,:]+self.dx_C*zoomed_temp_F_2)/(self.dx_F+self.dx_C)
            if self.order_C[1] == 2:
                zoomed_temp_F_2 = ndimage.zoom(self.temp_F[:,-2,:], zoom=[self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_F_3 = ndimage.zoom(self.temp_F[:,-3,:], zoom=[self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_left_C = (self.dx_C*((1./12.)*zoomed_temp_F_3+(-2./3.)*zoomed_temp_F_2)+self.dx_F*((1./12.)*self.temp_CR[:,2,:]+(-2./3.)*self.temp_CR[:,1,:]))/(((1./12.)-(2./3.))*(self.dx_C+self.dx_F))
            if self.order_C[1] >= 3:
                zoomed_temp_F_2 = ndimage.zoom(self.temp_F[:,-2,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_F_3 = ndimage.zoom(self.temp_F[:,-3,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_F_4 = ndimage.zoom(self.temp_F[:,-4,:], zoom=[1./self.num_y_ratio, 1./self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_left_C = (self.dx_C*((-1./60.)*zoomed_temp_F_4+(3./20.)*zoomed_temp_F_3+(-3./4.)*zoomed_temp_F_2)+self.dx_F*((-1./60.)*self.temp_CR[:,3,:]+(3./20.)*self.temp_CR[:,2,:]+(-3./4.)*self.temp_CR[:,1,:]))/(((-1./60.)+(3./20.)+(-3./4.))*(self.dx_C+self.dx_F))
            for i in range(self.order_C[1]):
                padded_temp_CR[self.order_C[0]:-self.order_C[0],i,self.order_C[2]:-self.order_C[2]] = virtual_temps_left_C
    
            # Get the laplacian of the coarse right temperature field
            laplacian_CR = (self.laplacian_CR@padded_temp_CR.flatten()).reshape(self.num_y_CR+2*self.order_C[0],self.num_x_CR+2*self.order_C[1],self.num_z_CR+2*self.order_C[2])[self.order_C[0]:-self.order_C[0],self.order_C[1]:-self.order_C[1],self.order_C[2]:-self.order_C[2]]
    
            # Calculate the FE cure rates for the coarse right field
            f = self.arrhenius[np.rint((self.temp_CR-self.T_start)/(self.T_step)).astype(int)]
            g = self.kinetics[np.rint((self.cure_CR-self.a_start)/(self.a_step)).astype(int)]
            cure_rate_CR = f*g
            
            # Limit the FE cure rate so that the max cure value is exactly 0.999999999
            cure_rate_CR[((self.cure_CR + cure_rate_CR*self.dt_C)>1.0)] = (0.999999999 - self.cure_CR[((self.cure_CR + cure_rate_CR*self.dt_C)>1.0)])/self.dt_C
            self.cure_CR = self.cure_CR + cure_rate_CR*self.dt_C
            
            # Step the temperature field via the Forward Euler method
            temp_rate_CR = (self.kappa/(self.rho*self.cp))*laplacian_CR + (self.hr_CR*cure_rate_CR)/self.cp
            self.temp_CR = self.temp_CR + temp_rate_CR*self.dt_C
        
        # Step the fine field the rest of the way to the next time point
        while self.times_F[self.time_ind_F] < next_time:
            self.step_fine_field(heat=heat_F)
            if self.time_ind_F + 1 < len(self.times_F):
                self.time_ind_F = self.time_ind_F + 1
            else:
                return True
        
        # Save image
        if self.save_frames and (self.times_C[self.time_ind_C] in self.frames):
            self.save_frame()
        
        # Determine if the fine field must be slid
        if self.front[-1][1] >= self.x_grid_F[0,int(np.floor(0.80*self.x_grid_F.shape[1])),0]:
            if self.num_x_CL == 0:
                while self.num_x_CL < 4:
                    self.slide_fine()
            elif self.num_x_CR <= 4:
                while self.num_x_CR > 0:
                    self.slide_fine()
            elif self.num_x_CR > 4:
                self.slide_fine()
        
        # When done stepping all fields return done as false
        return False
        

    def step_fine_field(self,heat=0.0):
        """
        The the fine domain forward by one fine time step.
        
        Parameters
        ----------
        heat : float or array-like, shape (num_y_F,num_x_F)
            Externally applied heat (positive in) to the z+ surface of the fine grid. The default is 0.0.
            
        Returns
        -------
        None.

        """
        
        # Get the trigger heat
        qy,qx,qz=self.get_trigger()
        qz[1] = qz[1] + heat
        
        # Get the fine boundaries with no pad on x- face if the coarse left grid is initialized and no pad on the x+ face if the coarse right grid still has some values
        padded_temp_F, qy_tot, qx_tot, qz_tot = self.get_boundaries(self.temp_F,self.dy_F,self.dx_F,self.dz_F,order=self.order_F,k=self.kappa,c=self.htc,bc=self.amb_T,qy=qy,qx=qx,qz=qz)
        self.qz_tot_F = copy.deepcopy(qz_tot[1])
        
        # Enforce Neumann boundaries between fine and coarse faces if the left coarse grid exists
        if self.num_x_CL > 0:
            if self.order_F[1] == 1:
                zoomed_temp_CL_2 = ndimage.zoom(self.temp_CL[:,-2,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_left_F = (self.dx_C*self.temp_F[:,1,:]+self.dx_F*zoomed_temp_CL_2)/(self.dx_F+self.dx_C)
            if self.order_F[1] == 2:
                zoomed_temp_CL_2 = ndimage.zoom(self.temp_CL[:,-2,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_CL_3 = ndimage.zoom(self.temp_CL[:,-3,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_left_F = (self.dx_F*((1./12.)*zoomed_temp_CL_3+(-2./3.)*zoomed_temp_CL_2)+self.dx_C*((1./12.)*self.temp_F[:,2,:]+(-2./3.)*self.temp_F[:,1,:]))/(((1./12.)-(2./3.))*(self.dx_C+self.dx_F))
            if self.order_F[1] >= 3:
                zoomed_temp_CL_2 = ndimage.zoom(self.temp_CL[:,-2,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_CL_3 = ndimage.zoom(self.temp_CL[:,-3,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_CL_4 = ndimage.zoom(self.temp_CL[:,-4,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_left_F = (self.dx_F*((-1./60.)*zoomed_temp_CL_4+(3./20.)*zoomed_temp_CL_3+(-3./4.)*zoomed_temp_CL_2)+self.dx_C*((-1./60.)*self.temp_F[:,3,:]+(3./20.)*self.temp_F[:,2,:]+(-3./4.)*self.temp_F[:,1,:]))/(((-1./60.)+(3./20.)+(-3./4.))*(self.dx_C+self.dx_F))
            for i in range(self.order_F[1]):
                padded_temp_F[self.order_F[0]:-self.order_F[0],i,self.order_F[2]:-self.order_F[2]] = virtual_temps_left_F
        
        # Enforce Neumann boundaries between fine and coarse faces if the right coarse grid exists
        if self.num_x_CR > 0:
            if self.order_F[1] == 1:
                zoomed_temp_CR_1 = ndimage.zoom(self.temp_CR[:,1,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_right_F = (self.dx_F*zoomed_temp_CR_1+self.dx_C*self.temp_F[:,-2,:])/(self.dx_C+self.dx_F)
            if self.order_F[1] == 2:
                zoomed_temp_CR_1 = ndimage.zoom(self.temp_CR[:,1,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_CR_2 = ndimage.zoom(self.temp_CR[:,2,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_right_F = (self.dx_C*((1./12.)*self.temp_F[:,-3,:]+(-2./3.)*self.temp_F[:,-2,:])+self.dx_F*((1./12.)*zoomed_temp_CR_2+(-2./3.)*zoomed_temp_CR_1))/(((1./12.)-(2./3.))*(self.dx_C+self.dx_F))
            if self.order_F[1] >= 3:
                zoomed_temp_CR_1 = ndimage.zoom(self.temp_CR[:,1,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_CR_2 = ndimage.zoom(self.temp_CR[:,2,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                zoomed_temp_CR_3 = ndimage.zoom(self.temp_CR[:,3,:], zoom=[self.num_y_ratio, self.num_z_ratio], mode='nearest', order=5)
                virtual_temps_right_F = (self.dx_C*((-1./60.)*self.temp_F[:,-4,:]+(3./20.)*self.temp_F[:,-3,:]+(-3./4.)*self.temp_F[:,-2,:])+self.dx_F*((-1./60.)*zoomed_temp_CR_3+(3./20.)*zoomed_temp_CR_2+(-3./4.)*zoomed_temp_CR_1))/(((-1./60.)+(3./20.)+(-3./4.))*(self.dx_C+self.dx_F))
            for i in range(self.order_F[1]):
                padded_temp_F[self.order_F[0]:-self.order_F[0],-i-1,self.order_F[2]:-self.order_F[2]] = virtual_temps_right_F
        
        # Get the laplacian of the fine temperature field
        laplacian_F = (self.laplacian_F@padded_temp_F.flatten()).reshape(self.num_y_F+2*self.order_F[0],self.num_x_F+2*self.order_F[1],self.num_z_F+2*self.order_F[2])[self.order_F[0]:-self.order_F[0],self.order_F[1]:-self.order_F[1],self.order_F[2]:-self.order_F[2]]
        
        # Calculate the FE cure rates for the fine field
        try:
            f = self.arrhenius[np.rint((self.temp_F-self.T_start)/(self.T_step)).astype(int)]
        except:
            print("WARNING: Detected temperature outside of valid range.")
            f = self.arrhenius[-1]
        try:
            g = self.kinetics[np.rint((self.cure_F-self.a_start)/(self.a_step)).astype(int)]
        except:
            print("WARNING: Detected degree of cure outside of valid range.")
            g = self.kinetics[-1]
        cure_rate_F = f*g
        
        # Determine which points are integrated via implicit trapezoid and which points are integrated via FE
        imp_points = cure_rate_F>self.imp_cure_rate
        exp_points = np.invert(imp_points)
        front_points = cure_rate_F>1.5
        
        # Record front location if it has moved in the propogation direction
        if front_points.any() and (abs(np.mean(self.x_grid_F[front_points]) - self.front[-1][1])>0.5*self.dx_F or np.mean(self.x_grid_F[front_points])==0.0):
            self.front.append([np.mean(self.y_grid_F[front_points]),np.mean(self.x_grid_F[front_points]),np.mean(self.z_grid_F[front_points]),self.times_F[self.time_ind_F]]) 
        
        # If there are points that require implicit integration, use implicit integration there
        if (imp_points).any():

            # Get the current conditions of cure regions with high cure rates
            a1 = self.cure_F[imp_points]
            f1 = f[imp_points]
            g1 = g[imp_points]
            m = 0.5*self.dt_F*f1
            b = a1 + m*g1
            min_a2 = copy.copy(a1)
            max_a2 = np.ones(a1.shape)
            
            # Use a binary search to optimize the a2 guess
            for i in range(20):
                a2 = 0.5*(min_a2+max_a2)
                g2 = self.kinetics[np.rint((a2-self.a_start)/(self.a_step)).astype(int)]
                err = b+m*g2-a2
                next_guess_bigger = err>3.*self.a_step
                next_guess_smaller = err<-3.*self.a_step
                if next_guess_bigger.any() or next_guess_smaller.any():
                    if next_guess_bigger.any():
                        min_a2[next_guess_bigger] = a2[next_guess_bigger]
                    if next_guess_smaller.any():
                        max_a2[next_guess_smaller] = a2[next_guess_smaller]
                else:
                    break
                    
            # Update the cure and cure rate for the implicit points
            cure_rate_F[imp_points] = 0.5*f1*(g1+g2)
            self.cure_F[imp_points] = a2
            
            # Update the cure and cure rate for the explicit points
            overshoot_points = (self.cure_F + cure_rate_F*self.dt_F)>1.0
            if overshoot_points.any():
                cure_rate_F[overshoot_points] = (0.999999999 - self.cure_F[overshoot_points])/self.dt_F
            self.cure_F[exp_points] = self.cure_F[exp_points] + cure_rate_F[exp_points]*self.dt_F
        
        # If there are no points that require implicit integration, use explicit integration
        else:
            # Limit the FE cure rate so that the max cure value is exactly 0.999999999
            cure_rate_F[((self.cure_F + cure_rate_F*self.dt_F)>1.0)] = (0.999999999 - self.cure_F[((self.cure_F + cure_rate_F*self.dt_F)>1.0)])/self.dt_F
        
            # Use the FE method to step the cure
            self.cure_F = self.cure_F + cure_rate_F*self.dt_F

        # Step the temperature field via the Forward Euler method
        temp_rate_F = (self.kappa/(self.rho*self.cp))*laplacian_F + (self.hr_F*cure_rate_F)/self.cp
        self.temp_F = self.temp_F + temp_rate_F*self.dt_F
        
    
    def get_trigger(self):
        """
        Calculates the trigger heat based on the front condition. Will apply heat until front is ignited.

        Returns
        -------
        qy : array_like, shape (2,num_x_F,num_z_F)
            The heat applied to the -y face and the +y face.
        qx : array_like, shape (2,num_y_F,num_z_F)
            The heat applied to the -x face and the +x face.
        qz : array_like, shape (2,num_y_F,num_x_F)
            The heat applied to the -z face and the +z face.

        """
        
        qy = np.zeros((2,self.num_x_F,self.num_z_F))
        qx = np.zeros((2,self.num_y_F,self.num_z_F))
        qz = np.zeros((2,self.num_y_F,self.num_x_F))
        if self.front[-1][1]<=5.*np.max([self.dy_F,self.dx_F,self.dz_F]):
            qx[0,:,:]=self.q_trig
        return qy,qx,qz

    
    def get_boundaries(self,field,dy,dx,dz,k=0.152,c=20.0,bc=298.15,qy=[[0.0],[0.0]],qx=[[0.0],[0.0]],qz=[[0.0],[0.0]],order=[1,1,1],cval=True):
        """
        Adds Neumann boundary conditions to a given field.
    
        Parameters
        ----------
        field : array_like, shape (num_y,num_x,num_z)
            3D scalar field to which the boundaries are added.
        dy : float
            Discrete step size of the rectangular field in the y direction (dim=0).
        dx : float
            Discrete step size of the rectangular field in the x direction (dim=1).
        dz : float
            Discrete step size of the rectangular field in the z direction (dim=2).
        k : float, optional
            Conductivity of material in [W/m-K]. Defines Neumann boundary conditions for the heat equation: c(bc-F0)=-k(dF/dn). The default is 0.152.
        c : float, optional
            Heat transfer coefficient of material in [W/m^2-K]. Set to 0.0 for adiabatic boundary conditions.
            Defines Neumann boundary conditions for the heat equation: c(bc-F0)=-k(dF/dn). The default is 20.0.
        bc : float, optional
            Ambient field value in [K]. Defines Neumann boundary conditions for the heat equation: c(bc-F0)=-k(dF/dn). The default is 298.15.
        qy : 3darray, optional
            Heat applied to y boundaries (dim=0) in [W/m^2]. Positive going into material. The first entry in the list is on the y- face, the second entry is on the y+ face.
            Heat values may be scalars applied to entire face or arrays of size field[0,:,:]. The default is [[0.0],[0.0]].
        qx : 3darray, optional
            Heat applied to x boundaries (dim=1) in [W/m^2]. Positive going into material. The first entry in the list is on the x- face, the second entry is on the x+ face.
            Heat values may be scalars applied to entire face or arrays of size field[:,0,:]. The default is [[0.0],[0.0]].
        qz : 3darray, optional
            Heat applied to z boundaries (dim=2) in [W/m^2]. Positive going into material. The first entry in the list is on the z- face, the second entry is on the z+ face.
            Heat values may be scalars applied to entire face or arrays of size field[:,:,0]. The default is [[0.0],[0.0]].
        order : array_like, shape (3,), optional
            Order of boundaries added to field in each dimension. Must be bewtween 1 and 3. The default is [1,1,1].
        cval : bool, optional
            Indicates whether virtual temperatures are constant value
            Face selections may be delimited by any value. The default is ''.
    
        Returns
        -------
        padded_field : array_like, shape (num_y,num_x,num_z)
            Neumann type boundary padded field.
        qy_tot : array_like, shape (2,num_x,num_z)
            Total heat flux on y boundaries.
        qx_tot : array_like, shape (2,num_y,num_z)
            Total heat flux on x boundaries.
        qz_tot : array_like, shape (2,num_y,num_x)
            Total heat flux on z boundaries.
        """
            
        # Extend the field toinclude virtual temperatures on the boundaries
        padded_field = np.zeros((len(field)+2*order[0], len(field[0])+2*order[1], len(field[0][0])+2*order[2]))
        padded_field[order[0]:-order[0],order[1]:-order[1],order[2]:-order[2]] = field
        
        # Calculate all heat fluxes out of domain (positive out)
        q_ym = c*(field[0,:,:] - bc)  - qy[0]
        q_yp = c*(field[-1,:,:] - bc) - qy[1]
        q_xm = c*(field[:,0,:] - bc)  - qx[0]
        q_xp = c*(field[:,-1,:] - bc) - qx[1]
        q_zm = c*(field[:,:,0] - bc)  - qz[0]
        q_zp = c*(field[:,:,-1] - bc) - qz[1]
        
        # Apply Fourier’s law to calculate temperature gradient at each boundary
        dT_dy_ym = (1./k)*q_ym
        dT_dy_yp = (-1./k)*q_yp
        dT_dx_xm = (1./k)*q_xm
        dT_dx_xp = (-1./k)*q_xp
        dT_dz_zm = (1./k)*q_zm
        dT_dz_zp = (-1./k)*q_zp
        
        if cval:
            # Calculate y face boundary temperatures
            if order[0]==1:
                T_ym = (dT_dy_ym*dy - (1./2.)*field[1,:,:]) / (-1./2.)
                T_yp = (dT_dy_yp*dy + (1./2.)*field[-2,:,:]) / (1./2.)
            elif order[0]==2:
                T_ym = (dT_dy_ym*dy - (2./3.)*field[1,:,:] + (1./12.)*field[2,:,:]) / ((1./12.) + (-2./3.))
                T_yp = (dT_dy_yp*dy + (2./3.)*field[-2,:,:] - (1./12.)*field[-3,:,:]) / ((-1./12.) + (2./3.))
            elif order[0]>=3:
                T_ym = (dT_dy_ym*dy - (3./4.)*field[1,:,:] + (3./20.)*field[2,:,:] - (1./60.)*field[3,:,:]) / ((-1./60.) + (3./20.) + (-3./4.))
                T_yp = (dT_dy_yp*dy + (3./4.)*field[-2,:,:] - (3./20.)*field[-3,:,:] + (1./60.)*field[-4,:,:]) / ((1./60.) + (-3./20.) + (3./4.))
            for i in range(order[0]):
                padded_field[i,order[1]:-order[1],order[2]:-order[2]] = T_ym
                padded_field[-i-1,order[1]:-order[1],order[2]:-order[2]] = T_yp
                
            # Calculate x face boundary temperatures
            if order[1]==1:
                T_xm = (dT_dx_xm*dx - (1./2.)*field[:,1,:]) / (-1./2.)
                T_xp = (dT_dx_xp*dx + (1./2.)*field[:,-2,:]) / (1./2.)
            elif order[1]==2:
                T_xm = (dT_dx_xm*dx - (2./3.)*field[:,1,:] + (1./12.)*field[:,2,:]) / ((1./12.) + (-2./3.))
                T_xp = (dT_dx_xp*dx + (2./3.)*field[:,-2,:] - (1./12.)*field[:,-3,:]) / ((-1./12.) + (2./3.))
            elif order[1]>=3:
                T_xm = (dT_dx_xm*dx - (3./4.)*field[:,1,:] + (3./20.)*field[:,2,:] - (1./60.)*field[:,3,:]) / ((-1./60.) + (3./20.) + (-3./4.))
                T_xp = (dT_dx_xp*dx + (3./4.)*field[:,-2,:] - (3./20.)*field[:,-3,:] + (1./60.)*field[:,-4,:]) / ((1./60.) + (-3./20.) + (3./4.))
            for i in range(order[1]):
                padded_field[order[0]:-order[0],i,order[2]:-order[2]] = T_xm
                padded_field[order[0]:-order[0],-i-1,order[2]:-order[2]] = T_xp
            
            # Calculate z face boundary temperatures
            if order[2]==1:
                T_zm = (dT_dz_zm*dz - (1./2.)*field[:,:,1]) / (-1./2.)
                T_zp = (dT_dz_zp*dz + (1./2.)*field[:,:,-2]) / (1./2.)
            elif order[2]==2:
                T_zm = (dT_dz_zm*dz - (2./3.)*field[:,:,1] + (1./12.)*field[:,:,2]) / ((1./12.) + (-2./3.))
                T_zp = (dT_dz_zp*dz + (2./3.)*field[:,:,-2] - (1./12.)*field[:,:,-3]) / ((-1./12.) + (2./3.))
            elif order[2]>=3:
                T_zm = (dT_dz_zm*dz - (3./4.)*field[:,:,1] + (3./20.)*field[:,:,2] - (1./60.)*field[:,:,3]) / ((-1./60.) + (3./20.) + (-3./4.))
                T_zp = (dT_dz_zp*dz + (3./4.)*field[:,:,-2] - (3./20.)*field[:,:,-3] + (1./60.)*field[:,:,-4]) / ((1./60.) + (-3./20.) + (3./4.))
            for i in range(order[2]):
                padded_field[order[0]:-order[0],order[1]:-order[1],i] = T_zm
                padded_field[order[0]:-order[0],order[1]:-order[1],-i-1] = T_zp
            
        else:
            # Calculate 1st order central difference virtual temperatures
            T_ym1 = -2.0*(dT_dy_ym*dy-0.5*field[1,:,:])
            T_yp1 = 2.0*(dT_dy_yp*dy+0.5*field[-2,:,:])
            padded_field[order[0]-1,order[1]:-order[1],order[2]:-order[2]] = T_ym1
            padded_field[-order[0],order[1]:-order[1],order[2]:-order[2]] = T_yp1
            T_xm1 = -2.0*(dT_dx_xm*dx-0.5*field[:,1,:])
            T_xp1 = 2.0*(dT_dx_xp*dx+0.5*field[:,-2,:])
            padded_field[order[0]:-order[0],order[1]-1,order[2]:-order[2]] = T_xm1
            padded_field[order[0]:-order[0],-order[1],order[2]:-order[2]] = T_xp1
            T_zm1 = -2.0*(dT_dz_zm*dz-0.5*field[:,:,1])
            T_zp1 = 2.0*(dT_dz_zp*dz+0.5*field[:,:,-2])
            padded_field[order[0]:-order[0],order[1]:-order[1],order[2]-1] = T_zm1
            padded_field[order[0]:-order[0],order[1]:-order[1],-order[2]] = T_zp1
            
            # Calculate 2nd order central difference virtual temperatures
            if order[0]>=1:
                T_ym2 = 12.*(dT_dy_ym*dy + (2./3.)*T_ym1 - (2./3.)*field[1,:,:] + (1./12.)*field[2,:,:])
                T_yp2 = -12.*(dT_dy_yp*dy - (1./12.)*field[-3,:,:] + (2./3.)*field[-2,:,:] - (2./3.)*T_yp1)
                padded_field[order[0]-2,order[1]:-order[1],order[2]:-order[2]] = T_ym2
                padded_field[-order[0]+1,order[1]:-order[1],order[2]:-order[2]] = T_yp2
            if order[1]>=1:
                T_xm2 = 12.*(dT_dx_xm*dx + (2./3.)*T_xm1 - (2./3.)*field[:,1,:] + (1./12.)*field[:,2,:])
                T_xp2 = -12.*(dT_dx_xp*dx - (1./12.)*field[:,-3,:] + (2./3.)*field[:,-2,:] - (2./3.)*T_xp1)
                padded_field[order[0]:-order[0],order[1]-2,order[2]:-order[2]] = T_xm2
                padded_field[order[0]:-order[0],-order[1]+1,order[2]:-order[2]] = T_xp2
            if order[2]>=1:
                T_zm2 = 12.*(dT_dz_zm*dz + (2./3.)*T_zm1 - (2./3.)*field[:,:,1] + (1./12.)*field[:,:,2])
                T_zp2 = -12.*(dT_dz_zp*dz - (1./12.)*field[:,:,-3] + (2./3.)*field[:,:,-2] - (2./3.)*T_zp1)
                padded_field[order[0]:-order[0],order[1]:-order[1],order[2]-2] = T_zm2
                padded_field[order[0]:-order[0],order[1]:-order[1],-order[2]+1] = T_zp2
            
            # Calculate 3rd order central difference virtual temperatures
            if order[0]>=2:
                T_ym3 = -60.*(dT_dy_ym*dy - (3./20.)*T_ym2 + (3./4.)*T_ym1 - (3./4.)*field[1,:,:] + (3./20.)*field[2,:,:] - (1./60.)*field[3,:,:])
                T_yp3 = 60.*(dT_dy_yp*dy + (1./60.)*field[-4,:,:] - (3./20.)*field[-3,:,:] + (3./4.)*field[-2,:,:] - (3./4.)*T_yp1 + (3./20.)*T_yp2)
                padded_field[order[0]-3,order[1]:-order[1],order[2]:-order[2]] = T_ym3
                padded_field[-order[0]+2,order[1]:-order[1],order[2]:-order[2]] = T_yp3
            if order[1]>=2:
                T_xm3 = -60.*(dT_dx_xm*dx - (3./20.)*T_xm2 + (3./4.)*T_xm1 - (3./4.)*field[:,1,:] + (3./20.)*field[:,2,:] - (1./60.)*field[:,3,:])
                T_xp3 = 60.*(dT_dx_xp*dx + (1./60.)*field[:,-4,:] - (3./20.)*field[:,-3,:] + (3./4.)*field[:,-2,:] - (3./4.)*T_xp1 + (3./20.)*T_xp2)
                padded_field[order[0]:-order[0],order[1]-3,order[2]:-order[2]] = T_xm3
                padded_field[order[0]:-order[0],-order[1]+2,order[2]:-order[2]] = T_xp3
            if order[2]>=2:
                T_zm3 = -60.*(dT_dz_zm*dz - (3./20.)*T_zm2 + (3./4.)*T_zm1 - (3./4.)*field[:,:,1] + (3./20.)*field[:,:,2] - (1./60.)*field[:,:,3])
                T_zp3 = 60.*(dT_dz_zp*dz + (1./60.)*field[:,:,-4] - (3./20.)*field[:,:,-3] + (3./4.)*field[:,:,-2] - (3./4.)*T_zp1 + (3./20.)*T_zp2)
                padded_field[order[0]:-order[0],order[1]:-order[1],order[2]-3] = T_zm3
                padded_field[order[0]:-order[0],order[1]:-order[1],-order[2]+2] = T_zp3
            
        # Combine heat fluxes
        qy_tot = [q_ym, q_yp]
        qx_tot = [q_xm, q_xp]
        qz_tot = [q_zm, q_zp]
            
        return padded_field, qy_tot, qx_tot, qz_tot


    def save_frame(self):
        """
        Appends the top layer of the current temperature and cure fields to the temperature and cure image buffer, respectively. Saves top layer heat flux.

        Returns
        -------
        None.

        """
        
        # Save temperature, cure, and heat fields
        temps = []
        cures = []
        heats = []
        if self.num_x_CL > 0:
            temps.append(copy.deepcopy(self.temp_CL[:,:,-1]))
            cures.append(copy.deepcopy(self.cure_CL[:,:,-1]))
            heats.append(copy.deepcopy(self.qz_tot_CL))
        temps.append(copy.deepcopy(self.temp_F[:,:,-1]))
        cures.append(copy.deepcopy(self.cure_F[:,:,-1]))
        heats.append(copy.deepcopy(self.qz_tot_F))
        if self.num_x_CR > 0:
            temps.append(copy.deepcopy(self.temp_CR[:,:,-1]))
            cures.append(copy.deepcopy(self.cure_CR[:,:,-1]))
            heats.append(copy.deepcopy(self.qz_tot_CR))
        self.temp_images.append(temps)
        self.cure_images.append(cures)
        self.heat_images.append(heats)        


    def slide_fine(self):
        """
        Slides the fine grid in the positive x direction by one coarse step:
            1. Pop the x- slice of the coarse right temperature, cure, and enthalpy of rxn field
            2. Scale and append the popped x- slice of the coarse right temperature, cure, and enthalpy of rxn field to the x+ side of the fine fields
            3. Pop sufficient x- slices of the fine temperature, cure, and enthalpy of rxn field to be equivalent to one coarse step
            4. Scale and append the popped x- slices of the fine temperature, cure, and enthalpy of rxn field to the x+ side of the coarse left fields
    
        Returns
        -------
        None.
    
        """
        
        # Update number of grid points in x direction for both the left and right coarse grids
        self.num_x_CR = self.num_x_CR - 1
        self.num_x_CL = self.num_x_CL + 1
        
        # Increment the x coordinates of the fine grid by one coarse step
        self.x_grid_F = self.x_grid_F + self.dx_C
        
        # Remove the leftmost slice in the x direction from the coarse right grid
        self.x_grid_CR = self.x_grid_CR[:,1:,:]
        
        # If the coarse left grid has not been initialized, initialize it with x coordinate = 0.0
        if self.num_x_CL == 1:
            y_space_CL = np.linspace(0.,self.y_len,self.num_y_CL)
            x_space_CL = np.array([0.])
            z_space_CL = np.linspace(0.,self.z_len,self.num_z_CL)
            self.x_grid_CL, self.y_grid_CL, self.z_grid_CL = np.meshgrid(x_space_CL, y_space_CL, z_space_CL)
        
        # If the coarse left grid has already been initialized, append one slice in the x direction
        else:
            self.y_grid_CL = np.append(self.y_grid_CL,self.y_grid_CL[:,-1,:].reshape(self.y_grid_CL.shape[0],1,self.y_grid_CL.shape[2]),axis=1)
            self.x_grid_CL = np.append(self.x_grid_CL,(self.x_grid_CL[:,-1,:]+self.dx_C).reshape(self.x_grid_CL.shape[0],1,self.x_grid_CL.shape[2]),axis=1)
            self.z_grid_CL = np.append(self.z_grid_CL,self.z_grid_CL[:,-1,:].reshape(self.z_grid_CL.shape[0],1,self.z_grid_CL.shape[2]),axis=1)
        
        # Update the left and right coarse grid laplacians
        if self.num_x_CL > 0:
            self.laplacian_CL = self.build_laplacian_mat(self.num_y_CL,self.num_x_CL,self.num_z_CL,self.dy_C,self.dx_C,self.dz_C,order=self.order_C)
        if self.num_x_CR > 0:
            self.laplacian_CR = self.build_laplacian_mat(self.num_y_CR,self.num_x_CR,self.num_z_CR,self.dy_C,self.dx_C,self.dz_C,order=self.order_C)
        
        # Pop the leftmost slice of temperature and cure from the right coarse field
        popped_temp_CR = self.temp_CR[:,0,:].reshape(self.temp_CR.shape[0],1,self.temp_CR.shape[2])
        popped_cure_CR = self.cure_CR[:,0,:].reshape(self.cure_CR.shape[0],1,self.cure_CR.shape[2])
        popped_hr_CR = self.hr_CR[:,0,:].reshape(self.hr_CR.shape[0],1,self.hr_CR.shape[2])
        self.temp_CR = self.temp_CR[:,1:,:]
        self.cure_CR = self.cure_CR[:,1:,:]
        self.hr_CR = self.hr_CR[:,1:,:]
        
        # Append the coarse right popped temperature and cure slices to the fine fields
        self.temp_F = np.append(self.temp_F, ndimage.zoom(popped_temp_CR,zoom=[self.num_y_ratio,self.num_x_ratio,self.num_z_ratio], mode='nearest', order=5), axis=1)
        self.cure_F = np.append(self.cure_F, ndimage.zoom(popped_cure_CR,zoom=[self.num_y_ratio,self.num_x_ratio,self.num_z_ratio], mode='nearest', order=5), axis=1)
        self.hr_F = np.append(self.hr_F, ndimage.zoom(popped_hr_CR,zoom=[self.num_y_ratio,self.num_x_ratio,self.num_z_ratio], mode='nearest', order=5), axis=1)
        
        # Pop the leftmost slices of temperature and cure from the fine field
        num_2_pop = self.temp_F.shape[1]-self.x_grid_F.shape[1]
        popped_temp_F = self.temp_F[:,0:num_2_pop,:].reshape(self.temp_F.shape[0],num_2_pop,self.temp_F.shape[2])
        popped_cure_F = self.cure_F[:,0:num_2_pop,:].reshape(self.cure_F.shape[0],num_2_pop,self.cure_F.shape[2])
        popped_hr_F = self.hr_F[:,0:num_2_pop,:].reshape(self.hr_F.shape[0],num_2_pop,self.hr_F.shape[2])
        self.temp_F = self.temp_F[:,num_2_pop:,:]
        self.cure_F = self.cure_F[:,num_2_pop:,:]
        self.hr_F = self.hr_F[:,num_2_pop:,:]
        
        # If the coarse left grid has not been initialized, initialize the temperature, cure, and enthalpy fields with the fine popped slices
        if self.num_x_CL == 1:
            self.temp_CL = ndimage.zoom(popped_temp_F,zoom=[1./self.num_y_ratio,1./self.num_x_ratio,1./self.num_z_ratio], mode='nearest', order=5)
            if self.cure_post:
                self.cure_CL = ndimage.zoom(popped_cure_F,zoom=[1./self.num_y_ratio,1./self.num_x_ratio,1./self.num_z_ratio], mode='nearest', order=5)
            else:
                self.cure_CL = popped_cure_F
            self.hr_CL = ndimage.zoom(popped_hr_F,zoom=[1./self.num_y_ratio,1./self.num_x_ratio,1./self.num_z_ratio], mode='nearest', order=5)
        
        # If the coarse left grid has already been initialized, append the fine popped temperature and cure slices to the coarse left fields
        else:
            self.temp_CL = np.append(self.temp_CL, ndimage.zoom(popped_temp_F,zoom=[1./self.num_y_ratio,1./self.num_x_ratio,1./self.num_z_ratio], mode='nearest', order=5), axis=1)
            if self.cure_post:
                self.cure_CL = np.append(self.cure_CL, ndimage.zoom(popped_cure_F,zoom=[1./self.num_y_ratio,1./self.num_x_ratio,1./self.num_z_ratio], mode='nearest', order=5), axis=1)
            else:
                self.cure_CL = np.append(self.cure_CL, popped_cure_F, axis=1)
            self.hr_CL = np.append(self.hr_CL, ndimage.zoom(popped_hr_F,zoom=[1./self.num_y_ratio,1./self.num_x_ratio,1./self.num_z_ratio], mode='nearest', order=5), axis=1)

                         
    def process_front(self):
        """
        Processes front location and time data to generate front speed as a function of time, and estimate steady state speed of front.

        Returns
        -------
        interp_x_loc : array_like, shape (n,)
            The mean x location (mm), of the front as a function of time interpolated from data generated during simulation.
        interp_time : array_like, shape (n,)
            The interpolated time points (seconds).
        interp_speed : array_like, shape (n,)
            The speed of the front (mm/s) as a function of time interpolated from data generated during simulation.
        ss_speed : float
            The estimated steady state front speed (mm/s).
        ss_std : float
            The standard deviation of the front speed from the steady state front speed (mm/s).

        """
        
        # Interpolate the front location
        self.front = np.array(self.front)
        if len(np.diff(self.front[2:,3]))==0:
            mean_dt = 0.5/self.fr
        else:
            mean_dt = np.mean(np.diff(self.front[2:,3]))
        interp_time, interp_dt = np.linspace(self.front[0,3],self.front[-1,3],int(np.rint((self.front[-1,3]-self.front[0,3])*(1./mean_dt))),retstep=True)
        interp_x_loc = 1000.0*np.interp(interp_time,self.front[:,3],self.front[:,1])
        
        # Use the finite difference method to estimate front speed
        fd_coeff_1 = np.array([1/2,0,-1/2])
        fd_coeff_2 = np.array([-1/12,2/3,0,-2/3,1/12])
        fd_coeff_3 = np.array([1/60,-3/20,3/4,0,-3/4,3/20,-1/60])
        fd_coeff_4 = np.array([-1/280,4/105,-1/5,4/5,0,-4/5,1/5,-4/105,1/280])
        interp_speed = np.zeros(interp_time.shape)
        interp_speed[0]=(interp_x_loc[1]-interp_x_loc[0])/interp_dt
        interp_speed[1]=(fd_coeff_1[0]*interp_x_loc[0]+fd_coeff_1[2]*interp_x_loc[2])/-interp_dt
        interp_speed[2]=(fd_coeff_2[0]*interp_x_loc[0]+fd_coeff_2[1]*interp_x_loc[1]+fd_coeff_2[3]*interp_x_loc[3]+fd_coeff_2[4]*interp_x_loc[4])/-interp_dt
        interp_speed[3]=(fd_coeff_3[0]*interp_x_loc[0]+fd_coeff_3[1]*interp_x_loc[1]+fd_coeff_3[2]*interp_x_loc[2]+fd_coeff_3[4]*interp_x_loc[4]+fd_coeff_3[5]*interp_x_loc[5]+fd_coeff_3[6]*interp_x_loc[6])/-interp_dt
        interp_speed[4:-4]=np.convolve(interp_x_loc,fd_coeff_4, mode='valid')/interp_dt
        interp_speed[-4]=(fd_coeff_3[0]*interp_x_loc[-7]+fd_coeff_3[1]*interp_x_loc[-6]+fd_coeff_3[2]*interp_x_loc[-5]+fd_coeff_3[4]*interp_x_loc[-3]+fd_coeff_3[5]*interp_x_loc[-2]+fd_coeff_3[6]*interp_x_loc[-1])/-interp_dt
        interp_speed[-3]=(fd_coeff_2[0]*interp_x_loc[-5]+fd_coeff_2[1]*interp_x_loc[-4]+fd_coeff_2[3]*interp_x_loc[-2]+fd_coeff_2[4]*interp_x_loc[-1])/-interp_dt
        interp_speed[-2]=(fd_coeff_1[0]*interp_x_loc[-3]+fd_coeff_1[2]*interp_x_loc[-1])/-interp_dt
        interp_speed[-1]=(interp_x_loc[-1]-interp_x_loc[-2])/interp_dt
        
        # Smooth data with Savitzky-Golay filter
        interp_speed = signal.savgol_filter(interp_speed,window_length=int(np.rint(1.0/interp_dt)),polyorder=4)
        
        # Estimate steady state speed
        if (abs(np.diff(interp_speed))>1.0e-5).any():
            speed_after_ignition = interp_speed[np.arange(len(interp_speed)-1)[abs(np.diff(interp_speed))>1.0e-5][0]:]
            ss_speed = np.mean(speed_after_ignition[np.logical_and(speed_after_ignition>(np.median(speed_after_ignition)-np.std(speed_after_ignition)),speed_after_ignition<(np.median(speed_after_ignition)+np.std(speed_after_ignition)))])
            ss_std = np.std(speed_after_ignition[np.logical_and(speed_after_ignition>(np.median(speed_after_ignition)-np.std(speed_after_ignition)),speed_after_ignition<(np.median(speed_after_ignition)+np.std(speed_after_ignition)))])
        else:
            ss_speed = 0.0
            ss_std = 0.0
        
        # Return data
        return interp_x_loc, interp_time, interp_speed, ss_speed, ss_std
    
    
    def plot(self, x_loc, time, speed, ss_speed, ss_std, name=""):
        """
        Plots simulation results.

        Parameters
        ----------
        x_loc : array_like, shape (n,)
            The mean x location (mm), of the front as a function of time interpolated from data generated during simulation.
        time : array_like, shape (n,)
            The interpolated time points (seconds).
        speed : array_like, shape (n,)
            The speed of the front (mm/s) as a function of time interpolated from data generated during simulation.
        ss_speed : float
            The estimated steady state front speed (mm/s).
        ss_std : float
            The standard deviation of the front speed from the steady state front speed (mm/s).
        name : str, optional
            The prepended file name for plots. The default is "".

        Returns
        -------
        None.

        """

        #Check that folder exists
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        # Plot front speed
        plt.clf()
        plt.gcf().set_size_inches(8, 5)
        plt.axhline(ss_speed,c='k',ls='--',label='Steady state\nspeed = '+str(np.round(ss_speed,2))+' mm/s',lw=1.5)
        plt.fill_between(time,ss_speed+ss_std,ss_speed-ss_std,color='k',alpha=0.15,lw=0.)
        plt.plot(time,speed,c='r',lw=2.,label='Front speed')
        plt.xlabel("Time [Seconds]",fontsize=16)
        plt.ylabel("Speed [mm/s]",fontsize=16)
        plt.title("Front Propogation Speed",fontsize=20)
        min_time = np.min(time)
        max_time = np.max(time)
        if min_time != max_time:
            plt.xlim(min_time,max_time)
        else:
            plt.xlim(0.0,self.tot_t)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        handles, labels = plt.gca().get_legend_handles_labels()
        order=[1,0]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=14)
        plt.savefig(self.path + "/" + name + "_speed.svg", dpi = 500)
        plt.close()
        
        # Plot front position
        plt.clf()
        plt.gcf().set_size_inches(8, 5)
        plt.plot(time,x_loc,c='r',lw=2.)
        plt.xlabel("Time [Seconds]",fontsize=16)
        plt.ylabel("Position [mm]",fontsize=16)
        plt.title("Front Position",fontsize=20)
        min_time = np.min(time)
        max_time = np.max(time)
        if min_time != max_time:
            plt.xlim(min_time,max_time)
        else:
            plt.xlim(0.0,self.tot_t)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(self.path + "/" + name + "_position.svg", dpi = 500)
        plt.close()
        
        
    def render(self,name):
        """
        Renders the temperature and cure field images gathered during a simulation.

        Parameters
        ----------
        name : string
            Name of rendered images.

        Returns
        -------
        None.

        """
        
        #Check that folder exists
        if not os.path.exists(self.path+'/render'):
            os.mkdir(self.path+'/render')
        
        # Determine temperature range
        adb_temp_rxn = self.mean_T0+self.dev_T0+((self.mean_hr+self.dev_hr)*(1.0-self.mean_a0+self.dev_a0))/(self.cp)
        min_possible_temp = self.mean_T0-self.dev_T0
        max_temp = self.temp_images[0][0][0][0]
        for x in range(len(self.temp_images)):
            for y in range(len(self.temp_images[x])):
                curr_max = np.max(self.temp_images[x][y])
                if curr_max > max_temp:
                    max_temp = curr_max
        max_normalized = 0.1*((((max_temp - min_possible_temp)/(adb_temp_rxn - min_possible_temp))//0.1) + 1)
        
        # Determine heat flux range
        if self.htc > 0.0:
            max_heat = 100.*((self.htc*(adb_temp_rxn - self.amb_T))//100. + 1)
            min_heat = -1.*max_heat
        else:
            max_heat = 4000.
            min_heat = 0.
        
        # Plot temperature and cure frames
        for i in range(len(self.temp_images)):
            
            if len(self.temp_images[i])==2:
                if len(self.temp_images[i][0]) < len(self.temp_images[i][1]):
                    temp_CL = ndimage.zoom(self.temp_images[i][0], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                    temp_F = self.temp_images[i][1]
                    temp = np.append(temp_CL, temp_F, axis=1)
                    
                    heat_CL = ndimage.zoom(self.heat_images[i][0], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                    heat_F = self.heat_images[i][1]
                    heat = np.append(heat_CL, heat_F, axis=1)
                    
                    if self.cure_post:
                        cure_CL = ndimage.zoom(self.cure_images[i][0], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                    else:
                        cure_CL = self.cure_images[i][0]
                    cure_F = self.cure_images[i][1]
                    cure = np.append(cure_CL, cure_F, axis=1)
                else:
                    temp_F = self.temp_images[i][0]
                    temp_CR = ndimage.zoom(self.temp_images[i][1], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                    temp = np.append(temp_F, temp_CR, axis=1)
                    
                    heat_F = self.heat_images[i][0]
                    heat_CR = ndimage.zoom(self.heat_images[i][1], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                    heat = np.append(heat_F, heat_CR, axis=1)
                    
                    cure_F = self.cure_images[i][0]
                    cure_CR = ndimage.zoom(self.cure_images[i][1], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                    cure = np.append(cure_F, cure_CR, axis=1)
            else:
                temp_CL = ndimage.zoom(self.temp_images[i][0], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                temp_F = self.temp_images[i][1]
                temp_CR = ndimage.zoom(self.temp_images[i][2], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                temp = np.append(temp_CL, temp_F, axis=1)
                temp = np.append(temp, temp_CR, axis=1)
                
                heat_CL = ndimage.zoom(self.heat_images[i][0], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                heat_F = self.heat_images[i][1]
                heat_CR = ndimage.zoom(self.heat_images[i][2], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                heat = np.append(heat_CL, heat_F, axis=1)
                heat = np.append(heat, heat_CR, axis=1)
                
                if self.cure_post:
                    cure_CL = ndimage.zoom(self.cure_images[i][0], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                else:
                    cure_CL =self.cure_images[i][0]
                cure_F = self.cure_images[i][1]
                cure_CR = ndimage.zoom(self.cure_images[i][2], zoom=[self.num_y_ratio, self.num_x_ratio], mode='nearest', order=5)
                cure = np.append(cure_CL, cure_F, axis=1)
                cure = np.append(cure, cure_CR, axis=1)
            
            # Update the maximum cumulative temperature
            if i==0:
                max_cum_temp = temp
            else:
                if temp.shape!=max_cum_temp.shape:
                    max_cum_temp = ndimage.zoom(max_cum_temp, zoom=[temp.shape[0]/max_cum_temp.shape[0], temp.shape[1]/max_cum_temp.shape[1]], mode='nearest', order=5)
                max_cum_temp[temp>max_cum_temp]=temp[temp>max_cum_temp]
            
            # Build the figure
            plt.cla()
            plt.clf()
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
            fig.set_size_inches(10,5)
             
            # Build the grids
            y_space = np.linspace(0,self.y_len,temp.shape[0])
            x_space = np.linspace(0,self.x_len,temp.shape[1])
            x_grid, y_grid = np.meshgrid(x_space, y_space)
            
            # Plot heat flux
            c0 = ax0.pcolormesh(1000.0*y_grid, 1000.0*x_grid, -1.0*heat, shading='gouraud', cmap='jet', vmin=min_heat, vmax=max_heat)
            cbar0 = fig.colorbar(c0, ax=ax0)
            cbar0.set_label("[W/m^2]",labelpad=10,fontsize=12)
            cbar0.ax.tick_params(labelsize=12)
            ax0.set_xlabel('[mm]',fontsize=12)
            ax0.set_ylabel('[mm]',fontsize=12)
            ax0.tick_params(axis='x',labelsize=12)
            ax0.tick_params(axis='y',labelsize=12)
            ax0.set_aspect('equal', adjustable='box')
            ax0.set_title('Heat Flux In',pad=15,fontsize=14)
            
            # Plot maximum cumulative temperature
            c1 = ax1.pcolormesh(1000.0*y_grid, 1000.0*x_grid, (max_cum_temp-min_possible_temp)/(adb_temp_rxn-min_possible_temp), shading='gouraud', cmap='jet', vmin=0.0, vmax=max_normalized)
            cbar1 = fig.colorbar(c1, ax=ax1)
            cbar1.set_label("[Normalized Temperature]",labelpad=10,fontsize=12)
            cbar1.ax.tick_params(labelsize=12)
            ax1.set_xlabel('[mm]',fontsize=12)
            ax1.set_ylabel('[mm]',fontsize=12)
            ax1.tick_params(axis='x',labelsize=12)
            ax1.tick_params(axis='y',labelsize=12)
            ax1.set_aspect('equal', adjustable='box')
            ax1.set_title('Max. Temperature',pad=15,fontsize=14)
            
            # Plot temperature
            c2 = ax2.pcolormesh(1000.0*y_grid, 1000.0*x_grid, temp, shading='gouraud', cmap='jet', vmin=10.*(min_possible_temp//10.), vmax=10.*((adb_temp_rxn//10.) + 1))
            cbar2 = fig.colorbar(c2, ax=ax2)
            cbar2.set_label("[Kelvin]",labelpad=10,fontsize=12)
            cbar2.ax.tick_params(labelsize=12)
            ax2.set_xlabel('[mm]',fontsize=12)
            ax2.set_ylabel('[mm]',fontsize=12)
            ax2.tick_params(axis='x',labelsize=12)
            ax2.tick_params(axis='y',labelsize=12)
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_title('Temperature',pad=15,fontsize=14)
            
            # Build the grids
            y_space = np.linspace(0,self.y_len,cure.shape[0])
            x_space = np.linspace(0,self.x_len,cure.shape[1])
            x_grid, y_grid = np.meshgrid(x_space, y_space)
            
            # Plot cure
            c3 = ax3.pcolormesh(1000.0*y_grid, 1000.0*x_grid, cure, shading='gouraud', cmap='YlOrBr', vmin=0.0, vmax=1.0)
            cbar3 = fig.colorbar(c3, ax=ax3)
            cbar3.set_label("[Conversion]",labelpad=10,fontsize=12)
            cbar3.ax.tick_params(labelsize=12)
            ax3.set_xlabel('[mm]',fontsize=12)
            ax3.set_ylabel('[mm]',fontsize=12)
            ax3.tick_params(axis='x',labelsize=12)
            ax3.tick_params(axis='y',labelsize=12)
            ax3.set_aspect('equal', adjustable='box')
            ax3.set_title('Degree Cure',pad=15,fontsize=14)
            
            # Save the figure
            plt.suptitle('Time since trigger: {:.2f}'.format(self.frames[i]),fontsize=16,family='monospace')
            plt.tight_layout()
            plt.savefig(self.path+'/render/'+name+"_"+str(i).zfill(4)+'.png', dpi=100)
            plt.close()