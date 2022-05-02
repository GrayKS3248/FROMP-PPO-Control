# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:34:32 2022

@author: Grayson Schaer
"""

import numpy as np
import scipy.ndimage as nd
import opensimplex
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import time
from scipy import stats
from scipy import sparse
import networkx as nx


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def init_rect_field(num_y,num_x,num_z,mean,max_dev=0.0,feat_size=0.25,ar=[1.,1.,1.],seed=10000):
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
    ar : 3darray, optional
        Aspect ratio of the y dimension, x dimension, and z dimension, respectively. 
        Aspect ratio of the ith dimension is defined as (length_i / max_length_xyz). The default is [1.,1.,1.].
    seed : int, optional
        Seed used to generate simplex noise. The default is 10000.

    Returns
    -------
    field : 3darray
        The rectangular scalar field with specified dimensions, mean, and noise.

    """
    
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

    # Scale field so that mean and max deviation are satisfied
    field = field - np.mean(field)             ## Set mean to 0
    field = field * (1.0/np.max(abs(field)))   ## Set max deviation magnitude to 1
    field = field * max_dev + mean             ## Set max deviation magnitude to max_dev, set mean to mean
    return field


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def get_laplace_kernel(dy,dx,dz,order=[1,1,1]):
    """
    Calculates the 3D convolutional laplacian kernel.

    Parameters
    ----------
    dy : float
        Discrete step size of the rectangular field in the y direction (dim=0).
    dx : float
        Discrete step size of the rectangular field in the x direction (dim=1).
    dz : float
        Discrete step size of the rectangular field in the z direction (dim=2).
    order : 1darray with 3 entries, optional
        The discrete laplacian order for each dimension. Minimum of 1. Maximum of 3. The default is [1,1,1].

    Returns
    -------
    kernel : 3darray
        3D convolutional laplacian kernel with dimensions max(order) X max(order) X max(order).

    """
    
    # Set the second derivatice finite difference coefficients
    second_cd_coeffs = [[1.,-2.,1.], [-1./12.,4./3.,-5./2.,4./3.,-1./12.], [1./90.,-3./20.,3./2.,-49./18.,3./2.,-3./20.,1./90.]]
    
    # Calculate the laplacian kernel
    kernel = np.zeros((2*max(order)+1,2*max(order)+1,2*max(order)+1))
    min_ind = ((2*max(order)+1)//2)-((2*order[0]+1)//2)
    max_ind = ((2*max(order)+1)//2)+((2*order[0]+1)//2)+1
    x_ind = (2*max(order)+1)//2
    z_ind = (2*max(order)+1)//2
    kernel[min_ind:max_ind,x_ind,z_ind]=kernel[min_ind:max_ind,x_ind,z_ind]+np.array(second_cd_coeffs[order[0]-1])/(dy*dy)
    y_ind = (2*max(order)+1)//2
    min_ind = ((2*max(order)+1)//2)-((2*order[1]+1)//2)
    max_ind = ((2*max(order)+1)//2)+((2*order[1]+1)//2)+1
    z_ind = (2*max(order)+1)//2
    kernel[y_ind,min_ind:max_ind,z_ind]=kernel[y_ind,min_ind:max_ind,z_ind]+np.array(second_cd_coeffs[order[1]-1])/(dx*dx)
    y_ind = (2*max(order)+1)//2
    x_ind = (2*max(order)+1)//2
    min_ind = ((2*max(order)+1)//2)-((2*order[2]+1)//2)
    max_ind = ((2*max(order)+1)//2)+((2*order[2]+1)//2)+1
    kernel[y_ind,x_ind,min_ind:max_ind]=kernel[y_ind,x_ind,min_ind:max_ind]+np.array(second_cd_coeffs[order[2]-1])/(dz*dz)
    return kernel


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def get_laplace(field,kernel,dy,dx,dz,k=0.152,c=20.0,bc=298.15,qy=[[0.0],[0.0]],qx=[[0.0],[0.0]],qz=[[0.0],[0.0]],no_pad=''):
    """
    Calculates the laplacian of a rectangular field given a set of Neumann boundary conditions.

    Parameters
    ----------
    field : 3darray
        3D scalar field over which the laplacian is calculated.
    kernel : 3darray
        3D convolutional laplacian kernel with dimensions.
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
    no_pad : string, optional
        Indicates which all faces that will not be padded with Neumann type virtual temperatures. Select from '', 'y-', 'y+', 'x-', 'x+', 'z-', 'z+'.
        Face selections may be delimited by any value. The default is ''.

    Returns
    -------
    3darray
        Laplacian of field given boundary conditions. Same dimensions as field.

    """
    
    
    # Determine order
    y_order = np.sum(abs(kernel[:,len(kernel[:,0,:])//2,len(kernel[:,:,0])//2])>1.0e-10)//2
    x_order = np.sum(abs(kernel[len(kernel[0,:,:])//2,:,len(kernel[:,:,0])//2])>1.0e-10)//2
    z_order = np.sum(abs(kernel[len(kernel[0,:,:])//2,len(kernel[:,0,:])//2,:])>1.0e-10)//2
    order=[y_order,x_order,z_order]
        
    # Calculate virtual values to enforce boundary conditions
    padded_field = np.zeros((len(field)+2*order[0], len(field[0])+2*order[1], len(field[0][0])+2*order[2]))
    padded_field[order[0]:-order[0],order[1]:-order[1],order[2]:-order[2]] = field
    
    # y- and y+ boundary
    if 'y-' not in no_pad:
        q_ym = c*(bc-field[0,:,:])+qy[0]
        padded_field[order[0]-1,order[1]:-order[1],order[2]:-order[2]] = (2.*q_ym*dy/k)+field[1,:,:]
    if 'y+' not in no_pad:
        q_yp = c*(bc-field[-1,:,:])+qy[1]
        padded_field[-order[0],order[1]:-order[1],order[2]:-order[2]] = (2.*q_yp*dy/k)+field[-2,:,:]
    if order[0]>1:
        if 'y-' not in no_pad:
            vals = 12.*((-q_ym*dy/k) + (2./3.)*padded_field[order[0]-1,order[1]:-order[1],order[2]:-order[2]] + (-2./3.)*field[1,:,:] + (1./12.)*field[2,:,:])
            padded_field[order[0]-2,order[1]:-order[1],order[2]:-order[2]] = vals
        if 'y+' not in no_pad:
            vals = 12.*((-q_yp*dy/k) + (1./12.)*field[-3,:,:] + (-2./3.)*field[-2,:,:] + (2./3.)*padded_field[-order[0],order[1]:-order[1],order[2]:-order[2]])
            padded_field[-order[0]+1,order[1]:-order[1],order[2]:-order[2]] = vals
    if order[0]>2:
        if 'y-' not in no_pad:
            vals = -60.*((-q_ym*dy/k)-(3./20.)*padded_field[order[0]-2,order[1]:-order[1],order[2]:-order[2]]+(3./4.)*padded_field[order[0]-1,order[1]:-order[1],order[2]:-order[2]]-(3./4.)*field[1,:,:]+(3./20.)*field[2,:,:]-(1./60.)*field[3,:,:])
            padded_field[order[0]-3,order[1]:-order[1],order[2]:-order[2]] = vals
        if 'y+' not in no_pad:
            vals = 60.*((q_yp*dy/k)+(1./60.)*field[-4,:,:]-(3./20.)*field[-3,:,:]+(3./4.)*field[-2,:,:]-(3./4.)*padded_field[-order[0],order[1]:-order[1],order[2]:-order[2]]+(3./20.)*padded_field[-order[0]+1,order[1]:-order[1],order[2]:-order[2]])
            padded_field[-order[0]+2,order[1]:-order[1],order[2]:-order[2]] = vals
    
    # x- and x+ boundary
    if 'x-' not in no_pad:
        q_xm = c*(bc-field[:,0,:])+qx[0]
        padded_field[order[0]:-order[0],order[1]-1,order[2]:-order[2]] = (2.*q_xm*dx/k)+field[:,1,:]
    if 'x+' not in no_pad:
        q_xp = c*(bc-field[:,-1,:])+qx[1]
        padded_field[order[0]:-order[0],-order[1],order[2]:-order[2]] = (2.*q_xp*dx/k)+field[:,-2,:]
    if order[1]>1:
        if 'x-' not in no_pad:
            vals = 12.*((-q_xm*dx/k) + (2./3.)*padded_field[order[0]:-order[0],order[1]-1,order[2]:-order[2]] + (-2./3.)*field[:,1,:] + (1./12.)*field[:,2,:])
            padded_field[order[0]:-order[0],order[1]-2,order[2]:-order[2]] = vals
        if 'x+' not in no_pad:
            vals = 12.*((-q_xp*dx/k) + (1./12.)*field[:,-3,:] + (-2./3.)*field[:,-2,:] + (2./3.)*padded_field[order[0]:-order[0],-order[1],order[2]:-order[2]])
            padded_field[order[0]:-order[0],-order[1]+1,order[2]:-order[2]] = vals
    if order[1]>2:
        if 'x-' not in no_pad:
            vals = -60.*((-q_xm*dx/k)-(3./20.)*padded_field[order[0]:-order[0],order[1]-2,order[2]:-order[2]]+(3./4.)*padded_field[order[0]:-order[0],order[1]-1,order[2]:-order[2]]-(3./4.)*field[:,1,:]+(3./20.)*field[:,2,:]-(1./60.)*field[:,3,:])
            padded_field[order[0]:-order[0],order[1]-3,order[2]:-order[2]] = vals
        if 'x+' not in no_pad:
            vals = 60.*((q_xp*dx/k)+(1./60.)*field[:,-4,:]-(3./20.)*field[:,-3,:]+(3./4.)*field[:,-2,:]-(3./4.)*padded_field[order[0]:-order[0],-order[1],order[2]:-order[2]]+(3./20.)*padded_field[order[0]:-order[0],-order[1]+1,order[2]:-order[2]])
            padded_field[order[0]:-order[0],-order[1]+2,order[2]:-order[2]] = vals
        
    # z- and z+ boundary
    if 'z-' not in no_pad:
        q_zm = c*(bc-field[:,:,0])+qz[0]
        padded_field[order[0]:-order[0],order[1]:-order[1],order[2]-1] = (2.*q_zm*dz/k)+field[:,:,1]
    if 'z+' not in no_pad:
        q_zp = c*(bc-field[:,:,-1])+qz[1]
        padded_field[order[0]:-order[0],order[1]:-order[1],-order[2]] = (2.*q_zp*dz/k)+field[:,:,-2]
    if order[2]>1:
        if 'z-' not in no_pad:
            vals = 12.*((-q_zm*dz/k) + (2./3.)*padded_field[order[0]:-order[0],order[1]:-order[1],order[2]-1] + (-2./3.)*field[:,:,1] + (1./12.)*field[:,:,2])
            padded_field[order[0]:-order[0],order[1]:-order[1],order[2]-2] = vals
        if 'z+' not in no_pad:
            vals = 12.*((-q_zp*dz/k) + (1./12.)*field[:,:,-3] + (-2./3.)*field[:,:,-2] + (2./3.)*padded_field[order[0]:-order[0],order[1]:-order[1],-order[2]])
            padded_field[order[0]:-order[0],order[1]:-order[1],-order[2]+1] = vals
    if order[2]>2:
        if 'z-' not in no_pad:
            vals = -60.*((-q_zm*dz/k)-(3./20.)*padded_field[order[0]:-order[0],order[1]:-order[1],order[2]-2]+(3./4.)*padded_field[order[0]:-order[0],order[1]:-order[1],order[2]-1]-(3./4.)*field[:,:,1]+(3./20.)*field[:,:,2]-(1./60.)*field[:,:,3])
            padded_field[order[0]:-order[0],order[1]:-order[1],order[2]-3] = vals
        if 'z+' not in no_pad:
            vals = 60.*((q_zp*dz/k)+(1./60.)*field[:,:,-4]-(3./20.)*field[:,:,-3]+(3./4.)*field[:,:,-2]-(3./4.)*padded_field[order[0]:-order[0],order[1]:-order[1],-order[2]]+(3./20.)*padded_field[order[0]:-order[0],order[1]:-order[1],-order[2]+1])
            padded_field[order[0]:-order[0],order[1]:-order[1],-order[2]+2] = vals
    
    # Remove the no pad boundaries
    if 'y-' in no_pad:
        padded_field = padded_field[order[0]:,:,:]
    if 'y+' in no_pad:
        padded_field = padded_field[0:-order[0],:,:]
    if 'x-' in no_pad:
        padded_field = padded_field[:,order[1]:,:]
    if 'x+' in no_pad:
        padded_field = padded_field[:,0:-order[1],:]
    if 'z-' in no_pad:
        padded_field = padded_field[:,:,order[2]:]
    if 'z+' in no_pad:
        padded_field = padded_field[:,:,0:-order[2]]
    
    # Calculate the laplcian
    laplacian = nd.convolve(padded_field, kernel)
        
    # Return the VALID laplacian
    return laplacian[order[0]:-order[0],order[1]:-order[1],order[2]:-order[2]]


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def get_err(next_cure, *args):
    """
    Calculates the implicit trapezoid method error given a guess of the next cure state 

    Parameters
    ----------
    next_cure : float
        Guess of next cure state. Must be bounded between [cure, 1.0]
    cure : float
        Current cure state.
    cure_rate : float
        Current time derivative of the cure state.
    f : float
        Current solution to the Arrhenius equation.
    dt : float
        Time step size.

    Raises
    ------
    TypeError
        If required arguments are missing: 'cure', 'cure_rate', 'f', 'dt'.

    Returns
    -------
    err : float
        Implicit trapezoid method error given guess.

    """
    # Check args
    try:
        cure=args[0]
    except:
        raise TypeError("Missing 4 required positional arguments: 'cure', 'cure_rate', 'f', 'dt'")
    try:
        cure_rate=args[1]
    except:
        raise TypeError("Missing 3 required positional arguments: 'cure_rate', 'f', 'dt'")
    try:
        f=args[2]
    except:
        raise TypeError("Missing 2 required positional arguments: 'f', 'dt'")
    try:
        dt=args[3]
    except:
        raise TypeError("Missing 1 required positional argument: 'dt'")
        
    next_cure_rate = f * kinetics[np.round((next_cure-a_start)/(a_step)).astype(int)]
    err = (cure + 0.5*dt*(cure_rate+next_cure_rate))-next_cure
    return err


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
if __name__ == "__main__":
    """
    ########################################################  USER INPUT  ########################################################
    """
    # Define material thermal properties
    k = 0.152    ## Watts / Meter * Kelvin
    cp = 1600.0  ## Joules / Kilogram * Kelvin
    p = 980.0    ## Kilograms / Meter ^ 3
    
    # Define material cure properties
    mean_hr = 350000.0
    dev_hr = 0.01*mean_hr
    A = 8.55e15
    E = 110750.0
    n = 1.72
    m = 0.77
    C = 14.48
    ac = 0.41
    R = 8.314
    
    # Define time parameters
    tot_t = 10.0       ## Seconds
    dt = 0.01          ## Seconds
    fine_dt = 0.00167  ## Seconds
    
    # Define domain size
    y_len = 0.008     ## Meters
    x_len = 0.05      ## Meters
    z_len = 0.001     ## Meters
    fine_len = 0.008  ## Meters
    
    # Define discrete grid step size
    dy = 0.0002        ## Meters
    dx = 0.0002        ## Meters
    dz = 0.0002        ## Meters
    fine_dy = 0.0002  ## Meters
    fine_dx = 0.0002  ## Meters
    fine_dz = 0.0002  ## Meters
    
    # Define initial conditions
    mean_T0 = 305.15
    dev_T0 = 2.0
    mean_a0 = 0.07
    dev_a0 = 0.005
    
    # Define boundary conditions
    htc = 0.0
    amb_T = 298.15
    
    # Define heat inputs
    q_trig = 20000.0  ## Watts / Meter ^ 2
    loc_trig = 'x-'   ## 'y-', 'y+', 'x-', 'x+', 'z-', 'z+'
    
    # Define save options
    fr = 30.0
    
    
    """
    ########################################################  PRE-SIMULATION CALCULATIONS  ########################################################
    """
    print("Set front propogation direction...")
    if loc_trig == 'y-' or loc_trig == 'y+':
        fine_y_len = fine_len
        fine_x_len = x_len
        fine_z_len = z_len
    elif loc_trig == 'x-' or loc_trig == 'x+':
        fine_y_len = y_len
        fine_x_len = fine_len
        fine_z_len = z_len
    elif loc_trig == 'z-' or loc_trig == 'z+':
        fine_y_len = y_len
        fine_x_len = x_len
        fine_z_len = fine_len
    
    print("Calculate number of points in coarse grid...")
    num_y=int(np.rint(y_len/dy+1.))
    num_x=int(np.rint(x_len/dx+1.))
    num_z=int(np.rint(z_len/dz+1.))
    y_len = (num_y-1)*dy
    x_len = (num_x-1)*dx
    z_len = (num_z-1)*dz
    
    print("Ensure fine grid perfectly divides coarse grid...")
    fine_y_len = (np.rint(fine_y_len/dy+1.)-1.)*dy
    fine_x_len = (np.rint(fine_x_len/dx+1.)-1.)*dx
    fine_z_len = (np.rint(fine_z_len/dz+1.)-1.)*dz
    fine_num_y = int(np.rint(fine_y_len/fine_dy+1.))
    fine_num_x = int(np.rint(fine_x_len/fine_dx+1.))
    fine_num_z = int(np.rint(fine_z_len/fine_dz+1.))
    fine_dy = fine_y_len / (fine_num_y - 1)
    fine_dx = fine_x_len / (fine_num_x - 1)
    fine_dz = fine_z_len / (fine_num_z - 1)
    fine_dt = dt/np.ceil(dt/fine_dt)
    
    print("Generate initial fields...")
    max_len = max(y_len,x_len,z_len)
    ar=[y_len/max_len,x_len/max_len,z_len/max_len]
    temp = init_rect_field(num_y,num_x,num_z,mean_T0,max_dev=dev_T0,ar=ar,seed=10000)
    cure = init_rect_field(num_y,num_x,num_z,mean_a0,max_dev=dev_a0,ar=ar,seed=10001)
    hr =   init_rect_field(num_y,num_x,num_z,mean_hr,max_dev=dev_hr,ar=ar,seed=10002)
    
    print("Split fields into fine and coarse...")
    num_y_in_fine_y_len = int(np.rint(fine_y_len/dy+1.))
    num_x_in_fine_x_len = int(np.rint(fine_x_len/dx+1.))
    num_z_in_fine_z_len = int(np.rint(fine_z_len/dz+1.))
    fine_temp = nd.zoom(temp[0:num_y_in_fine_y_len,0:num_x_in_fine_x_len,0:num_z_in_fine_z_len], [fine_num_y/num_y_in_fine_y_len, fine_num_x/num_x_in_fine_x_len, fine_num_z/num_z_in_fine_z_len])
    fine_temp = fine_temp[0:fine_num_y,0:fine_num_x,0:fine_num_z]
    fine_cure = nd.zoom(cure[0:num_y_in_fine_y_len,0:num_x_in_fine_x_len,0:num_z_in_fine_z_len], [fine_num_y/num_y_in_fine_y_len, fine_num_x/num_x_in_fine_x_len, fine_num_z/num_z_in_fine_z_len])
    fine_cure = fine_cure[0:fine_num_y,0:fine_num_x,0:fine_num_z]
    fine_hr   = nd.zoom(hr[0:num_y_in_fine_y_len,0:num_x_in_fine_x_len,0:num_z_in_fine_z_len], [fine_num_y/num_y_in_fine_y_len, fine_num_x/num_x_in_fine_x_len, fine_num_z/num_z_in_fine_z_len])
    fine_hr   = fine_hr[0:fine_num_y,0:fine_num_x,0:fine_num_z]
    fine_field_ind = 1 ##TODO adjust of starting from + side
    slide_fine_field_ind = False
    
    print("Generate laplacian kernels...")
    kernel = get_laplace_kernel(dy,dx,dz,order=[1,1,1])
    fine_kernel = get_laplace_kernel(fine_dy,fine_dx,fine_dz,order=[3,3,3])
    
    print("Generate cure kinetics lookup tables...")
    T_start = 0.99*min(mean_T0-dev_T0,amb_T)
    T_end = 1.5*(((mean_hr+dev_hr)*(1.0-(mean_a0-dev_a0)))/cp+(mean_T0+dev_T0))
    T_space, T_step = np.linspace(T_start, T_end,100000,retstep=True)
    a_start = 0.99*(mean_a0-dev_a0)
    a_space, a_step = np.linspace(a_start,1.0,100000,retstep=True)
    arrhenius = A*np.exp(-E/(R*T_space))
    kinetics = ((a_space**m)*((1.0-a_space)**n))/(1.0+np.exp(C*(a_space-ac)))
    
    print("Setup front tracker...")
    front=[[0.0,0.0,0.0,0.0]]
    
    print("Setup frame buffer...")
    temp_images = []
    pre_temp = 0.0
    post_temp = 0.0
    
    print("Generate time...")
    times = np.linspace(0.0,tot_t,int(np.floor(tot_t/dt))+1)

    print("Build graph...")    
    verts = np.arange(1,num_y*num_x*num_z+1).reshape(num_y,num_x,num_z)
    edges = []
    weights = []
    for i in range(len(verts)):
        for j in range(len(verts[0,:,:])):
            for k in range(len(verts[0,0,:])):
                base = verts[i,j,k]
                if i!=0:
                    edges.append([base,verts[i-1,j,k]])
                    weights.append(dy**-2.0)
                if i!=len(verts)-1:
                    edges.append([base,verts[i+1,j,k]])
                    weights.append(dy**-2.0)
                if j!=0:
                    edges.append([base,verts[i,j-1,k]])
                    weights.append(dx**-2.0)
                if j!=len(verts[0,:,:])-1:
                    edges.append([base,verts[i,j+1,k]])
                    weights.append(dx**-2.0)
                if k!=0:
                    edges.append([base,verts[i,j,k-1]])
                    weights.append(dz**-2.0)
                if k!=len(verts[0,0,:])-1:
                    edges.append([base,verts[i,j,k+1]])
                    weights.append(dz**-2.0)
    edges=np.array(edges)
    weights=np.array(weights)
    verts = verts.flatten()
    
    print("Build laplace matrix...")
    G = nx.DiGraph()
    G.add_nodes_from(verts)
    G.add_edges_from(edges)
    Bve = -nx.incidence_matrix(G, oriented=True)
    W=sparse.diags(weights)
    laplacian = (Bve @ W) @ Bve.transpose()
    

    """
    ########################################################  SIMULATION ########################################################
    """
    print("Start simulation...")
    t0 = time.time()
    frames=times[times%(1./fr)<dt]
    for t in times:
        
        # Save frame
        if t in frames:
            if t==0.0:
                pass
            else:
                fine_temp_image = nd.zoom(fine_temp[:,:,0], [fine_num_y/num_y_in_fine_y_len, fine_num_x/num_x_in_fine_x_len])
                if loc_trig == 'y-' or loc_trig == 'y+':
                    temp_images.append(np.append(pre_temp[:,:,0],fine_temp_image,axis=0))
                    temp_images.append(np.append(temp_images,post_temp[:,:,0],axis=0))
                elif loc_trig == 'x-' or loc_trig == 'x+':
                    temp_images.append(np.append(pre_temp[:,:,0],fine_temp_image,axis=1))
                    temp_images.append(np.append(temp_images,post_temp[:,:,0],axis=1))
                elif loc_trig == 'z-' or loc_trig == 'z+':
                    if fine_field_ind==0:
                        temp_images.append(fine_temp_image)
                    else:
                        temp_images.append(np.append(pre_temp[:,:,0],fine_temp_image,axis=1))
        
        # Calculate trigger heat
        qy = [[0.0],[0.0]]
        qx = [[0.0],[0.0]]
        qz = [[0.0],[0.0]]
        if len(front)<2:
            if loc_trig=='y-':
                qy[0]=[q_trig]
            elif loc_trig=='y+':
                qy[1]=[q_trig]
            elif loc_trig=='x-':
                qx[0]=[q_trig]
            elif loc_trig=='x+':
                qx[1]=[q_trig]
            elif loc_trig=='z-':
                qz[0]=[q_trig]
            elif loc_trig=='z+':
                qz[1]=[q_trig]
        
        # Split the coarse field into pre and post fine field
        if slide_fine_field_ind or t==0.0:
            if loc_trig == 'y-' or loc_trig == 'y+':
                pre_temp = temp[0:fine_field_ind,:,:]
                pre_cure = cure[0:fine_field_ind,:,:]
                pre_hr   = hr[0:fine_field_ind,:,:]
                post_temp = temp[fine_field_ind+num_y_in_fine_y_len:,:,:]
                post_cure = cure[fine_field_ind+num_y_in_fine_y_len:,:,:]
                post_hr   = hr[fine_field_ind+num_y_in_fine_y_len:,:,:]
            elif loc_trig == 'x-' or loc_trig == 'x+':
                pre_temp = temp[:,0:fine_field_ind,:]
                pre_cure = cure[:,0:fine_field_ind,:]
                pre_hr   = hr[:,0:fine_field_ind,:]
                post_temp = temp[:,fine_field_ind+num_x_in_fine_x_len:,:]
                post_cure = cure[:,fine_field_ind+num_x_in_fine_x_len:,:]
                post_hr   = hr[:,fine_field_ind+num_x_in_fine_x_len:,:]
            elif loc_trig == 'z-' or loc_trig == 'z+':
                pre_temp = temp[:,:,0:fine_field_ind]
                pre_cure = cure[:,:,0:fine_field_ind]
                pre_hr   = hr[:,:,0:fine_field_ind]
                post_temp = temp[:,:,fine_field_ind+num_z_in_fine_z_len:]
                post_cure = cure[:,:,fine_field_ind+num_z_in_fine_z_len:]
                post_hr   = hr[:,:,fine_field_ind+num_z_in_fine_z_len:]
        
        # Step the pre coarse fields
        if pre_temp.size != 0:
            # Step the cure field
            pre_f = arrhenius[np.rint((pre_temp-T_start)/(T_step)).astype(int)]
            pre_g = kinetics[np.rint((pre_cure-a_start)/(a_step)).astype(int)]
            pre_cure_rate = pre_f*pre_g 
            pre_cure = pre_cure + pre_cure_rate*dt
            
            # Add pre temperature field boundary conditions
            if loc_trig == 'y-' or loc_trig == 'y+':
                pre_pad = nd.zoom(fine_temp[0,:,:], [num_x_in_fine_x_len/fine_num_x, num_z_in_fine_z_len/fine_num_z])
                pre_pad = pre_pad.reshape(1,len(pre_pad),len(pre_pad[0]))
                pre_temp = np.append(pre_temp,pre_pad,axis=0)
            elif loc_trig == 'x-' or loc_trig == 'x+':
                pre_pad = nd.zoom(fine_temp[:,0,:], [num_y_in_fine_y_len/fine_num_y, num_z_in_fine_z_len/fine_num_z])
                pre_pad = pre_pad.reshape(len(pre_pad),1,len(pre_pad[0]))
                pre_temp = np.append(pre_temp,pre_pad,axis=1)
            elif loc_trig == 'z-' or loc_trig == 'z+':
                pre_pad = nd.zoom(fine_temp[:,:,0], [num_y_in_fine_y_len/fine_num_y, num_x_in_fine_x_len/fine_num_x])
                pre_pad = pre_pad.reshape(len(pre_pad),len(pre_pad[0]),1)
                pre_temp = np.append(pre_temp,pre_pad,axis=2)
            
            # Step the temperature field
            pre_temp_rate = (k/(p*cp))*get_laplace(pre_temp,kernel,dy,dx,dz,k=k,c=htc,bc=amb_T,qy=qy,qx=qx,qz=qz,no_pad=loc_trig[0]+'+') + (pre_hr/cp)*pre_cure_rate
            if loc_trig == 'y-' or loc_trig == 'y+':
                pre_temp = pre_temp[0:-1,:,:] + pre_temp_rate*dt
            elif loc_trig == 'x-' or loc_trig == 'x+':
                pre_temp = pre_temp[:,0:-1,:] + pre_temp_rate*dt
            elif loc_trig == 'z-' or loc_trig == 'z+':
                pre_temp = pre_temp[:,:,0:-1] + pre_temp_rate*dt
            
        # Step the post coarse fields
        if post_temp.size != 0:
            # Step the cure field
            post_f = arrhenius[np.rint((post_temp-T_start)/(T_step)).astype(int)]
            post_g = kinetics[np.rint((post_cure-a_start)/(a_step)).astype(int)]
            post_cure_rate = post_f*post_g 
            post_cure = post_cure + post_cure_rate*dt
            
            # Add post temperature field boundary conditions
            if loc_trig == 'y-' or loc_trig == 'y+':
                post_pad = nd.zoom(fine_temp[-1,:,:], [num_x_in_fine_x_len/fine_num_x, num_z_in_fine_z_len/fine_num_z])
                post_pad = post_pad.reshape(1,len(post_pad),len(post_pad[0]))
                post_temp = np.append(post_pad,post_temp,axis=0)
            elif loc_trig == 'x-' or loc_trig == 'x+':
                post_pad = nd.zoom(fine_temp[:,-1,:], [num_y_in_fine_y_len/fine_num_y, num_z_in_fine_z_len/fine_num_z])
                post_pad = post_pad.reshape(len(post_pad),1,len(post_pad[0]))
                post_temp = np.append(post_pad,post_temp,axis=1)
            elif loc_trig == 'z-' or loc_trig == 'z+':
                post_pad = nd.zoom(fine_temp[:,:,-1], [num_y_in_fine_y_len/fine_num_y, num_x_in_fine_x_len/fine_num_x])
                post_pad = post_pad.reshape(len(post_pad),len(post_pad[0]),1)
                post_temp = np.append(post_pad,post_temp,axis=2)
            
            # Step the temperature field
            post_temp_rate = (k/(p*cp))*get_laplace(post_temp,kernel,dy,dx,dz,k=k,c=htc,bc=amb_T,qy=qy,qx=qx,qz=qz,no_pad=loc_trig[0]+'-') + (post_hr/cp)*post_cure_rate
            if loc_trig == 'y-' or loc_trig == 'y+':
                post_temp = post_temp[1:,:,:] + post_temp_rate*dt
            elif loc_trig == 'x-' or loc_trig == 'x+':
                post_temp = post_temp[:,1:,:] + post_temp_rate*dt
            elif loc_trig == 'z-' or loc_trig == 'z+':
                post_temp = post_temp[:,:,1:] + post_temp_rate*dt
        
        # Step the fine fields
        for sub_t in range(int(dt/fine_dt)):
            
            # Calculate fine cure field rate
            fine_f = arrhenius[np.rint((fine_temp-T_start)/(T_step)).astype(int)]
            fine_g = kinetics[np.rint((fine_cure-a_start)/(a_step)).astype(int)]
            fine_cure_rate = fine_f*fine_g 
        
            # Step cure via implicit trapezoid method if cure rate is high
            imp_loc = fine_cure_rate>1.0
            if imp_loc.any():
                imp_inds = np.transpose(np.array(np.unravel_index([i for i, x in enumerate(imp_loc.flatten()) if x], imp_loc.shape)))
                
                # Get the front location
                front.append([np.mean(dy*imp_inds[:,0]),np.mean(dx*imp_inds[:,1]),np.mean(dz*imp_inds[:,2]),t])
            
                for inds in imp_inds:
                    curr_fine_cure = fine_cure[inds[0],inds[1],inds[2]]
                    fine_cure[inds[0],inds[1],inds[2]] = minimize_scalar(get_err,bounds=(curr_fine_cure,1.0),method='Bounded',args=(curr_fine_cure,fine_cure_rate[inds[0],inds[1],inds[2]],fine_f[inds[0],inds[1],inds[2]],fine_dt),options={'xatol':a_step, 'maxiter':14}).x
                    fine_cure_rate[inds[0],inds[1],inds[2]] = (fine_cure[inds[0],inds[1],inds[2]] - curr_fine_cure)/fine_dt
                
            # Step cure via forward Euler method if cure rate is low
            exp_loc = np.invert(imp_loc)
            if exp_loc.any():
                fine_cure[exp_loc] = fine_cure[exp_loc] + fine_cure_rate[exp_loc]*fine_dt
            
            # Get bondary temperatures from coarse fields for front propogating in y direction
            fine_pad_minus = 0.0
            fine_pad_plus = 0.0
            if loc_trig == 'y-' or loc_trig == 'y+':
                if pre_temp.size != 0:
                    fine_pad_minus = nd.zoom(pre_temp[-1,:,:],[fine_num_x/num_x_in_fine_x_len, fine_num_z/num_z_in_fine_z_len])
                    l0 = (fine_pad_minus + ((fine_temp[0,:,:] - fine_pad_minus) / dy)*(dy-3.*fine_dy)).reshape(1,len(fine_pad_minus),len(fine_pad_minus[0]))
                    l1 = (fine_pad_minus + ((fine_temp[0,:,:] - fine_pad_minus) / dy)*(dy-2.*fine_dy)).reshape(1,len(fine_pad_minus),len(fine_pad_minus[0]))
                    l2 = (fine_pad_minus + ((fine_temp[0,:,:] - fine_pad_minus) / dy)*(dy-1.*fine_dy)).reshape(1,len(fine_pad_minus),len(fine_pad_minus[0]))
                    fine_temp = np.append(l2,fine_temp,axis=0)
                    fine_temp = np.append(l1,fine_temp,axis=0)
                    fine_temp = np.append(l0,fine_temp,axis=0)
                if post_temp.size != 0:
                    fine_pad_plus = nd.zoom(post_temp[0,:,:],[fine_num_x/num_x_in_fine_x_len, fine_num_z/num_z_in_fine_z_len])
                    l0 = (fine_temp[-1,:,:] + ((fine_pad_plus-fine_temp[-1,:,:]) / dy)*(3.*fine_dy)).reshape(1,len(fine_pad_plus),len(fine_pad_plus[0]))
                    l1 = (fine_temp[-1,:,:] + ((fine_pad_plus-fine_temp[-1,:,:]) / dy)*(2.*fine_dy)).reshape(1,len(fine_pad_plus),len(fine_pad_plus[0]))
                    l2 = (fine_temp[-1,:,:] + ((fine_pad_plus-fine_temp[-1,:,:]) / dy)*(1.*fine_dy)).reshape(1,len(fine_pad_plus),len(fine_pad_plus[0]))
                    fine_temp = np.append(fine_temp,l2,axis=0)
                    fine_temp = np.append(fine_temp,l1,axis=0)
                    fine_temp = np.append(fine_temp,l0,axis=0)
                
            # Get bondary temperatures from coarse fields for front propogating in x direction
            elif loc_trig == 'x-' or loc_trig == 'x+':
                if pre_temp.size != 0:
                    fine_pad_minus = nd.zoom(pre_temp[:,-1,:],[fine_num_y/num_y_in_fine_y_len, fine_num_z/num_z_in_fine_z_len])
                    l0 = (fine_pad_minus + ((fine_temp[:,0,:] - fine_pad_minus) / dx)*(dx-3.*fine_dx)).reshape(len(fine_pad_minus),1,len(fine_pad_minus[0]))
                    l1 = (fine_pad_minus + ((fine_temp[:,0,:] - fine_pad_minus) / dx)*(dx-2.*fine_dx)).reshape(len(fine_pad_minus),1,len(fine_pad_minus[0]))
                    l2 = (fine_pad_minus + ((fine_temp[:,0,:] - fine_pad_minus) / dx)*(dx-1.*fine_dx)).reshape(len(fine_pad_minus),1,len(fine_pad_minus[0]))
                    fine_temp = np.append(l2,fine_temp,axis=1)
                    fine_temp = np.append(l1,fine_temp,axis=1)
                    fine_temp = np.append(l0,fine_temp,axis=1)
                if post_temp.size != 0:
                    fine_pad_plus = nd.zoom(post_temp[:,0,:],[fine_num_y/num_y_in_fine_y_len, fine_num_z/num_z_in_fine_z_len])
                    l0 = (fine_temp[:,-1,:] + ((fine_pad_plus-fine_temp[:,-1,:]) / dx)*(3.*fine_dx)).reshape(len(fine_pad_plus),1,len(fine_pad_plus[0]))
                    l1 = (fine_temp[:,-1,:] + ((fine_pad_plus-fine_temp[:,-1,:]) / dx)*(2.*fine_dx)).reshape(len(fine_pad_plus),1,len(fine_pad_plus[0]))
                    l2 = (fine_temp[:,-1,:] + ((fine_pad_plus-fine_temp[:,-1,:]) / dx)*(1.*fine_dx)).reshape(len(fine_pad_plus),1,len(fine_pad_plus[0]))
                    fine_temp = np.append(fine_temp,l2,axis=1)
                    fine_temp = np.append(fine_temp,l1,axis=1)
                    fine_temp = np.append(fine_temp,l0,axis=1)
            
            # Get bondary temperatures from coarse fields for front propogating in z direction
            elif loc_trig == 'z-' or loc_trig == 'z+':
                if pre_temp.size != 0:
                    fine_pad_minus = nd.zoom(pre_temp[:,:,-1],[fine_num_y/num_y_in_fine_y_len, fine_num_x/num_x_in_fine_x_len])
                    l0 = (fine_pad_minus + ((fine_temp[:,:,0] - fine_pad_minus) / dz)*(dz-3.*fine_dz)).reshape(len(fine_pad_minus),len(fine_pad_minus[0]),1)
                    l1 = (fine_pad_minus + ((fine_temp[:,:,0] - fine_pad_minus) / dz)*(dz-2.*fine_dz)).reshape(len(fine_pad_minus),len(fine_pad_minus[0]),1)
                    l2 = (fine_pad_minus + ((fine_temp[:,:,0] - fine_pad_minus) / dz)*(dz-1.*fine_dz)).reshape(len(fine_pad_minus),len(fine_pad_minus[0]),1)
                    fine_temp = np.append(l2,fine_temp,axis=2)
                    fine_temp = np.append(l1,fine_temp,axis=2)
                    fine_temp = np.append(l0,fine_temp,axis=2)
                if post_temp.size != 0:
                    fine_pad_plus = nd.zoom(post_temp[:,:,0],[fine_num_y/num_y_in_fine_y_len, fine_num_x/num_x_in_fine_x_len])
                    l0 = (fine_temp[:,:,-1] + ((fine_pad_plus-fine_temp[:,:,-1]) / dz)*(3.*fine_dz)).reshape(len(fine_pad_plus),len(fine_pad_plus[0],1))
                    l1 = (fine_temp[:,:,-1] + ((fine_pad_plus-fine_temp[:,:,-1]) / dz)*(2.*fine_dz)).reshape(len(fine_pad_plus),len(fine_pad_plus[0],1))
                    l2 = (fine_temp[:,:,-1] + ((fine_pad_plus-fine_temp[:,:,-1]) / dz)*(1.*fine_dz)).reshape(len(fine_pad_plus),len(fine_pad_plus[0],1))
                    fine_temp = np.append(fine_temp,l2,axis=2)
                    fine_temp = np.append(fine_temp,l1,axis=2)
                    fine_temp = np.append(fine_temp,l0,axis=2)
        
            # Step fine temperature via forward Euler method
            no_pad = ''
            if not isinstance(fine_pad_minus,float):
                no_pad=no_pad+loc_trig[0]+'-'
            if not isinstance(fine_pad_plus,float):
                no_pad=no_pad+loc_trig[0]+'+'
            fine_temp_rate = (k/(p*cp))*get_laplace(fine_temp,fine_kernel,fine_dy,fine_dx,fine_dz,k=k,c=htc,bc=amb_T,qy=qy,qx=qx,qz=qz,no_pad=no_pad) + (fine_hr/cp)*fine_cure_rate
            if not isinstance(fine_pad_minus,float):
                if loc_trig == 'y-' or loc_trig == 'y+':
                    fine_temp = fine_temp[3:,:,:]
                elif loc_trig == 'x-' or loc_trig == 'x+':
                    fine_temp = fine_temp[:,3:,:]
                elif loc_trig == 'z-' or loc_trig == 'z+':
                    fine_temp = fine_temp[:,:,3:]
            if not isinstance(fine_pad_plus,float):
                if loc_trig == 'y-' or loc_trig == 'y+':
                    fine_temp = fine_temp[:-3,:,:]
                elif loc_trig == 'x-' or loc_trig == 'x+':
                    fine_temp = fine_temp[:,:-3,:]
                elif loc_trig == 'z-' or loc_trig == 'z+':
                    fine_temp = fine_temp[:,:,:-3]
            fine_temp = fine_temp + fine_temp_rate*fine_dt
            
        #TODO copy everything back, detect front, slide fleids, etc.
    
    
    """
    ########################################################  POST PROCESSING  ########################################################
    """
    # # Calculate front speed
    # front = np.array(front)
    # speed = np.zeros(front[:,0:3].shape)
    # for i in range(len(speed)):
    #     min_time = front[i,3]-0.5
    #     if min_time < 0.0:
    #         min_time = 0.0
    #     max_time = min_time+1.0
    #     if max_time > front[-1,3]:
    #         max_time = front[-1,3]
    #     min_time = max_time-1.0
    #     min_ind = np.argmin(abs(front[:,3]-min_time))
    #     max_ind = np.argmin(abs(front[:,3]-max_time))
    #     if min_ind==max_ind:
    #         max_ind=min_ind+2
    #     speed[i,0] = 1000.0*stats.linregress(front[min_ind:max_ind,3],front[min_ind:max_ind,0]).slope
    #     speed[i,1] = 1000.0*stats.linregress(front[min_ind:max_ind,3],front[min_ind:max_ind,1]).slope
    #     speed[i,2] = 1000.0*stats.linregress(front[min_ind:max_ind,3],front[min_ind:max_ind,2]).slope
    # speed = np.sqrt(np.sum(speed**2,axis=1))
    # ss_speed = np.mean(speed[int(0.25*len(speed))//2:-int(0.25*len(speed))//2])
    
    
    """
    ########################################################  PLOTTING AND RENDERING  ########################################################
    """
    
    
    # # Print results
    # tf = time.time()
    # print(tf-t0)
        
    
    
    
    
    
    
    
    
    