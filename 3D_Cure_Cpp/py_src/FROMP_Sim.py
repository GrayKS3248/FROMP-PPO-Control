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
from collections import deque
import pickle
import os
import copy
from scipy import optimize


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def build_rect_field(num_y,num_x,num_z,mean,max_dev=0.0,feat_size=0.25,ar=[1.,1.,1.],seed=10000):
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
    return field
    

#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def build_laplacian_mat(num_y,num_x,num_z,dy,dx,dz,order=[1,1,1]):
    """
    Builds a matrix that, when multiplied by a row-major (C-style) collapsed 3D field, returns the row-major collapsed laplaician of that field.

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
    Compressed Sparse Row matrix
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


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def get_boundaries(field,dy,dx,dz,k=0.152,c=20.0,bc=298.15,qy=[[0.0],[0.0]],qx=[[0.0],[0.0]],qz=[[0.0],[0.0]],order=[1,1,1],no_pad=''):
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
    no_pad : string, optional
        Indicates which all faces that will not be padded with Neumann type virtual temperatures. Select from '', 'y-', 'y+', 'x-', 'x+', 'z-', 'z+'.
        Face selections may be delimited by any value. The default is ''.

    Returns
    -------
    array_like, shape (num_y,num_x,num_z)
        Neumann type boundary padded field.

    """
        
    # Extend the field
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
        
    return padded_field


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def get_err(an,*args):
    """
    Calculates the square implicit error between the current cure at a point and an estimate of the next cure at the same point

    Parameters
    ----------
    an : array_like, shape (n,)
        Current guess of the next cure at every point.
    gn : array_like, shape (n,)
        The cure kinetics of each guessed cure.
    a1 : array_like, shape (n,)
        The current cure at every point.
    f1 : array_like, shape (n,)
        The current solution to the Arhhenius equation at every point.
    g1 : array_like, shape (n,)
        The current cure kinetics at every point.
    dt : float
        The discrete time step in seconds.

    Raises
    ------
    TypeError
        If required arguments are missing: 'gn', 'a1', 'f1', 'g1', or 'dt'

    Returns
    -------
    err : array_like, shape (n,)
        Array of square implicit errors between the current cure at all points and an estimate of the next cure at the same points.

    """

    try:
        gn = args[0]
    except:
        raise TypeError("Missing 5 required positional arguments: 'gn', 'a1', 'f1', 'g1', and 'dt'")
    try:
        a1 = args[1]
    except:
        raise TypeError("Missing 4 required positional arguments: 'a1', 'f1', 'g1', and 'dt'")
    try:
        f1 = args[2]
    except:
        raise TypeError("Missing 3 required positional arguments: 'f1', 'g1', and 'dt'")
    try:
        g1 = args[3]
    except:
        raise TypeError("Missing 2 required positional arguments: 'g1' and 'dt'")
    try:
        dt = args[4]
    except:
        raise TypeError("Missing 1 required positional argument: 'dt'")
    
    # Calculate square implicit error
    err = ((a1 + 0.5*dt*f1*(g1 + gn)) - an)*((a1 + 0.5*dt*f1*(g1 + gn)) - an)
    return err


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def get_jac(an,*args):
    """
    Calculates the derivative of the square implicit error with respect to the next cure guess

    Parameters
    ----------
    an : array_like, shape (n,)
        Current guess of the next cure at every point.
    gn : array_like, shape (n,)
        The cure kinetics of each guessed cure.
    dgn_dan : array_like, shape (n,)
        The derivative of the cure kinetics with respect to cure evaluated at the current guess of the next cure at every point.
    a1 : array_like, shape (n,)
        The current cure at every point.
    f1 : array_like, shape (n,)
        The current solution to the Arhhenius equation at every point.
    g1 : array_like, shape (n,)
        The current cure kinetics at every point.
    dt : float
        The discrete time step in seconds.
        
    Raises
    ------
    TypeError
        If required arguments are missing: 'gn', 'dgn_dan', 'a1', 'f1', 'g1', or 'dt'

    Returns
    -------
    jac : array_like, shape (n,)
        The derivative of the square implicit error with respect to the next cure guess at each point.

    """

    try:
        gn = args[0]
    except:
        raise TypeError("Missing 6 required positional arguments: 'gn', 'dgn_dan', 'a1', 'f1', 'g1', and 'dt'")
    try:
        dgn_dan = args[1]
    except:
        raise TypeError("Missing 5 required positional arguments: 'dgn_dan', 'a1', 'f1', 'g1', and 'dt'")
    try:
        a1 = args[2]
    except:
        raise TypeError("Missing 4 required positional arguments: 'a1', 'f1', 'g1', and 'dt'")
    try:
        f1 = args[3]
    except:
        raise TypeError("Missing 3 required positional arguments: 'f1', 'g1', and 'dt'")
    try:
        g1 = args[4]
    except:
        raise TypeError("Missing 2 required positional arguments: 'g1' and 'dt'")
    try:
        dt = args[5]
    except:
        raise TypeError("Missing 1 required positional argument: 'dt'")
    
    # Calculate gradient vector
    jac = 2. * (a1+0.5*dt*f1*(g1+gn)-an) * (0.5*dt*f1*dgn_dan-1.)
    return jac


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def get_hess(an,*args):
    """
    Calculates the second derivative of the square implicit error with respect to the next cure guess at each point.

    Parameters
    ----------
    an : array_like, shape (n,)
        Current guess of the next cure at every point.
    gn : array_like, shape (n,)
        The cure kinetics of each guessed cure.
    dgn_dan : array_like, shape (n,)
        The derivative of the cure kinetics with respect to cure evaluated at the current guess of the next cure at every point.
    d2gn_dan2 : array_like, shape (n,)
        The second derivative of the cure kinetics with respect to cure evaluated at the current guess of the next cure at every point.
    a1 : array_like, shape (n,)
        The current cure at every point.
    f1 : array_like, shape (n,)
        The current solution to the Arhhenius equation at every point.
    g1 : array_like, shape (n,)
        The current cure kinetics at every point.
    dt : float
        The discrete time step in seconds.

    Raises
    ------
    TypeError
        If required arguments are missing: 'gn', 'dgn_dan', 'd2gn_dan2', 'a1', 'f1', 'g1', or 'dt'.

    Returns
    -------
    hess : array_like, shape (n,)
        The second derivative of the square implicit error with respect to the next cure guess at each point.

    """
    try:
        gn = args[0]
    except:
        raise TypeError("Missing 7 required positional arguments: 'gn', 'dgn_dan', 'd2gn_dan2', 'a1', 'f1', 'g1', and 'dt'")
    try:
        dgn_dan = args[1]
    except:
        raise TypeError("Missing 6 required positional arguments: 'dgn_dan', 'd2gn_dan2', 'a1', 'f1', 'g1', and 'dt'")
    try:
        d2gn_dan2 = args[2]
    except:
        raise TypeError("Missing 5 required positional arguments: 'd2gn_dan2', 'a1', 'f1', 'g1', and 'dt'")
    try:
        a1 = args[3]
    except:
        raise TypeError("Missing 4 required positional arguments: 'a1', 'f1', 'g1', and 'dt'")
    try:
        f1 = args[4]
    except:
        raise TypeError("Missing 3 required positional arguments: 'f1', 'g1', and 'dt'")
    try:
        g1 = args[5]
    except:
        raise TypeError("Missing 2 required positional arguments: 'g1' and 'dt'")
    try:
        dt = args[6]
    except:
        raise TypeError("Missing 1 required positional argument: 'dt'")

    hess = 2.*(0.5*dt*f1*dgn_dan-1.)*(0.5*dt*f1*dgn_dan-1.) + 2.*(a1+0.5*dt*f1*(g1+gn)-an)*(0.5*dt*f1*d2gn_dan2)
    return hess
    

#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
if __name__ == "__main__":
    """
    ########################################################  USER INPUT  ########################################################
    """
    # Define material thermal properties
    kappa = 0.152  ## Watts / Meter * Kelvin
    cp = 1600.0    ## Joules / Kilogram * Kelvin
    rho = 980.0    ## Kilograms / Meter ^ 3
    
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
    tot_t = 10.0          ## Seconds
    dt = 0.002          ## Seconds
    
    # Define domain size
    y_len = 0.008     ## Meters
    x_len = 0.025      ## Meters
    z_len = 0.001     ## Meters
    
    # Define discrete grid step size
    dy = 0.0002        ## Meters
    dx = 0.0002        ## Meters
    dz = 0.0002        ## Meters
    
    # Define laplacian order for each dimension
    order = [3,3,3]
    imp_cure_rate = 1.0
    scale = 0.80
    decay = 0.80
    
    # Define initial conditions
    mean_T0 = 294.15
    dev_T0 = 0.5
    mean_a0 = 0.07
    dev_a0 = 0.01
    
    # Define boundary conditions
    htc = 60.0
    amb_T = 293.65
    
    # Define heat inputs
    q_trig = 20000.0  ## Watts / Meter ^ 2
    loc_trig = 'x-'   ## 'y-', 'y+', 'x-', 'x+', 'z-', 'z+'
    
    # Define save options
    fr = 30000.0
    path = "../results"
    
    
    """
    ########################################################  PRE-SIMULATION CALCULATIONS  ########################################################
    """
    print("Calculate number of points in grid...")
    num_y=int(np.rint(y_len/dy+1.))
    num_x=int(np.rint(x_len/dx+1.))
    num_z=int(np.rint(z_len/dz+1.))
    y_len = (num_y-1)*dy
    x_len = (num_x-1)*dx
    z_len = (num_z-1)*dz
    
    print("Generate initial fields...")
    max_len = max(y_len,x_len,z_len)
    ar=[y_len/max_len,x_len/max_len,z_len/max_len]
    temp = build_rect_field(num_y,num_x,num_z,mean_T0,max_dev=dev_T0,ar=ar,seed=10000)
    cure = build_rect_field(num_y,num_x,num_z,mean_a0,max_dev=dev_a0,ar=ar,seed=10001)
    hr =   build_rect_field(num_y,num_x,num_z,mean_hr,max_dev=dev_hr,ar=ar,seed=10002)
    x_grid,y_grid,z_grid = np.meshgrid(np.linspace(0,x_len,num_x),np.linspace(0,y_len,num_y),np.linspace(0,z_len,num_z))
    
    print("Build dynamical matrices...")
    laplacian = build_laplacian_mat(num_y,num_x,num_z,dy,dx,dz,order=order)
    
    print("Generate cure kinetics lookup tables...")    
    T_start = 0.90*min(mean_T0-dev_T0,amb_T)
    T_end = 1.5*(((mean_hr+dev_hr)*(1.0-(mean_a0-dev_a0)))/cp+(mean_T0+dev_T0))
    T_space, T_step = np.linspace(T_start, T_end,1000000,retstep=True)
    a_start = 0.99*(mean_a0-dev_a0)
    a_space, a_step = np.linspace(a_start,0.999999999,1000000,retstep=True)
    arrhenius = A*np.exp(-E/(R*T_space))
    kinetics = (abs(a_space)/a_space)*(((abs(a_space)**m)*((1.0-abs(a_space))**n))/(1.0+np.exp(C*(abs(a_space)-ac))))
    kinetics_deriv = (abs(a_space)/a_space)*((m*abs(a_space)**(m-1.)*(1.-abs(a_space))**n)/(1.+np.exp(C*(abs(a_space)-ac))) + 
                      (-n*abs(a_space)**m*(1.-abs(a_space))**(n-1.))/(1.+np.exp(C*(abs(a_space)-ac))) + 
                      (-C*abs(a_space)**m*(1.-abs(a_space))**n*np.exp(C*(abs(a_space)-ac)))/((1.0+np.exp(C*(abs(a_space)-ac)))**2))
    kinetics_2_deriv =  (abs(a_space)/a_space)*((m*(m-1.)*abs(a_space)**(m-2.)*(1.-abs(a_space))**(n)) / (1.+np.exp(C*(abs(a_space)-ac))) + 
                        (-m*n*abs(a_space)**(m-1.)*(1.-abs(a_space))**(n-1.)) / (1.+np.exp(C*(abs(a_space)-ac))) + 
                        (-m*C*abs(a_space)**(m-1.)*(1-abs(a_space))**(n)*np.exp(C*(abs(a_space)-ac))) / (1.+np.exp(C*(abs(a_space)-ac)))**2. + 
                        (-m*n*abs(a_space)**(m-1.)*(1.-abs(a_space))**(n-1.)) / (1.+np.exp(C*(abs(a_space)-ac))) + 
                        (n*(n-1.)*abs(a_space)**(m)*(1.-abs(a_space))**(n-2.)) / (1.+np.exp(C*(abs(a_space)-ac))) + 
                        (n*C*abs(a_space)**(m)*(1-abs(a_space))**(n-1.)*np.exp(C*(abs(a_space)-ac))) / (1.+np.exp(C*(abs(a_space)-ac)))**2. + 
                        (-m*C*abs(a_space)**(m-1.)*(1.-abs(a_space))**(n)*np.exp(C*(abs(a_space)-ac))) / (1.+np.exp(C*(abs(a_space)-ac)))**2. + 
                        (n*C*abs(a_space)**(m)*(1.-abs(a_space))**(n-1.)*np.exp(C*(abs(a_space)-ac))) / (1.+np.exp(C*(abs(a_space)-ac)))**2. + 
                        (-C*C*abs(a_space)**(m)*(1.-abs(a_space))**(n)*np.exp(C*(abs(a_space)-ac))) / (1.+np.exp(C*(abs(a_space)-ac)))**2. + 
                        (2.*C*C*abs(a_space)**(m)*(1.-abs(a_space))**(n)*np.exp(2.*C*(abs(a_space)-ac))) / (1.+np.exp(C*(abs(a_space)-ac)))**3.)
    del(T_space)
    del(a_space)
    
    print("Setup front tracker...")
    front=[[0.0,0.0,0.0,0.0]]
    
    print("Generate time sequence...")
    times = np.linspace(0.0,tot_t,int(np.floor(tot_t/dt))+1)
    
    print("Setup frame buffer...")
    temp_images = deque([])
    temp_rate_images = deque([])
    cure_images = deque([])
    cure_rate_images = deque([])
    frames=times[times%(1./fr)<dt]

    
    """
    ########################################################  SIMULATION ########################################################
    """
    print("Start simulation...")
    t0 = time.time()
    for t in times:
        
        # Save frame
        if t in frames:
            temp_images.append(copy.deepcopy(temp[len(temp)//2,:,:]))
            cure_images.append(copy.deepcopy(cure[len(cure)//2,:,:]))
        
        # Calculate trigger heat
        qy = [[0.0],[0.0]]
        qx = [[0.0],[0.0]]
        qz = [[0.0],[0.0]]
        if front[-1][1]<=5.*dx:
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
        
        # Get the laplacian of the temperature field
        padded_temp = get_boundaries(temp,dy,dx,dz,k=kappa,c=htc,bc=amb_T,qy=qy,qx=qx,qz=qz,order=order).flatten()
        temp_laplacian = (laplacian@padded_temp).reshape(num_y+2*order[0],num_x+2*order[1],num_z+2*order[2])[order[0]:-order[0],order[1]:-order[1],order[2]:-order[2]]
        
        # Calculate the FE cure rates
        f = arrhenius[np.rint((temp-T_start)/(T_step)).astype(int)]
        g = kinetics[np.rint((cure-a_start)/(a_step)).astype(int)]
        cure_rate = f*g
        
        # Determine which points are integrated via implicit trapezoid and which points are integrated via FE
        imp_points = cure_rate>imp_cure_rate
        exp_points = np.invert(imp_points)
        
        if (imp_points).any():

            # Record front location if it has moved in the propogation direction
            if 'y' in loc_trig:
                if abs(np.mean(y_grid[imp_points]) - front[-1][0])>0.5*dy or np.mean(y_grid[imp_points])==0.0:
                    front.append([np.mean(y_grid[imp_points]),np.mean(x_grid[imp_points]),np.mean(z_grid[imp_points]),t])   
            elif 'x' in loc_trig:
                if abs(np.mean(x_grid[imp_points]) - front[-1][1])>0.5*dx or np.mean(x_grid[imp_points])==0.0:
                    front.append([np.mean(y_grid[imp_points]),np.mean(x_grid[imp_points]),np.mean(z_grid[imp_points]),t])   
            elif 'z' in loc_trig:
                if abs(np.mean(z_grid[imp_points]) - front[-1][2])>0.5*dz or np.mean(z_grid[imp_points])==0.0:
                    front.append([np.mean(y_grid[imp_points]),np.mean(x_grid[imp_points]),np.mean(z_grid[imp_points]),t])   

            # Get the current conditions of cure regions with high expected cure rates
            itr = 0
            a1 = cure[imp_points]
            f1 = f[imp_points]
            g1 = g[imp_points]
            dg1 = scale * kinetics_deriv[np.rint((cure[imp_points]-a_start)/(a_step)).astype(int)]
            
            # Get the first step of gradient descent toward implicit solution
            an = (a1+0.5*dt*f1*(g1+g1-dg1*a1)) / (1. - 0.5*dt*f1*dg1)
            an[an>1.0] = 0.5 + 0.5*a1[an>1.0]
            gn = kinetics[np.rint((an-a_start)/(a_step)).astype(int)]
            dgn = (decay**(itr+1)*scale) * kinetics_deriv[np.rint((an-a_start)/(a_step)).astype(int)]
            
            # Calcualte the error to determine which instances require further gradient descent
            err = get_err(an,gn,a1,f1,g1,dt)
            update = err>2.*a_step
               
            # Use gradient descent to calculate implicit cure solutions
            while update.any():
                # Update iterator
                itr = itr + 1
                if itr > 14 and update.any():
                    print()
                
                # Take another gradient descent step
                next_an = (a1[update]+0.5*dt*f1[update]*(g1[update]+gn[update]-dgn[update]*an[update])) / (1. - 0.5*dt*f1[update]*dgn[update])
                next_an[next_an>1.0] = 0.5 + 0.5*an[update][next_an>1.0]
                an[update] = next_an
                gn[update] = kinetics[np.rint((an[update]-a_start)/(a_step)).astype(int)]
                dgn[update] = (decay**(itr+1)*scale) * kinetics_deriv[np.rint((an[update]-a_start)/(a_step)).astype(int)]
                
                # Calcualte the error to determine which instances require further gradient descent
                err = get_err(an,gn,a1,f1,g1,dt)
                update = err>2.*a_step
                
            # Update implicit and explicit points
            cure[imp_points] = an
            cure_rate[imp_points] = 0.5*f1*(g1 + gn)
            cure[exp_points] = cure[exp_points] + cure_rate[exp_points]*dt
        else:
            # Limit the FE cure rate so that the max cure value is exactly 0.99999
            cure_rate[((cure + cure_rate*dt)>1.0)] = (0.99999 - cure[((cure + cure_rate*dt)>1.0)])/dt
        
            # Use the FE method to step the cure
            cure = cure + cure_rate*dt
        
        # Step the temperature field via the Forward Euler method
        temp_rate = (kappa/(rho*cp))*temp_laplacian + (hr*cure_rate)/cp
        temp = temp + temp_rate*dt
        
        # Save frame
        if t in frames:
            temp_rate_images.append(copy.deepcopy(temp_rate[len(temp_rate)//2,:,:]))
            cure_rate_images.append(copy.deepcopy(cure_rate[len(cure_rate)//2,:,:]))
    
    """
    ########################################################  POST PROCESSING  ########################################################
    """
    # Remove instances where front is not moving
    front = np.array(front)
    if 'y' in loc_trig:
        front_started = abs(front[:,0]>=0.5*dy)
    elif 'x' in loc_trig:
        front_started = abs(front[:,1]>=0.5*dx)
    elif 'z' in loc_trig:
        front_started = abs(front[:,2]>=0.5*dz)
    front_started[0]=True
    front_started[len(front_started)-np.flip(front_started).argmin()-1] = True
    front = front[front_started,:]
    
    # Estimate speed by smoothing front position slope data
    speed = np.zeros(front.shape)
    min_time = front[1,3]
    max_time = front[-1,3]
    for i in range(1,len(speed)):
        
        # Determine what type of fit to apply
        use_forward = front[i,3] - 0.5 < min_time
        use_backward = front[i,3] + 0.5 > max_time
        
        # Forward finit difference
        if use_forward:
            min_ind = i
            max_ind = np.argmin(abs(front[:,3]-(front[i,3]+0.5)))
            if max_ind-min_ind == 1:
                max_ind = max_ind + 1
        
        # Backward finite differnce
        elif use_backward:
            min_ind = np.argmin(abs(front[:,3]-(front[i,3]-0.5)))
            max_ind = i
            if max_ind-min_ind == 1:
                min_ind = min_ind - 1
        
        # Central finite difference
        else:
            min_ind = np.argmin(abs(front[:,3]-(front[i,3]-0.5)))
            max_ind = np.argmin(abs(front[:,3]-(front[i,3]+0.5)))
            if max_ind-min_ind == 1:
                min_ind = min_ind - 1
                max_ind = max_ind + 1
            
        # Estimate speed via linear regression
        speed[i,0] = 1000.0*stats.linregress(front[min_ind:max_ind,3],front[min_ind:max_ind,0]).slope
        speed[i,1] = 1000.0*stats.linregress(front[min_ind:max_ind,3],front[min_ind:max_ind,1]).slope
        speed[i,2] = 1000.0*stats.linregress(front[min_ind:max_ind,3],front[min_ind:max_ind,2]).slope
        speed[i,3] = front[i,3]
        
    # Calculate speed in direction of propogation
    speed = np.insert(speed,1,[0.,0.,0.,speed[1,3]-dt],axis=0)
    if 'y' in loc_trig:
        prop_speed = speed[:,0]
        prop_pos = 1000.*front[:,0]
    elif 'x' in loc_trig:
        prop_speed = speed[:,1]
        prop_pos = 1000.*front[:,1]
    elif 'z' in loc_trig:
        prop_speed = speed[:,2]
        prop_pos = 1000.*front[:,2]
        
    # Estimate steady state speed
    ss_speed = np.mean(prop_speed[np.logical_and(prop_speed>(np.median(prop_speed)-np.std(prop_speed)),prop_speed<(np.median(prop_speed)+np.std(prop_speed)))])
    ss_std = np.std(prop_speed[np.logical_and(prop_speed>(np.median(prop_speed)-np.std(prop_speed)),prop_speed<(np.median(prop_speed)+np.std(prop_speed)))])
    
    
    """
    ########################################################  PLOTTING AND RENDERING  ########################################################
    """
    # Print time
    tf = time.time()
    print("Simulation took: " + str(np.round(tf-t0,1)) + " seconds\n\t(" + str(np.round((tf-t0)/tot_t,2)) + " CPU seconds per sim second)")
    
    # Plot front speed
    plt.clf()
    plt.gcf().set_size_inches(8, 5)
    plt.axhline(ss_speed,c='k',ls='--',label='Steady state\nspeed = '+str(np.round(ss_speed,2))+' mm/s',lw=1.5)
    plt.fill_between(speed[:,3],ss_speed+ss_std,ss_speed-ss_std,color='k',alpha=0.15,lw=0.)
    plt.plot(speed[:,3],prop_speed,c='r',lw=2.,label='Front speed')
    plt.xlabel("Time [Seconds]",fontsize=16)
    plt.ylabel("Speed [mm/s]",fontsize=16)
    plt.title("Front Propogation Speed",fontsize=20)
    plt.xlim(np.min(speed[:,3]),np.max(speed[:,3]))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    order=[1,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=14)
    plt.savefig(path + "/speed.svg", dpi = 500)
    plt.close()
    
    # Plot front position
    plt.clf()
    plt.gcf().set_size_inches(8, 5)
    plt.plot(front[:,3],prop_pos,c='r',lw=2.)
    plt.xlabel("Time [Seconds]",fontsize=16)
    plt.ylabel("Position [mm]",fontsize=16)
    plt.title("Front Position",fontsize=20)
    plt.xlim(np.min(speed[:,3]),np.max(speed[:,3]))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(path + "/position.svg", dpi = 500)
    plt.close()
    
        
    
    
    
    
    
    
    
    
    