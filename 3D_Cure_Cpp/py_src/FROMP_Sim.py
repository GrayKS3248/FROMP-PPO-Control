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
def build_rect_graph(num_y,num_x,num_z,dy,dx,dz):
    """
    Builds a graph representation of a regular, 3 dimensional, rectangular grid.

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

    Returns
    -------
    graph : DiGraph — Directed graphs with self loops
        Returns weighted DiGraph object.
    vert_type : List of strings
        Defines the type of each vertex with a string of symbols

    """
    
    # Initialize vertices, edges, and weights
    verts = np.arange(1,num_y*num_x*num_z+1).reshape(num_y,num_x,num_z)
    vert_type=[]
    edges = []
    weights = []
    
    # Assign edges based on connectivity and weights based on step size. Note that weights are defined as step_size ^ -2
    for i in range(len(verts)):
        for j in range(len(verts[0,:,:])):
            for k in range(len(verts[0,0,:])):
                # Define edges
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
                    
                # Define vert type
                string=""
                if i==0:
                    string = string + "y-"
                elif i==len(verts)-1:
                    string = string + "y+"
                if j==0:
                    string = string + "x-"
                elif j==len(verts[0,:,:])-1:
                    string = string + "x+"
                if k==0:
                    string = string + "z-"
                elif k==len(verts[0,0,:])-1:
                    string = string + "z+"
                if string=="":
                    string="bulk"
                vert_type.append(string)
    
    # Format verts, edges, and weights
    edges=np.array(edges)
    weights=np.array(weights)
    verts = verts.flatten()
    
    # Generate and return a weight graph
    graph = nx.DiGraph()
    graph.add_nodes_from(verts)
    graph.add_weighted_edges_from(np.append(edges,weights.reshape(len(weights),1),axis=1))
    return graph,vert_type
    

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
    order : 3darray, optional
        Order of discrete laplacian in each dimension. Must be bewtween 1 and 3. The default is [1,1,1].

    Returns
    -------
    Compressed Sparse Row matrix
        Sparse matrix containing the laplacian matrix

    """
    fd_coeffs = [np.array([1.,-2.,1.]),np.array([-1./12.,4./3.,-5./2.,4./3.,-1./12.]),np.array([1./90.,-3./20.,3./2.,-49./18.,3./2.,-3./20.,1./90.])]
    padded_shape = (num_y+2*order[0])*(num_x+2*order[1])*(num_z+2*order[2])
    laplacian_rows = deque([])
    laplacian_cols = deque([])
    laplacian_dat = deque([])
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
            
    # Build laplacian matrix
    return sparse.csr_matrix((laplacian_dat,(laplacian_rows,laplacian_cols)),shape=(padded_shape,padded_shape))

    
#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#    
def build_rect_graph_mats(graph,dt,k=0.152,p=980.0,cp=1600.0,tol=1e-8,lap=True,forward=True,backward=True):
    """
    Creates laplcian matrix give a graph

    Parameters
    ----------
    graph : DiGraph — Directed graphs with self loops
        Weighted graph containing node and edge data.
    dt : float
        Discrete time step in seconds.
    k : float, optional
        Conductivity of material in [W/m-K]. The default is 0.152.
    p : float, optional
        Density of the material in [kg/m^3]. The default is 980.0.
    cp : float, optional
        Specific heat capacity of the material in [J/kg-K]. The default is 1600.0.
    tol : float, optional
        Absolute tolerance of the L2 matrix. The default is 1e-8.
    lap : bool, optional
        Determines whether the laplacian matrix is generated. The default is True.
    forward : bool, optional
        Determines whether the forward matrix is generated. The default is True.
    backward : bool, optional
        Determines whether the backward matrix is generated. The default is True.

    Returns
    -------
    tuple
        Containing the laplacian, L1, and L2 matrices as sparse csc matrices.

    """

    # Get the weights matrix
    W = []
    for edge in graph.edges:
        W.append(graph.edges[edge]['weight'])
    W=sparse.diags(np.array(W))
    
    # Get the directed incidence matrix
    Bve = nx.incidence_matrix(graph, oriented=True)
    
    # Calculate the laplacian matrix based on weights and incidence
    laplacian = -1.0*((Bve @ W) @ Bve.transpose())
    
    # Calculate the forward dynamics matrix
    if forward:
        L1 = sparse.csc_matrix(sparse.eye(len(graph.nodes))+(k*dt/(2.0*p*cp))*laplacian)
    
    # Calculate the backward dynamics matrix
    if backward:
        L2 = sparse.csc_matrix(sparse.eye(len(graph.nodes))-(k*dt/(2.0*p*cp))*laplacian)
        L2 = sparse.linalg.inv(L2)
        L2[tol>=abs(L2)]=0.0
    
    # Return calculated matrices
    ret = []
    if lap:
        ret.append(laplacian)
    if forward:
        ret.append(L1)
    if backward:
        ret.append(L2)
    return tuple(ret)


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def estimate_L2(vert_type,num_y,num_x,num_z,dy,dx,dz,dt):
    """
    Estimates the backward dynamics matrix. Faster than true L2 calculation via build_rect_graph_mats function when total number of vertices is greater than ~1000.

    Parameters
    ----------
    vert_type : List of strings
        Defines the type of each vertex with a string of symbols
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
    dt : float
        Discrete time step in seconds.

    Returns
    -------
    L2 : Compressed Sparse Column matrix
        Sparse matrix containing the backward discrete dynamics of the temperature field.

    """
    
    # Generate reduced domain graph and L2 matrix
    reduced_graph,reduced_vert_type = build_rect_graph(min(num_y,10),min(num_x,10),min(num_z,10),dy,dx,dz)
    reduced_L2 = build_rect_graph_mats(reduced_graph,dt,lap=False,forward=False)[0].toarray()
    
    # Collect all vertex types
    types=[]
    for type in reduced_vert_type:
        if type not in types:
            types.append(type)
            
    # Average each vertex type's L2 row
    reduced_type_avg_L2 = []
    cen_ind = ((2*len(reduced_L2[0,:])-1)//2)
    for type in types:
        reduced_type_avg_L2.append([])
        for vert in range(len(reduced_vert_type)):
            if reduced_vert_type[vert]==type:
                left = reduced_L2[vert,0:vert]
                center = reduced_L2[vert,vert]
                right = reduced_L2[vert,(vert+1):]
                ext_row = np.zeros(2*len(reduced_L2[0,:])-1)
                ext_row[cen_ind-len(left):cen_ind] = left
                ext_row[cen_ind] = center
                ext_row[cen_ind+1:len(right)+cen_ind+1] = right
                reduced_type_avg_L2[-1].append(ext_row)
        if len(reduced_type_avg_L2[-1])==0:
            reduced_type_avg_L2[-1] = np.array(reduced_type_avg_L2[-1])
        else:
            reduced_type_avg_L2[-1] = np.mean(np.array(reduced_type_avg_L2[-1]),axis=0)
    reduced_type_avg_L2 = np.array(reduced_type_avg_L2)  
    
    # Construct full sized L2 type rows
    type_avg_L2 = np.zeros((len(reduced_type_avg_L2),2*(num_y*num_x*num_z)-1))
    reduced_inds = np.arange(len(reduced_type_avg_L2[0])//2+1)
    reduced_coords = np.array([(reduced_inds)//(min(num_x,10)*min(num_z,10)), 
                               ((reduced_inds)%(min(num_x,10)*min(num_z,10)))//(min(num_z,10)), 
                               ((reduced_inds)%(min(num_x,10)*min(num_z,10)))%(min(num_z,10))])
    rel_inds = reduced_coords[0,:]*(num_x*num_z) + reduced_coords[1,:]*(num_z) + reduced_coords[2,:]
    rel_inds = np.append(-1*np.flip(rel_inds)[0:-1],rel_inds)
    inds = rel_inds + (2*(num_y*num_x*num_z)-1)//2
    for type in range(len(type_avg_L2)):
        type_avg_L2[type][inds] = reduced_type_avg_L2[type]

    # Construct reduced, estimated L2 matrix
    L2_dat = deque([])
    L2_i_ind = deque([])
    L2_j_ind = deque([])
    cen_ind = (2*len(vert_type)-1)//2
    str_lookup={}
    for row in range(len(vert_type)):
        
        # Determine which type current L2 row 
        try:
            types_ind = str_lookup[vert_type[row]]
        except:
            type = vert_type[row]
            found = False
            types_ind = 0
            while not found:
                if types[types_ind]==type:
                    found = True
                else:
                    types_ind = types_ind + 1
            str_lookup[type] = types_ind
        
        # Gather L2 type data, shift to diagonal, and store for sparse matrix
        cols = type_avg_L2[types_ind][cen_ind-row:len(vert_type)-row+cen_ind]
        col_inds = np.nonzero(cols)[0]
        L2_i_ind.extend(row*np.ones(col_inds.shape,dtype=np.int64))
        L2_j_ind.extend(col_inds)
        L2_dat.extend(cols[col_inds])
        
    # Create and return sparse L2 matrix
    L2 = sparse.csc_matrix((L2_dat,(L2_i_ind,L2_j_ind)))
    return L2


#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def build_rect_dynamics(num_y,num_x,num_z,dy,dx,dz,dt,k=0.152,p=980.,cp=1600.):
    
    # Attempt to load cached data
    string = str((num_y,num_x,num_z,dy,dx,dz))
    string = string.replace(", ","_")
    string = string.replace("(","")
    string = string.replace(")","")
    string = string.replace(".","-")
    string="cache/"+string+'.dat'
    
    # If cached data does not exist, create new dynamical matrices
    if not os.path.exists(string):
        graph,vert_type = build_rect_graph(num_y,num_x,num_z,dy,dx,dz) 
        
        # If graph is small, calculate exact matrices
        if (num_y*num_x*num_z <= 1000):
            laplacian,L1,L2 = build_rect_graph_mats(graph,dt,k=k,p=p,cp=cp)
        
        # Otherwise, estimate L2
        else:
            laplacian,L1 = build_rect_graph_mats(graph,dt,k=k,p=p,cp=cp,backward=False)
            L2 = estimate_L2(vert_type,num_y,num_x,num_z,dy,dx,dz,dt)
        
        # Cache the calculated data
        dat = {'graph':graph, 'vert_type':vert_type, 'laplacian':laplacian, 'L1':L1, 'L2':L2}
        with open(string, 'wb') as file:
            pickle.dump(dat, file)
    
    # If cached data does exist, load it
    else: 
        with open(string, 'rb') as file:
            dat = pickle.load(file)
            graph = dat['graph']
            vert_type = dat['vert_type']
            laplacian = dat['laplacian']
            L1 = dat['L1']
            L2 = dat['L2']
            
    # Return dynamical matrices
    return laplacian, L1, L2

#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
def get_boundaries(field,dy,dx,dz,k=0.152,c=20.0,bc=298.15,qy=[[0.0],[0.0]],qx=[[0.0],[0.0]],qz=[[0.0],[0.0]],order=[1,1,1],no_pad=''):
    """
    Adds Neumann boundary conditions to a given field.

    Parameters
    ----------
    field : 3darray
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
    order : 3darray, optional
        Order of boundaries added to field in each dimension. Must be bewtween 1 and 3. The default is [1,1,1].
    no_pad : string, optional
        Indicates which all faces that will not be padded with Neumann type virtual temperatures. Select from '', 'y-', 'y+', 'x-', 'x+', 'z-', 'z+'.
        Face selections may be delimited by any value. The default is ''.

    Returns
    -------
    3darray
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
    tot_t = 10.0       ## Seconds
    dt = 0.01          ## Seconds
    
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
    
    # Define laplacian order for each dimension
    order = [3,3,3]
    
    # Define initial conditions
    mean_T0 = 305.15
    dev_T0 = 2.0
    mean_a0 = 0.07
    dev_a0 = 0.005
    
    # Define boundary conditions
    htc = 30.0
    amb_T = 298.15
    
    # Define heat inputs
    q_trig = 0.0  ## Watts / Meter ^ 2
    loc_trig = 'x-'   ## 'y-', 'y+', 'x-', 'x+', 'z-', 'z+'
    
    # Define save options
    fr = 30.0
    
    
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
    
    print("Build dynamical matrices...")
    laplacian = build_laplacian_mat(num_y,num_x,num_z,dy,dx,dz,order=order)
    
    print("Generate cure kinetics lookup tables...")
    T_start = 0.90*min(mean_T0-dev_T0,amb_T)
    T_end = 1.5*(((mean_hr+dev_hr)*(1.0-(mean_a0-dev_a0)))/cp+(mean_T0+dev_T0))
    T_space, T_step = np.linspace(T_start, T_end,1000000,retstep=True)
    a_start = 0.99*(mean_a0-dev_a0)
    a_space, a_step = np.linspace(a_start,1.0,1000000,retstep=True)
    arrhenius = A*np.exp(-E/(R*T_space))
    kinetics = ((a_space**m)*((1.0-a_space)**n))/(1.0+np.exp(C*(a_space-ac)))
    kinetics_deriv = ((m*a_space**(m-1.)*(1.-a_space)**n)/(1.+np.exp(C*(a_space-ac))) + 
                      (-n*a_space**m*(1.-a_space)**(n-1.))/(1.+np.exp(C*(a_space-ac))) + 
                      (-C*a_space**m*(1.-a_space)**n*np.exp(C*(a_space-ac)))/((1.0+np.exp(C*(a_space-ac)))**2))
    del(T_space)
    del(a_space)
    
    print("Setup front tracker...")
    front=[[0.0,0.0,0.0,0.0]]
    
    print("Generate time sequence...")
    times = np.linspace(0.0,tot_t,int(np.floor(tot_t/dt))+1)
    
    print("Setup frame buffer...")
    temp_images = []
    frames=times[times%(1./fr)<dt]

    
    """
    ########################################################  SIMULATION ########################################################
    """
    print("Start simulation...")
    t0 = time.time()
    for t in times:
        
        # Save frame
        if t in frames:
            temp_images.append(temp[:,:,0])
        
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
        
        # Step the cure field
        f = arrhenius[np.rint((temp-T_start)/(T_step)).astype(int)]
        g = kinetics[np.rint((cure-a_start)/(a_step)).astype(int)]
        cure_rate = np.zeros((f*g).shape)
        cure = cure + cure_rate*dt
        
        # Pad the temperature field with boundary conditions
        padded_temp = get_boundaries(temp,dy,dx,dz,k=kappa,c=htc,bc=amb_T,qy=qy,qx=qx,qz=qz,order=order).flatten()
        padded_cure_rate = np.zeros((len(cure_rate)+2*order[0], len(cure_rate[0])+2*order[1], len(cure_rate[0][0])+2*order[2]))
        padded_cure_rate[order[0]:-order[0],order[1]:-order[1],order[2]:-order[2]] = (cure_rate*hr*dt)/cp
        padded_cure_rate = padded_cure_rate.flatten()
        
        # Step the temperature field via the Forward Euler method
        temp_rate = (kappa/(rho*cp))*(laplacian@padded_temp).reshape(num_y+2*order[0],num_x+2*order[1],num_z+2*order[2])
        temp_rate = temp_rate[order[0]:-order[0],order[1]:-order[1],order[2]:-order[2]]
        temp_rate =  temp_rate + (hr*cure_rate)/cp
        temp = temp + temp_rate*dt
        
        # Step the temperature field via the implicit trapezoidal method
        # temp = (L2L1@temp.flatten() + L2@cure_rate.flatten()).reshape(num_y,num_x,num_z)
        
    
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
    
    
    # Print results
    tf = time.time()
    print(tf-t0)
        
    
    
    
    
    
    
    
    
    