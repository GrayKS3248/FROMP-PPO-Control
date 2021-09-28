# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import numpy as np
import os
import pickle

class Estimator:

    def __init__(self, x_dim, y_dim, x_max, y_max, order, load_path=""):
            
        if (load_path == ""):
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.x_max = x_max
            self.y_max = y_max
            self.order = order
            self.x_range = np.linspace(0.0, self.x_max, self.x_dim-1)   
            self.y_coords = np.linspace(0.0, self.y_max, self.y_dim)   
            self.A = self.get_A_matrix(self.y_coords, self.order)
            self.A_inv = np.linalg.inv(self.A)
            self.means = np.zeros((self.order+1))
            self.stdevs = np.ones((self.order+1))
            
        else:
            self.load(load_path)
            
    # Loads a given saved estimator
    # @param the path from which the estimator will be loaded
    def load(self, path):
        
        if not os.path.isdir(path):
            raise RuntimeError("Could not find " + path)
        else:
            with open(path+"/output", 'rb') as file:
                loaded_data = pickle.load(file)
        
        self.x_dim = loaded_data['x_dim']
        self.y_dim = loaded_data['y_dim']
        self.x_max = loaded_data['x_max']
        self.y_max = loaded_data['y_max']
        self.order = loaded_data['order']
        self.x_range = loaded_data['x_range']  
        self.y_coords = loaded_data['y_coords'] 
        self.A = loaded_data['A']
        self.A_inv = loaded_data['A_inv']
        self.means = loaded_data['means']
        self.stdevs = loaded_data['stdevs']
        
    # Calculates the A matirx used for shape fitting
    def get_A_matrix(self, y_coords, order):
        
        A = np.zeros((order+1,order+1))
        
        A_entries = []
        A_entries.append(len(y_coords))
        for i in range(1,2*order+1):
            A_entries.append(np.sum((y_coords)**i))
            
        for i in range(order+1):
            for j in range(order+1):
                A[i][j] = A_entries[i+j]
                
        return A
    
    # Estimates the front's x location based on the temperature image
    # @param the input temperature image
    # @return the estimated x location of the front
    def get_estimate(self, state):

        x_coords = np.diff(state,axis=0)
        x_coords = abs(x_coords)
        x_coords = np.argmax(x_coords,axis=0)
        x_coords = self.x_range[x_coords]
        
        B = np.zeros((self.order+1,1))
            
        for i in range(self.order+1):
            B[i] = np.sum(self.y_coords**i * x_coords)
        
        estimate = np.matmul(self.A_inv,B).squeeze()
        
        for i in range(self.order+1):
            estimate[i] = (estimate[i] - self.means[i]) / self.stdevs[i]
        
        return estimate
    
    # Saves the means and stdevs
    # @param list of all the fit coefficients
    # @return path at which data has been saved
    def save(self, estimates_list):
        print("\nSaving shape estimator results...")

        # Calculate means and stdevs
        self.means = np.mean(np.array(estimates_list),axis=0)
        self.stdevs = np.std(np.array(estimates_list),axis=0)

        # Store data to dictionary
        data = {
            'x_dim' : self.x_dim,
            'y_dim' : self.y_dim,
            'x_max' : self.x_max,
            'y_max' : self.y_max,
            'order' : self.order,
            'x_range' : self.x_range,
            'y_coords' : self.y_coords, 
            'A' : self.A,
            'A_inv' : self.A_inv,
            'means' : self.means,
            'stdevs' : self.stdevs
        }

        # Find save paths
        initial_path = "results/shape_estimator"
        path = initial_path
        done = False
        curr_dir_num = 1
        while not done:
            if not os.path.isdir(path):
                os.mkdir(path)
                done = True
            else:
                curr_dir_num = curr_dir_num + 1
                path = initial_path + "(" + str(curr_dir_num) + ")"

        # Pickle all important outputs
        save_file = path + "/output"
        with open(save_file, 'wb') as file:
            pickle.dump(data, file)

        return path