# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import numpy as np
import scipy.ndimage

class Estimator:

    def __init__(self, x_dim, y_dim, x_max, y_max, threshold):
            
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_max = x_max
        self.y_max = y_max
        
        self.threshold = threshold
        
        self.x_range = np.linspace(0.0, self.x_max, self.x_dim)   
        self.y_range = np.linspace(0.0, self.y_max, self.y_dim)   
        self.y_grid, self.x_grid = np.meshgrid(self.y_range, self.x_range)
    
    # Estimates location and shape of hotspots
    # @param the input temperature image
    # @return location and shape of hotspots
    def get_estimate(self, state, test):

        classification_mask = state >= self.threshold
        segmentation_mask, num_regions = scipy.ndimage.label(classification_mask)
            
        max_temps = []
        y_range = []
        x_range = []
        for i in range(num_regions):
            max_temps.append(max(state[segmentation_mask == i+1]))
            y_range.append([min(self.y_grid[segmentation_mask==i+1]), max(self.y_grid[segmentation_mask==i+1])])
            x_range.append([min(self.x_grid[segmentation_mask==i+1]), max(self.x_grid[segmentation_mask==i+1])])
        
        max_temps = np.array(max_temps)
        y_range = np.array(y_range)
        x_range = np.array(x_range)
        
        ret_val = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
        if len(max_temps) == 1:
            ret_val = (max_temps[0], x_range[0,0]/self.x_max, x_range[0,1]/self.x_max, y_range[0,0]/self.y_max, y_range[0,1]/self.y_max, -1.0, -1.0, -1.0, -1.0, -1.0)
        elif len(max_temps) >= 2:
            sort_indices = np.argsort(max_temps)
            index_1 = sort_indices[-1]
            index_2 = sort_indices[-2]
            ret_val = [max_temps[index_1]]
            ret_val.append(x_range[index_1,0]/self.x_max)
            ret_val.append(x_range[index_1,1]/self.x_max)
            ret_val.append(y_range[index_1,0]/self.y_max)
            ret_val.append(y_range[index_1,1]/self.y_max)
            ret_val.append(max_temps[index_2])
            ret_val.append(x_range[index_2,0]/self.x_max)
            ret_val.append(x_range[index_2,1]/self.x_max)
            ret_val.append(y_range[index_2,0]/self.y_max)
            ret_val.append(y_range[index_2,1]/self.y_max)
            ret_val = tuple(ret_val)
            
        return ret_val