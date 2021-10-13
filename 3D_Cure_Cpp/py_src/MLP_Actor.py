# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    
    def __init__(self, len_encoded_image, num_additional_states, num_outputs):
        
        # Initialize inherited class
        super(Model, self).__init__()
        
        # Save size data
        self.len_encoded_image = len_encoded_image
        self.num_additional_states = num_additional_states
        self.num_outputs = num_outputs
        
        # Initialize the stdev for each input
        self.stdev = torch.nn.Parameter(np.log(0.75)*torch.ones(self.num_outputs,dtype=torch.double).double())

        #Initialize the fully connected layers for action generation
        self.fc1 = nn.Linear(self.len_encoded_image + self.num_additional_states, self.len_encoded_image + self.num_additional_states)
        self.fc2 = nn.Linear(self.len_encoded_image + self.num_additional_states, self.len_encoded_image + self.num_additional_states)
        self.fc3 = nn.Linear(self.len_encoded_image + self.num_additional_states, self.num_outputs)

    def forward(self, x):
        
        #Feed-forward encoded image and additional inputs through FC layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        
        # Calculate stdev
        stdevs = torch.exp(self.stdev)
        
        #Return action means and stdevs
        return x, stdevs
    
    def reset_stdev(self):
        self.stdev = torch.nn.Parameter(np.log(0.75)*torch.ones(self.num_outputs,dtype=torch.double).double())