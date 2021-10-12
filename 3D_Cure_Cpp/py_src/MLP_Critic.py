# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self, len_encoded_image, num_additional_states):
        
        # Initialize inherited class
        super(Model, self).__init__()
        
        # Save size data
        self.len_encoded_image = len_encoded_image
        self.num_additional_states = num_additional_states

        #Initialize the fully connected layers for action generation
        self.fc1 = nn.Linear(self.len_encoded_image + self.num_additional_states, self.len_encoded_image + self.num_additional_states)
        self.fc2 = nn.Linear(self.len_encoded_image + self.num_additional_states, self.len_encoded_image + self.num_additional_states)
        self.fc3 = nn.Linear(self.len_encoded_image + self.num_additional_states, 1)

    def forward(self, x):
        
        #Feed-forward encoded image and additional inputs through FC layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        
        #Return value estimate
        return x