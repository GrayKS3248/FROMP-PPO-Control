# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, x_dim, y_dim, bottleneck, kernal_size, num_states):
        
        # Initialize inherited class
        super(Model, self).__init__()
        
        #Initialize class variables
        self.num_states = num_states
        self.size = 8*self.num_states * x_dim//8 * y_dim//8
        self.conv_dim = torch.Size([1, 8*self.num_states, x_dim//8, y_dim//8])
        self.bottleneck = bottleneck
        
        # Initialize the max pool function
        self.pool = nn.MaxPool2d(2, 2)
        
        #Initialize the encoding convolutional layers
        self.conv1 = nn.Conv2d(self.num_states, 2*self.num_states,  kernal_size, padding=kernal_size//2)
        self.conv2 = nn.Conv2d(2*self.num_states, 4*self.num_states,  kernal_size, padding=kernal_size//2)
        self.conv3 = nn.Conv2d(4*self.num_states, 8*self.num_states, kernal_size, padding=kernal_size//2)
        
        #Initialize the encoding linear layer
        self.fc0 = nn.Linear(self.size, bottleneck)
        
        #Initialize the fully connected layers for state estimation
        self.fc1 = nn.Linear(bottleneck, bottleneck)
        self.fc2 = nn.Linear(bottleneck, (bottleneck)//2)
        self.fc3 = nn.Linear((bottleneck)//2, 1)
        

    def forward(self, states):
        
        #Feed-forward states through encoder
        states = self.pool(F.relu(self.conv1(states)))
        states = self.pool(F.relu(self.conv2(states)))
        states = self.pool(F.relu(self.conv3(states)))
        states = states.view(-1, self.size)
        states = torch.sigmoid(self.fc0(states))
        
        #Feed-forward through FC layers
        states = torch.tanh(self.fc1(states))
        states = torch.tanh(self.fc2(states))
        states = torch.tanh(self.fc3(states))
        
        #Return estimated state
        return states