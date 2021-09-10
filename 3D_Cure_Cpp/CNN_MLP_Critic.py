# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, x_dim, y_dim, bottleneck, kernal_size, num_states, num_inputs):
        
        # Initialize inherited class
        super(Model, self).__init__()
        
        #Initialize class variables
        self.size = 16 * x_dim//8 * y_dim//8
        self.conv_dim = torch.Size([1, 16, x_dim//8, y_dim//8])
        self.bottleneck = bottleneck
        self.num_states = num_states
        self.num_inputs = num_inputs
        
        # Initialize the max pool function
        self.pool = nn.MaxPool2d(2, 2)
        
        #Initialize the encoding convolutional layers
        self.conv1 = nn.Conv2d(1, 2,  kernal_size, padding=kernal_size//2)
        self.conv2 = nn.Conv2d(2, 4,  kernal_size, padding=kernal_size//2)
        self.conv3 = nn.Conv2d(4, 16, kernal_size, padding=kernal_size//2)
        
        #Initialize the encoding linear layer
        self.fc0 = nn.Linear(self.size, bottleneck)

        #Initialize the fully connected layers
        self.fc1 = nn.Linear(num_states*bottleneck+num_inputs, num_states*bottleneck+num_inputs)
        self.fc2 = nn.Linear(num_states*bottleneck+num_inputs, (num_states*bottleneck+num_inputs)//2)
        self.fc3 = nn.Linear((num_states*bottleneck+num_inputs)//2, 1)
        

    def forward(self, states, inputs):
        
        #Feed-forward states through encoder
        states = self.pool(F.relu(self.conv1(states)))
        states = self.pool(F.relu(self.conv2(states)))
        states = self.pool(F.relu(self.conv3(states)))
        states = states.view(-1, self.size)
        states = torch.sigmoid(self.fc0(states))
        states = states.view(-1, self.bottleneck*self.num_states)
        
        #Feed-forward through FC layers
        output = torch.cat((states,inputs),1)
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = torch.tanh(self.fc3(output))
        
        #Return value estimation
        return output