# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, x_dim, y_dim, bottleneck, kernal_size):
        
        # Initialize inherited class
        super(Model, self).__init__()
        
        #Initialize class variables
        self.size = 16 * x_dim//8 * y_dim//8
        self.conv_dim = torch.Size([1, 16, x_dim//8, y_dim//8])
        
        # Initialize the max pool function
        self.pool = nn.MaxPool2d(2, 2)
        
        # Initialize the stdev for each output
        self.stdev_1 = torch.nn.Parameter(-0.9 * torch.ones(1, dtype=torch.double).double())
        self.stdev_2 = torch.nn.Parameter(-0.9 * torch.ones(1, dtype=torch.double).double())
        self.stdev_3 = torch.nn.Parameter(-0.9 * torch.ones(1, dtype=torch.double).double())
        
        #Initialize the encoding convolutional layers
        self.conv1 = nn.Conv2d(1, 2,  kernal_size, padding=kernal_size//2)
        self.conv2 = nn.Conv2d(2, 4,  kernal_size, padding=kernal_size//2)
        self.conv3 = nn.Conv2d(4, 16, kernal_size, padding=kernal_size//2)
        
        #Initialize the encoding linear layer
        self.fc0 = nn.Linear(self.size, bottleneck)

        #Initialize the fully connected layers
        self.fc1 = nn.Linear(bottleneck, bottleneck)
        self.fc2 = nn.Linear(bottleneck, bottleneck//2)
        self.fc3 = nn.Linear(bottleneck//2, 3)
        

    def forward(self, x):
        #Feed-forward through encoder
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.size)
        x = torch.sigmoid(self.fc0(x))
        
        #Feed-forward through FC layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        # Calculate stdev
        stdev_1 = torch.exp(self.stdev_1)
        stdev_2 = torch.exp(self.stdev_2)
        stdev_3 = torch.exp(self.stdev_3)
        
        #Return x
        return x, stdev_1, stdev_2, stdev_3