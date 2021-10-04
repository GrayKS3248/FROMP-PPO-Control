# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    
    def __init__(self, dim, num_additional_inputs, num_outputs):
        
        # Initialize inherited class
        super(Model, self).__init__()
        
        #Initialize convolution size variables
        self.dim = dim
        self.num_additional_inputs = num_additional_inputs
        self.num_outputs = num_outputs
        self.bottleneck = 128
        
        # Conv layer 1
        self.f_1 = 6
        self.k_1 = 11
        self.p_1 = 4
        self.s_1 = 4
        
        # Conv layer 2
        self.f_2 = 16
        self.k_2 = 5
        self.p_2 = 2
        self.s_2 = 1
        
        # Conv layer 3
        self.f_3 = 24
        self.k_3 = 3
        self.p_3 = 1
        self.s_3 = 1
        
        # Conv layer 4
        self.f_4 = 24
        self.k_4 = 3
        self.p_4 = 1
        self.s_4 = 1
        
        # Conv layer 5
        self.f_5 = 16
        self.k_5 = 3
        self.p_5 = 1
        self.s_5 = 1
        
        # Max pool
        self.k_pool = 3
        self.p_pool = 1
        self.s_pool = 2
        
        # Calcualte size after convolutional operations
        self.size_after_op1 = np.floor((self.dim + 2*self.p_1 - self.k_1) / self.s_1) + 1
        self.size_after_op2 = np.floor((self.size_after_op1 + 2*self.p_pool - self.k_pool) / self.s_pool) + 1
        self.size_after_op3 = np.floor((self.size_after_op2 + 2*self.p_2 - self.k_2) / self.s_2) + 1
        self.size_after_op4 = np.floor((self.size_after_op3 + 2*self.p_pool - self.k_pool) / self.s_pool) + 1
        self.size_after_op5 = np.floor((self.size_after_op4 + 2*self.p_3 - self.k_3) / self.s_3) + 1
        self.size_after_op6 = np.floor((self.size_after_op5 + 2*self.p_4 - self.k_4) / self.s_4) + 1
        self.size_after_op7 = np.floor((self.size_after_op6 + 2*self.p_5 - self.k_5) / self.s_5) + 1
        self.size_after_op8 = np.floor((self.size_after_op7 + 2*self.p_pool - self.k_pool) / self.s_pool) + 1
        self.latent_size = np.long(self.f_5 * self.size_after_op8 * self.size_after_op8)
        
        #Initialize the encoding convolutional layers
        self.conv1 = nn.Conv2d(1, self.f_1, self.k_1, stride=self.s_1, padding=self.p_1)
        self.conv2 = nn.Conv2d(self.f_1, self.f_2, self.k_2, stride=self.s_2, padding=self.p_2)
        self.conv3 = nn.Conv2d(self.f_2, self.f_3, self.k_3, stride=self.s_3, padding=self.p_3)
        self.conv4 = nn.Conv2d(self.f_3, self.f_4, self.k_4, stride=self.s_4, padding=self.p_4)
        self.conv5 = nn.Conv2d(self.f_4, self.f_5, self.k_5, stride=self.s_5, padding=self.p_5)
        self.pool = nn.MaxPool2d(self.k_pool, stride=self.s_pool, padding=self.p_pool)
        
        #Initialize the encoding linear layers
        self.fc1 = nn.Linear(self.latent_size, self.bottleneck)
        
        # Initialize the stdev for each input
        self.stdev = torch.nn.Parameter(np.log(0.5)*torch.ones(self.num_outputs,dtype=torch.double).double())

        #Initialize the fully connected layers for action generation
        self.fc2 = nn.Linear(self.bottleneck + self.num_additional_inputs, self.bottleneck + self.num_additional_inputs)
        self.fc3 = nn.Linear(self.bottleneck + self.num_additional_inputs, self.bottleneck + self.num_additional_inputs)
        self.fc4 = nn.Linear(self.bottleneck + self.num_additional_inputs, self.num_outputs)

    def forward(self, x, additional_x):
        
        #Feed-forward x through convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, self.latent_size)
        x = F.relu(self.fc1(x))
        
        #Feed-forward x and additional x through FC layers
        means = torch.cat((x,additional_x),1)
        means = torch.tanh(self.fc2(means))
        means = torch.tanh(self.fc3(means))
        means = self.fc4(means)
        
        # Calculate stdev
        stdevs = torch.exp(self.stdev)
        
        #Return action means and stdevs
        return means, stdevs