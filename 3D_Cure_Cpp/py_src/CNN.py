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
    
    def __init__(self, dim, num_outputs, num_latent):
        
        # Initialize inherited class
        super(Model, self).__init__()
        
        #Initialize convolution size variables
        self.dim = dim
        self.num_outputs = num_outputs
        self.num_latent = num_latent
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
        
        #Initialize the latent variable reconstruction layers
        if self.num_latent >= 1:
            self.fc2 = nn.Linear(self.latent_size, self.latent_size//2)
            self.fc3 = nn.Linear(self.latent_size//2, self.latent_size//4)
            self.fc4 = nn.Linear(self.latent_size//4, self.num_latent)
        
        #Initialize the decoding linear layers
        self.t_fc1 = nn.Linear(self.bottleneck, self.latent_size)
        
        #Initialize the decoding convolutional layers
        self.t_conv1 = nn.ConvTranspose2d(self.f_5, self.f_4, self.k_5-1, stride=2, padding=self.k_5//2-1)
        self.t_conv2 = nn.ConvTranspose2d(self.f_4, self.f_3, self.k_4, stride=1, padding=self.k_4//2)
        self.t_conv3 = nn.ConvTranspose2d(self.f_3, self.f_2, self.k_3, stride=1, padding=self.k_3//2)
        self.t_conv4 = nn.ConvTranspose2d(self.f_2, self.f_1, self.k_2, stride=1, padding=self.k_2//2)
        self.t_conv5 = nn.ConvTranspose2d(self.f_1, self.num_outputs, self.k_1, stride=1, padding=self.k_1//2)
        self.NN_Up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        #Feed-forward through encoder
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, self.latent_size)
        
        # Feed-forward through latent MLP
        if self.num_latent >= 1:
            latent = F.relu(self.fc2(x))
            latent = F.relu(self.fc3(latent))
            latent = self.fc4(latent)
        x = F.relu(self.fc1(x))
        
        #Feed-forward through decoder
        x = F.relu(self.t_fc1(x))
        x = x.view((1, self.f_5, np.long(self.size_after_op8), np.long(self.size_after_op8))) 
        x = self.NN_Up(F.relu(self.t_conv1(x)))
        x = self.NN_Up(F.relu(self.t_conv2(x)))
        x = self.NN_Up(F.relu(self.t_conv3(x)))
        x = self.NN_Up(F.relu(self.t_conv4(x)))
        x = torch.sigmoid(self.t_conv5(x))
        
        #Return x
        if self.num_latent >= 1:
            return x, latent
        else:
            return x
    
    def encode(self, x):
        #Feed-forward through encoder
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, self.latent_size)
        
        # Feed-forward through latent MLP
        if self.num_latent >= 1:
            latent = F.relu(self.fc2(x))
            latent = F.relu(self.fc3(latent))
            latent = self.fc4(latent)
        x = F.relu(self.fc1(x))
        
        #Return x
        if self.num_latent >= 1:
            return x, latent
        else:
            return x
    
    def generate(self, x):
        #Feed-forward x
        x = F.relu(self.t_fc1(x))
        x = x.view((1, self.f_5, np.long(self.size_after_op8), np.long(self.size_after_op8))) 
        x = self.NN_Up(F.relu(self.t_conv1(x)))
        x = self.NN_Up(F.relu(self.t_conv2(x)))
        x = self.NN_Up(F.relu(self.t_conv3(x)))
        x = self.NN_Up(F.relu(self.t_conv4(x)))
        x = torch.sigmoid(self.t_conv5(x))
        
        #Return x
        return x  