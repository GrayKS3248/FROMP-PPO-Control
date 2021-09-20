# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Model(nn.Module):
    
    def __init__(self, dim, bottleneck, num_output_features):
        
        # Initialize inherited class
        super(Model, self).__init__()
        
        #Initialize class variables
        #self.size = 16 * x_dim_input//8 * y_dim_input//8
        #self.conv_dim = torch.Size([1, 16, x_dim_input//8, y_dim_input//8])
        
        #Initialize the encoding convolutional layers
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 7, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 7, stride=2)
        self.conv4 = nn.Conv2d(128, 128, 5, stride=2)
        self.conv5 = nn.Conv2d(128, 128, 5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        #Initialize the encoding linear layers
        self.fc1 = nn.Linear(10, bottleneck)

        #Initialize the decoding linear layers
        self.t_fc1 = nn.Linear(bottleneck, 10)
        
        #Initialize the decoding convolutional layers
        self.t_conv1 = nn.ConvTranspose2d(16, 4, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(4,  2, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(2, num_output_features, 2, stride=2)

    def forward(self, x):
        #Feed-forward x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, self.size)
        x = torch.sigmoid(self.fc1(x))
        x = F.relu(self.t_fc1(x))
        x = x.view(self.conv_dim)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv3(x))
        
        #Return x
        return x
    
    def forward_features(self, x):
        #Feed-forward x
        x = self.pool(F.relu(self.conv1(x)))
        features_1 = deepcopy(x)
        x = self.pool(F.relu(self.conv2(x)))
        features_2 = deepcopy(x)
        x = self.pool(F.relu(self.conv3(x)))
        features_3 = deepcopy(x)
        x = x.view(-1, self.size)
        x = torch.sigmoid(self.fc1(x))
        x = F.relu(self.t_fc1(x))
        x = x.view(self.conv_dim)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv3(x))
        
        #Return x
        return x, features_1, features_2, features_3
    
    def encode(self, x):
        #Feed-forward x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.size)
        x = torch.sigmoid(self.fc1(x))
        
        #Return x
        return x  