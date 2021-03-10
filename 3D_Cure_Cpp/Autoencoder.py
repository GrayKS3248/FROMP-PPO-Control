# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    
    def __init__(self, x_dim, y_dim, out_size):
        
        # Initialize inherited class
        super(NN, self).__init__()
        
        #Initialize class variables
        self.x_dim = x_dim
        self.y_dim = y_dim
        
        #Initialize the encoding convolutional layers
        self.conv1 = nn.Conv2d(1,  8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        #Determine size of input after convultion
        test = torch.rand(1,1,x_dim,y_dim)
        test = self.pool(F.relu(self.conv1(test)))
        test = self.pool(F.relu(self.conv2(test)))
        self.size = test.size()[0]*test.size()[1]*test.size()[2]*test.size()[3]
        self.conv_dim = test.size()
        
        #Initialize the encoding linear layers
        self.fc1 = nn.Linear(self.size, self.size//4)
        self.fc2 = nn.Linear(self.size//4, (self.size//4+out_size)//4)
        self.fc3 = nn.Linear((self.size//4+out_size)//4, out_size)

        #Initialize the decoding linear layers
        self.t_fc1 = nn.Linear(out_size, (self.size//4+out_size)//4)
        self.t_fc2 = nn.Linear((self.size//4+out_size)//4, self.size//4)
        self.t_fc3 = nn.Linear(self.size//4, self.size)
        
        #Initialize the decoding convolutional layers
        self.t_conv1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8,  1, 2, stride=2)

    def forward(self, x):
        #Feed-forward x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.relu(self.t_fc1(x))
        x = F.relu(self.t_fc2(x))
        x = F.relu(self.t_fc3(x))
        x = x.view(self.conv_dim)
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
        
        #Return x
        return x
    
    def encode(self, x):
        #Feed-forward x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        #Return x
        return x
        
    
if __name__ == '__main__':
    autoencoder = NN(120,24,50)
    state = torch.rand(1,1,120,24)
    rebuilt_state = autoencoder.forward(state)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer.zero_grad()
    loss = criterion(rebuilt_state, state)
    loss.backward()
    optimizer.step()
    print(loss.item())