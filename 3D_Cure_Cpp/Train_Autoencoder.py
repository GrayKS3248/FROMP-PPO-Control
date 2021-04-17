# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:18:04 2021

@author: GKSch
"""

import time
from Autoencoder import Autoencoder
import numpy as np

if __name__ == "__main__":
    
    x_dim = 360
    y_dim = 40
    num_traj = 20
    samples_per_traj = 20
    samples_per_batch = 100
    
    autoencoder_1 = Autoencoder(1.0e-3, 1.0, x_dim, y_dim, 8, 16, 128, samples_per_batch, 1, 1)
    autoencoder_2 = Autoencoder(1.0e-3, 1.0, x_dim, y_dim, 8, 16, 128, samples_per_batch, 2, 2)
    autoencoder_3 = Autoencoder(1.0e-3, 1.0, x_dim, y_dim, 8, 16, 128, samples_per_batch, 3, 3)
    loss_1_buffer = []
    loss_2_buffer = []
    loss_3_buffer = []
    
    print('\n')
    print("Epoch " + 0 + "/" + str(int(num_traj//(samples_per_batch//samples_per_traj))) + " | Loss 1 = " + '{:.3f}'.format(0.000) + " | " + "Loss 2 = " + '{:.3f}'.format(0.000) + " | " + "Loss 3 = " + '{:.3f}'.format(0.000), end='\r')
    start_time = time.time()
    for i in range(int(num_traj//(samples_per_batch//samples_per_traj))):
        
            start = x_dim*samples_per_traj*int(samples_per_batch//samples_per_traj)*i
            end = num_traj*samples_per_traj*x_dim - x_dim*samples_per_traj*int(samples_per_batch//samples_per_traj)*(i+1)
            
            temp = np.genfromtxt('results/temp_data.csv', delimiter=',', skip_header=start, skip_footer=end)
            cure = np.genfromtxt('results/cure_data.csv', delimiter=',', skip_header=start, skip_footer=end)
    
            temp = temp.reshape(samples_per_batch, x_dim, y_dim)
            cure = cure.reshape(samples_per_batch, x_dim, y_dim)
    
            for j in range(samples_per_batch):
                loss_1 = autoencoder_1.learn(temp[j,:,:], cure[j,:,:])
                loss_2 = autoencoder_2.learn(temp[j,:,:], cure[j,:,:])
                loss_3 = autoencoder_3.learn(temp[j,:,:], cure[j,:,:])
            
                if loss_1 != -1:
                    print("Epoch " + str(i+1) + "/" + str(int(num_traj//(samples_per_batch//samples_per_traj))) + " | Loss 1 = " + '{:.3f}'.format(round(loss_1,3)) + " | " + "Loss 2 = " + '{:.3f}'.format(round(loss_2,3)) + " | " + "Loss 3 = " + '{:.3f}'.format(round(loss_3,3)), end='\r')
                    loss_1_buffer.append(loss_1)
                    loss_2_buffer.append(loss_2)
                    loss_3_buffer.append(loss_3)
    
    print("\n")
    path = autoencoder_1.save(loss_1_buffer)
    autoencoder_1.draw_training_curve(loss_1_buffer, path)
    
    print("\n")
    path = autoencoder_2.save(loss_2_buffer)
    autoencoder_2.draw_training_curve(loss_2_buffer, path)
    
    print("\n")
    path = autoencoder_3.save(loss_3_buffer)
    autoencoder_3.draw_training_curve(loss_3_buffer, path)
                
    elapsed = time.time()-start_time
    print("\nTraining took: " + str(round(elapsed,1)) + ' seconds')