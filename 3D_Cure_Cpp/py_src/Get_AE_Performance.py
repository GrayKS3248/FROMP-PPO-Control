# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:53:17 2021

@author: GKSch
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

if __name__ == "__main__":
    
    # Define load paths
    paths = [
        '../results/Uncertain/Uncertain_Controlled_AE', 
        ]
    
    # Define labels associated with each loaded autoencoder
    labels = [
        '16'
        ]
    
    # Ensure proper number of paths and labels
    if len(paths) != len(labels):
        raise RuntimeError('Unable to parse number of autoencoders to load.')
    
    # Plotting memory
    training_curves = []
    training_curve_avg = []
    training_curve_std = []
    
    for i in range(len(paths)):
        
        with open(paths[i]+"/output", 'rb') as file:
            loaded_data = pickle.load(file)
        
        # Load autoencoders and store their training curves
        training_curves.append(loaded_data['loss_curve'])
        
        # Get the moving average and stdev of the learning curve
        window = len(loaded_data['loss_curve']) // 5
        if window > 1:
            rolling_std = np.array(pd.Series(loaded_data['loss_curve']).rolling(window).std())
            rolling_avg = np.array(pd.Series(loaded_data['loss_curve']).rolling(window).mean())
            rolling_std = rolling_std[~np.isnan(rolling_std)]
            rolling_avg = rolling_avg[~np.isnan(rolling_avg)]
            training_curve_avg.append(rolling_avg)
            training_curve_std.append(rolling_std)
            
    # Set font to monospace
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.monospace'] = ['DejaVu Sans Mono']
        
    # Plot all training curves
    plt.close()
    plt.clf()
    plt.title('Autoencoder Learning Curves',fontsize='xx-large')
    plt.xlabel("Batch",fontsize='large', labelpad=15)
    plt.ylabel("Loss",fontsize='large', labelpad=15)
    for i in range(len(training_curves)):
        plt.plot(np.arange(len(training_curves[i])),training_curves[i],lw=2.0,label=labels[i])
    plt.yscale("log")
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.legend(title='Bottleneck', fontsize='large', loc='upper right')
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig("results/training.png", dpi = 500)
    plt.close()
    
    # Draw training curve with rolling values
    plt.clf()
    plt.title("Loss Curve, Window = " + str(window),fontsize='xx-large')
    plt.xlabel("Batch",fontsize='large')
    plt.ylabel("Loss",fontsize='large')
    for i in range(len(training_curves)):
        plt.plot(np.array([*range(len(training_curve_avg[i]))])+(len(training_curves[i])-len(training_curve_avg[i])+1),training_curve_avg[i],lw=2.0,label=labels[i])
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.legend(title='Bottleneck', fontsize='large', loc='upper right')
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig("results/training_window.png", dpi = 500)
    plt.close()