# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:53:17 2021

@author: GKSch
"""
from Autoencoder_2 import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

if __name__ == "__main__":
    
    # Define load paths
    paths = [
        'results/ks3_obj1_bn32_W', 
        'results/ks3_obj1_bn32_U', 
        'results/ks3_obj1_bn64_W', 
        'results/ks3_obj1_bn64_U',
        'results/ks3_obj3_bn32_W', 
        'results/ks3_obj3_bn32_U',
        'results/ks3_obj3_bn64_W', 
        'results/ks3_obj3_bn64_U', 
        'results/ks5_obj1_bn32_W',
        'results/ks5_obj1_bn32_U',
        'results/ks5_obj1_bn64_W',
        'results/ks5_obj1_bn64_U', 
        'results/ks5_obj3_bn32_W', 
        'results/ks5_obj3_bn32_U',
        'results/ks5_obj3_bn64_W', 
        'results/ks5_obj3_bn64_U'
        ]
    
    # Define labels associated with each loaded autoencoder
    label_description = 'Kernal Size___Objective___Bottleneck___Weighted/Unweighted'
    labels = [
        '3_1_32_W', 
        '3_1_32_U',
        '3_1_64_W',
        '3_1_64_U',
        '3_3_32_W',
        '3_3_32_U',
        '3_3_64_W',
        '3_3_64_U',
        '5_1_32_W', 
        '5_1_32_U',
        '5_1_64_W',
        '5_1_64_U',
        '5_3_32_W',
        '5_3_32_U',
        '5_3_64_W',
        '5_3_64_U',
        ]
    
    # Ensure proper number of paths and labels
    if len(paths) != len(labels):
        raise RuntimeError('Unable to parse number of autoencoders to load.')
    
    # Plotting memory
    autoencoders = []
    training_curves = []
    temp_reconstruction_errors = np.zeros(len(paths))
    objectives = []
    objective_types = [] 
    kernal_sizes = []
    kernal_sizes_types = []
    bottlenecks = []
    bottlenecks_types = []
    
    for i in range(len(paths)):
        
        # Load autoencoders and store their training curves
        autoencoders.append(Autoencoder(1, 1, 10, 10, 10, 10, 1, 1, 1))
        training_curve = autoencoders[i].load(paths[i])
        training_curve = savgol_filter(training_curve, 25, 3)
        training_curves.append(training_curve)
        
        # Gather temperature field reconstruction error
        num = 5
        for j in range(num):
            temp_batch = np.genfromtxt('training_data/DCPD_GC2/temp_data_'+str(j)+'.csv', delimiter=',')
            temp_batch = temp_batch.reshape(len(temp_batch)//autoencoders[i].x_dim, autoencoders[i].x_dim, autoencoders[i].y_dim)
            temp_reconstruction_errors[i] = temp_reconstruction_errors[i] + autoencoders[i].get_temp_error(temp_batch)
        temp_reconstruction_errors[i] = temp_reconstruction_errors[i] / float(num)
        
        # Determine which parameters are independent variables
        if len(objective_types)==0 or (np.array(objectives) != autoencoders[i].objective_fnc).all():
            objective_types.append(autoencoders[i].objective_fnc)
        if len(kernal_sizes_types)==0 or (np.array(kernal_sizes) != autoencoders[i].kernal_size).all():
            kernal_sizes_types.append(autoencoders[i].kernal_size)
        if len(bottlenecks_types)==0 or (np.array(bottlenecks) != autoencoders[i].bottleneck).all():
            bottlenecks_types.append(autoencoders[i].bottleneck)
        objectives.append(autoencoders[i].objective_fnc)
        kernal_sizes.append(autoencoders[i].kernal_size)
        bottlenecks.append(autoencoders[i].bottleneck)
    
    # Set font to monospace
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.monospace'] = ['DejaVu Sans Mono']
        
    # Plot all training curves
    plt.close()
    plt.clf()
    plt.title('Autoencoder Learning Curves',fontsize='xx-large')
    plt.xlabel("Optimization Batch",fontsize='large', labelpad=15)
    plt.ylabel("RMS Reconstruction Error",fontsize='large', labelpad=15)
    for i in range(len(training_curves)):
        plt.plot(np.arange(len(training_curves[i])),training_curves[i],lw=2.5,label=labels[i])
    plt.yscale("log")
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.legend(fontsize='large', loc='upper right')
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.savefig("results/training.png", dpi = 500)
    plt.close()
    
    # Plot temperature reconstruction errors
    fig,ax = plt.subplots()
    index = np.arange(len(temp_reconstruction_errors))
    bar_width=0.75
    rects_1=plt.barh(index, temp_reconstruction_errors, bar_width, alpha=0.90, edgecolor='k', color='b')
    for i, v in enumerate(temp_reconstruction_errors):
        ax.text(v-0.12*(max(temp_reconstruction_errors)-min(temp_reconstruction_errors)), i-0.20, str(round(v,2)),color='w',fontsize='large')
    plt.yticks(index, tuple(labels), fontsize='large')
    plt.xticks(fontsize='large')
    plt.ylabel(label_description,fontsize='large', labelpad=15)
    plt.xlabel('RMS Error [%]',fontsize='large', labelpad=15)
    plt.title("Temperature Field Reconstruction",fontsize='xx-large')
    plt.gcf().set_size_inches(11., 8.5)
    save_file = "results/temp_reconstruction.png"
    plt.savefig(save_file, dpi = 500)
    plt.close()
    
    # Plot ordered temperature reconstruction errors
    sorted_labels = [x for _, x in sorted(zip(temp_reconstruction_errors, labels))]
    sorted_labels.reverse()
    sorted_temp_reconstruction_errors = np.flip(np.sort(temp_reconstruction_errors))
    fig,ax = plt.subplots()
    index = np.arange(len(sorted_temp_reconstruction_errors))
    bar_width=0.75
    rects_1=plt.barh(index, sorted_temp_reconstruction_errors, bar_width, alpha=0.90, edgecolor='k', color='b')
    for i, v in enumerate(sorted_temp_reconstruction_errors):
        ax.text(v-0.12*(max(sorted_temp_reconstruction_errors)-min(sorted_temp_reconstruction_errors)), i-0.20, str(round(v,2)),color='w',fontsize='large')
    plt.yticks(index, tuple(sorted_labels), fontsize='large')
    plt.xticks(fontsize='large')
    plt.ylabel(label_description,fontsize='large', labelpad=15)
    plt.xlabel('RMS Error [%]',fontsize='large', labelpad=15)
    plt.title("Temperature Field Reconstruction",fontsize='xx-large')
    plt.gcf().set_size_inches(11., 8.5)
    save_file = "results/ordered_temp_reconstruction.png"
    plt.savefig(save_file, dpi = 500)
    plt.close()
        
    if len(objective_types) != 1:
        for i in range(len(objective_types)):
            
            # Plot training curves of same objective functions
            plt.close()
            plt.clf()
            plt.title('Autoencoder Learning Curves (Objective '+str(objective_types[i])+')',fontsize='xx-large')
            plt.xlabel("Optimization Batch",fontsize='large', labelpad=15)
            plt.ylabel("RMS Reconstruction Error",fontsize='large', labelpad=15)
            for j in range(len(training_curves)):
                if objectives[j] == objective_types[i]:
                    plt.plot(np.arange(len(training_curves[j])),training_curves[j],lw=2.5,label=labels[j])
            plt.yscale("log")
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.legend(fontsize='large', loc='upper right')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig("results/training_obj_"+str(objective_types[i])+".png", dpi = 500)
            plt.close()
    
    if len(kernal_sizes_types) != 1:
        for i in range(len(kernal_sizes_types)):
            
            # Plot training curves of same kernal sizes
            plt.close()
            plt.clf()
            plt.title('Autoencoder Learning Curves (Kernal Size '+str(kernal_sizes_types[i])+'x'+str(kernal_sizes_types[i])+')',fontsize='xx-large')
            plt.xlabel("Optimization Batch",fontsize='large', labelpad=15)
            plt.ylabel("RMS Reconstruction Error",fontsize='large', labelpad=15)
            for j in range(len(training_curves)):
                if kernal_sizes[j] == kernal_sizes_types[i]:
                    plt.plot(np.arange(len(training_curves[j])),training_curves[j],lw=2.5,label=labels[j])
            plt.yscale("log")
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.legend(fontsize='large', loc='upper right')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig("results/training_ks_"+str(kernal_sizes_types[i])+'x'+str(kernal_sizes_types[i])+".png", dpi = 500)
            plt.close()
    
    if len(bottlenecks_types) != 1:
        for i in range(len(bottlenecks_types)):
            
            # Plot training curves of same bottlenecks
            plt.close()
            plt.clf()
            plt.title('Autoencoder Learning Curves (Bottleneck '+str(bottlenecks_types[i])+')',fontsize='xx-large')
            plt.xlabel("Optimization Batch",fontsize='large', labelpad=15)
            plt.ylabel("RMS Reconstruction Error",fontsize='large', labelpad=15)
            for j in range(len(training_curves)):
                if bottlenecks[j] == bottlenecks_types[i]:
                    plt.plot(np.arange(len(training_curves[j])),training_curves[j],lw=2.5,label=labels[j])
            plt.yscale("log")
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.legend(fontsize='large', loc='upper right')
            plt.gcf().set_size_inches(8.5, 5.5)
            plt.savefig("results/training_bn_"+str(bottlenecks_types[i])+".png", dpi = 500)
            plt.close()