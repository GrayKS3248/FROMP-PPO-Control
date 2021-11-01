# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:53:17 2021

@author: GKSch
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy import integrate

if __name__ == "__main__":
    
    # Define load paths
    controlled_path = "../results/PPO_18" 
    uncontrolled_path = "../results/SIM_2" 
    random_path = "../results/SIM_1" 
    
    # Define the temperature at which adiabatic, uncontrolled front speed would exactly equal target speed
    req_uncontrolled_temp = 306.15;
    
    # Define monomer properties
    initial_cure = 0.07
    initial_temp = 296.15
    Hr = 350000.0
    Cp = 1600.0
    rho = 980.0
    volume = 0.0000004
    area = 0.0008 + 0.0001 + 0.000016
    h = 20.0
    ambient_temp = 294.65
    
    # Load data
    with open(controlled_path+"/output", 'rb') as file:
        controlled = pickle.load(file)
    with open(uncontrolled_path+"/output", 'rb') as file:
        uncontrolled = pickle.load(file)
    with open(random_path+"/output", 'rb') as file:
        random = pickle.load(file)
        
    # Calculate energy addition required for uncontrolled speed to match target speed
    T_max = (Hr * (1.0 - initial_cure))/Cp + initial_temp;
    uncontrolled_initial_temperature = np.mean(uncontrolled['temperature_field'][0,:,:] * (T_max - initial_temp) + initial_temp)
    required_delta_T = req_uncontrolled_temp - uncontrolled_initial_temperature
    required_energy = Cp*required_delta_T*rho*volume + uncontrolled['time'][-1]*h*area*(req_uncontrolled_temp-ambient_temp)
    
    # Calculate energy usage
    controlled_energy = integrate.cumtrapz(controlled['power'], x=controlled['time'])
    controlled_energy = np.insert(controlled_energy, 0, 0.0)
    uncontrolled_energy = integrate.cumtrapz(uncontrolled['power'], x=uncontrolled['time'])
    uncontrolled_energy = np.insert(uncontrolled_energy, 0, 0.0)
    random_energy = integrate.cumtrapz(random['power'], x=random['time'])
    random_energy = np.insert(random_energy, 0, 0.0)
    
    # Plot energy trajectory
    plt.plot(controlled['time'], controlled_energy,c='k',lw=2.5)
    plt.plot(uncontrolled['time'], uncontrolled_energy,c='r',lw=2.5)
    plt.plot(random['time'], random_energy,c='b',lw=2.5)
    plt.plot(random['time'], required_energy*np.ones(len(random['time'])),c='m',lw=2.5,ls=":")
    plt.xlim(0.0, np.round(controlled['time'][-1]))
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.title("External Energy Input",fontsize='xx-large')
    plt.xlabel("Simulation Time [s]",fontsize='large')
    plt.ylabel("Cumulative Energy Consumed [J]",fontsize='large')
    plt.legend(('Controlled','Uncontrolled','Random','Required'), bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig("../results/reward.png", dpi = 500)
    plt.close()
    
    #     # Load autoencoders and store their training curves
    #     training_curves.append(loaded_data['loss_curve'])
        
    #     # Get the moving average and stdev of the learning curve
    #     window = len(loaded_data['loss_curve']) // 5
    #     if window > 1:
    #         rolling_std = np.array(pd.Series(loaded_data['loss_curve']).rolling(window).std())
    #         rolling_avg = np.array(pd.Series(loaded_data['loss_curve']).rolling(window).mean())
    #         rolling_std = rolling_std[~np.isnan(rolling_std)]
    #         rolling_avg = rolling_avg[~np.isnan(rolling_avg)]
    #         training_curve_avg.append(rolling_avg)
    #         training_curve_std.append(rolling_std)
            
    # # Set font to monospace
    # plt.rcParams['font.family'] = 'monospace'
    # plt.rcParams['font.monospace'] = ['DejaVu Sans Mono']
        
    # # Plot all training curves
    # plt.close()
    # plt.clf()
    # plt.title('Autoencoder Learning Curves',fontsize='xx-large')
    # plt.xlabel("Batch",fontsize='large', labelpad=15)
    # plt.ylabel("Loss",fontsize='large', labelpad=15)
    # for i in range(len(training_curves)):
    #     plt.plot(np.arange(len(training_curves[i])),training_curves[i],lw=2.0,label=labels[i])
    # plt.yscale("log")
    # plt.xticks(fontsize='large')
    # plt.yticks(fontsize='large')
    # plt.legend(title='Bottleneck', fontsize='large', loc='upper right')
    # plt.gcf().set_size_inches(8.5, 5.5)
    # plt.savefig("results/training.png", dpi = 500)
    # plt.close()
    
    # # Draw training curve with rolling values
    # plt.clf()
    # plt.title("Loss Curve, Window = " + str(window),fontsize='xx-large')
    # plt.xlabel("Batch",fontsize='large')
    # plt.ylabel("Loss",fontsize='large')
    # for i in range(len(training_curves)):
    #     plt.plot(np.array([*range(len(training_curve_avg[i]))])+(len(training_curves[i])-len(training_curve_avg[i])+1),training_curve_avg[i],lw=2.0,label=labels[i])
    # plt.xticks(fontsize='large')
    # plt.yticks(fontsize='large')
    # plt.legend(title='Bottleneck', fontsize='large', loc='upper right')
    # plt.gcf().set_size_inches(8.5, 5.5)
    # plt.savefig("results/training_window.png", dpi = 500)
    # plt.close()