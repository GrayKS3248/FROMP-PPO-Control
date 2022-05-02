# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:53:17 2021

@author: GKSch
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import integrate
import re
from string import digits

if __name__ == "__main__":
    
    
    ## INPUTS ##
    ## ====================================================================================================================================================================================================== ##
    # Define load and save paths
    combined_path = ""
    controlled_path = "../results/SIM_20"
    uncontrolled_path = "../results/Uncertain/Uncertain_Controlled_Sim_20"
    save_path = "../results"
    
    
    ## LOAD PRE-COMBINED DATA ##
    ## ====================================================================================================================================================================================================== ##
    # Load previously generated combined data
    print("Loading...")
    
    if combined_path != "":
        with open(combined_path+"/combined_data", 'rb') as file:
            data = pickle.load(file)
            
        for entry in data:
            globals()[entry] = data[entry]
            
            
    ## LOAD CONTROLLED SIMULATION DATA ##
    ## ====================================================================================================================================================================================================== ##
    # Load all previous simulations, store their trajectory, front shape, and energy consumption
    if combined_path == "":
        print("Loading controlled simulation data...")
        
        num_controlled = 0
        controlled_time = []
        controlled_mesh_x_y0_z0 = []
        controlled_mesh_width = []
        controlled_target = []
        controlled_front_speed = []
        controlled_max_cum_temp = []
        controlled_global_fine_mesh_x = []
        controlled_global_fine_mesh_y = []
        controlled_sorted_mean_front_x_locations = []
        controlled_sorted_front_temperature = []
        controlled_front_width = []
        controlled_front_skewness = []
        controlled_power = []
        controlled_energy = []
        controlled_bulk_energy = []
        controlled_ideal_energy = []
        controlled_cure_profile = []
        controlled_temp_profile = []
        controlled_profile_coords = []
                    
        remove_digits = str.maketrans('', '', digits)
        for sim in range(1,int(re.search(r'\d+$', controlled_path[-2:]).group())+1):
            controlled_path_no_digits = controlled_path.translate(remove_digits)
            print(controlled_path_no_digits+str(sim)+"/output")
            with open(controlled_path_no_digits+str(sim)+"/output", 'rb') as file:
                dat = pickle.load(file)
                num_controlled = num_controlled + 1
                
                # Simulation constants
                adiabatic_rxn_temp = dat['adiabatic_rxn_temp']
                initial_temperature = dat['initial_temperature']
                specific_heat = dat['specific_heat']
                density = dat['density']
                volume = dat['volume']
                heat_transfer_coeff = dat['heat_transfer_coeff']
                surface_area = dat['surface_area']
                ambient_temp = dat['ambient_temp']
                interpolated_temp_field = dat['interpolated_temp_field']
                
                # Simulation data
                controlled_time.append(dat['time'])
                controlled_mesh_x_y0_z0.append(dat['mesh_x_z0'][:,0])
                controlled_mesh_width.append(dat['mesh_y_z0'][-1][-1])
                controlled_target.append(dat['target'])
                controlled_front_speed.append(dat['front_velocity'])
                controlled_max_cum_temp.append(dat['cum_max_temp_field'][-1])
                
                # Only collect large spatial data once
                if num_controlled == 1:
                    controlled_global_fine_mesh_x = dat['global_fine_mesh_x']
                    controlled_global_fine_mesh_y = dat['global_fine_mesh_y']
        
                # Front temperature calculations
                sorted_mean_front_x_locations = 1000.0*np.array(sorted(dat['mean_front_x_locations']))
                sorted_front_temperature = np.array([x for _, x in sorted(zip(dat['mean_front_x_locations'], dat['front_temperature']))])
                
                # Store calculated front temperature data
                controlled_sorted_mean_front_x_locations.append(sorted_mean_front_x_locations)
                controlled_sorted_front_temperature.append(sorted_front_temperature)
                
                # Front shape calculations
                front_width = []
                front_skewness = []
                for i in range(len(dat['time'])):
                    front_x_location = dat['front_curve'][i][0]
                    front_x_location = front_x_location[front_x_location >= 0.0]
                    front_x_location = 1000.0*front_x_location
                    front_instances = len(front_x_location)
                    if front_instances > 3:
                        front_width.append(np.max(front_x_location) - np.min(front_x_location))
                        if not np.std(front_x_location) == 0.0:
                            front_skewness.append(abs((np.mean(front_x_location)-np.median(front_x_location))/np.std(front_x_location)))
                        else:
                            front_skewness.append(0.0)
                    else:
                        front_width.append(0.0)
                        front_skewness.append(0.0)
                front_width = np.array(front_width)
                front_skewness = np.array(front_skewness)
                
                # Store calculated front shape data
                controlled_front_width.append(front_width)
                controlled_front_skewness.append(front_skewness)
                
                # Energy and power calculations
                mean_initial_temp = np.mean(interpolated_temp_field[0]) * (adiabatic_rxn_temp - initial_temperature) + initial_temperature
                C1 = 0.00070591
                C2 = 0.0067238
                C3 = 0.53699
                power = dat['source_power'] + dat['trigger_power']
                energy = integrate.cumtrapz(power, x=dat['time'])
                energy = np.insert(energy, 0, 0.0)
                trigger_energy = integrate.cumtrapz(dat['trigger_power'], x=dat['time'])
                trigger_energy = np.insert(trigger_energy, 0, 0.0)
                target_temp = ((np.sqrt(C2*C2 - 4.0*C1*(C3-1000.0*dat['target'][-1])) - C2) / (2.0 * C1)) + 273.15
                bulk_energy = specific_heat*density*volume*(target_temp - mean_initial_temp) + heat_transfer_coeff*surface_area*(target_temp - ambient_temp)*dat['time'] + trigger_energy
                ideal_energy = specific_heat*density*volume*(target_temp - mean_initial_temp) + trigger_energy
                
                # Store calculated energy and power data
                controlled_power.append(power)
                controlled_energy.append(energy)
                controlled_bulk_energy.append(bulk_energy)
                controlled_ideal_energy.append(ideal_energy)
                
                # Get the temperature and cure profiles at selected times after ignition
                steps = []
                if dat['time'][-1] > 8.0:
                    steps.append(np.argmin(np.abs(dat['time'] - 8.0)))
                if dat['time'][-1] > 18.0:
                    steps.append(np.argmin(np.abs(dat['time'] - 18.0)))
                if dat['time'][-1] > 28.0:
                    steps.append(np.argmin(np.abs(dat['time'] - 28.0)))
                controlled_cure_profile_list = []
                controlled_temp_profile_list = []
                for ind in range(len(steps)):
                    curr_step = steps[ind]
                    controlled_cure_profile_list.append(dat['interpolated_cure_field'][curr_step][:,len(dat['interpolated_cure_field'][curr_step][0])//2])
                    controlled_temp_profile_list.append(dat['interpolated_temp_field'][curr_step][:,len(dat['interpolated_temp_field'][curr_step][0])//2])
                    controlled_profile_coords_list = dat['global_fine_mesh_x'][:,0]
                controlled_cure_profile.append(controlled_cure_profile_list)
                controlled_temp_profile.append(controlled_temp_profile_list)
                controlled_profile_coords.append(controlled_profile_coords_list)
                
        # Standard deviation of controlled data
        std_controlled_front_speed = np.std(np.array(controlled_front_speed),axis=0)
        std_controlled_max_cum_temp = np.std(np.array(controlled_max_cum_temp),axis=0)
        std_controlled_sorted_mean_front_x_locations = np.std(np.array(controlled_sorted_mean_front_x_locations),axis=0)
        std_controlled_sorted_front_temperature = np.std(np.array(controlled_sorted_front_temperature),axis=0)
        std_controlled_front_width = np.std(np.array(controlled_front_width),axis=0)
        std_controlled_front_skewness = np.std(np.array(controlled_front_skewness),axis=0)
        std_controlled_power = np.std(np.array(controlled_power),axis=0)
        std_controlled_energy = np.std(np.array(controlled_energy),axis=0)
        std_controlled_bulk_energy = np.std(np.array(controlled_bulk_energy),axis=0)
        std_controlled_ideal_energy = np.std(np.array(controlled_ideal_energy),axis=0)
        std_controlled_cure_profile = np.std(np.array(controlled_cure_profile),axis=0)
        std_controlled_temp_profile = np.std(np.array(controlled_temp_profile),axis=0)
        std_controlled_profile_coords = np.std(np.array(controlled_profile_coords),axis=0)
        
        # Mean of controlled data
        controlled_time = np.mean(np.array(controlled_time),axis=0)
        controlled_mesh_x_y0_z0 = np.mean(np.array(controlled_mesh_x_y0_z0),axis=0)
        controlled_mesh_width = np.mean(np.array(controlled_mesh_width),axis=0)
        controlled_target = np.mean(np.array(controlled_target),axis=0)
        controlled_front_speed = np.mean(np.array(controlled_front_speed),axis=0)
        controlled_max_cum_temp = np.mean(np.array(controlled_max_cum_temp),axis=0)
        controlled_sorted_mean_front_x_locations = np.mean(np.array(controlled_sorted_mean_front_x_locations),axis=0)
        controlled_sorted_front_temperature = np.mean(np.array(controlled_sorted_front_temperature),axis=0)
        controlled_front_width = np.mean(np.array(controlled_front_width),axis=0)
        controlled_front_skewness = np.mean(np.array(controlled_front_skewness),axis=0)
        controlled_power = np.mean(np.array(controlled_power),axis=0)
        controlled_energy = np.mean(np.array(controlled_energy),axis=0)
        controlled_bulk_energy = np.mean(np.array(controlled_bulk_energy),axis=0)
        controlled_ideal_energy = np.mean(np.array(controlled_ideal_energy),axis=0)
        controlled_cure_profile = np.mean(np.array(controlled_cure_profile),axis=0)
        controlled_temp_profile = np.mean(np.array(controlled_temp_profile),axis=0)
        controlled_profile_coords = np.mean(np.array(controlled_profile_coords),axis=0)
    
    
    ## LOAD UNCONTROLLED SIMULATION DATA ##
    ## ====================================================================================================================================================================================================== ##
    if combined_path == "":
        print("Loading uncontrolled simulation data...")
        num_uncontrolled = 0
        uncontrolled_time = []
        uncontrolled_mesh_x_y0_z0 = []
        uncontrolled_mesh_width = []
        uncontrolled_target = []
        uncontrolled_front_speed = []
        uncontrolled_max_cum_temp = []
        uncontrolled_global_fine_mesh_x = []
        uncontrolled_global_fine_mesh_y = []
        uncontrolled_sorted_mean_front_x_locations = []
        uncontrolled_sorted_front_temperature = []
        uncontrolled_front_width = []
        uncontrolled_front_skewness = []
        uncontrolled_power = []
        uncontrolled_energy = []
        uncontrolled_bulk_energy = []
        uncontrolled_ideal_energy = []
        uncontrolled_cure_profile = []
        uncontrolled_temp_profile = []
        uncontrolled_profile_coords = []
        
        remove_digits = str.maketrans('', '', digits)
        for sim in range(1,int(re.search(r'\d+$', uncontrolled_path[-2:]).group())+1):
            uncontrolled_path_no_digits = uncontrolled_path.translate(remove_digits)
            print(uncontrolled_path_no_digits+str(sim)+"/output")
            with open(uncontrolled_path_no_digits+str(sim)+"/output", 'rb') as file:
                dat = pickle.load(file)
                num_uncontrolled = num_uncontrolled + 1
                
                # Simulation constants
                adiabatic_rxn_temp = dat['adiabatic_rxn_temp']
                initial_temperature = dat['initial_temperature']
                specific_heat = dat['specific_heat']
                density = dat['density']
                volume = dat['volume']
                heat_transfer_coeff = dat['heat_transfer_coeff']
                surface_area = dat['surface_area']
                ambient_temp = dat['ambient_temp']
                interpolated_temp_field = dat['interpolated_temp_field']
                
                # Simulation data
                uncontrolled_time.append(dat['time'])
                uncontrolled_mesh_x_y0_z0.append(dat['mesh_x_z0'][:,0])
                uncontrolled_mesh_width.append(dat['mesh_y_z0'][-1][-1])
                uncontrolled_target.append(dat['target'])
                uncontrolled_front_speed.append(dat['front_velocity'])
                uncontrolled_max_cum_temp.append(dat['cum_max_temp_field'][-1])
                
                # Only collect large spatial data once
                if num_uncontrolled == 1:
                    uncontrolled_global_fine_mesh_x = dat['global_fine_mesh_x']
                    uncontrolled_global_fine_mesh_y = dat['global_fine_mesh_y']
                    
                # Front temperature calculations
                sorted_mean_front_x_locations = 1000.0*np.array(sorted(dat['mean_front_x_locations']))
                sorted_front_temperature = np.array([x for _, x in sorted(zip(dat['mean_front_x_locations'], dat['front_temperature']))])
                
                # Store calculated front temperature data
                uncontrolled_sorted_mean_front_x_locations.append(sorted_mean_front_x_locations)
                uncontrolled_sorted_front_temperature.append(sorted_front_temperature)
                
                # Front shape calculations
                front_width = []
                front_skewness = []
                for i in range(len(dat['time'])):
                    front_x_location = dat['front_curve'][i][0]
                    front_x_location = front_x_location[front_x_location >= 0.0]
                    front_x_location = 1000.0*front_x_location
                    front_instances = len(front_x_location)
                    if front_instances > 3:
                        front_width.append(np.max(front_x_location) - np.min(front_x_location))
                        if not np.std(front_x_location) == 0.0:
                            front_skewness.append(abs((np.mean(front_x_location)-np.median(front_x_location))/np.std(front_x_location)))
                        else:
                            front_skewness.append(0.0)
                    else:
                        front_width.append(0.0)
                        front_skewness.append(0.0)
                front_width = np.array(front_width)
                front_skewness = np.array(front_skewness)
                
                # Store calculated front shape data
                uncontrolled_front_width.append(front_width)
                uncontrolled_front_skewness.append(front_skewness)
                
                # Energy and power calculations
                mean_initial_temp = np.mean(interpolated_temp_field[0]) * (adiabatic_rxn_temp - initial_temperature) + initial_temperature
                C1 = 0.00070591
                C2 = 0.0067238
                C3 = 0.53699
                power = dat['source_power'] + dat['trigger_power']
                energy = integrate.cumtrapz(power, x=dat['time'])
                energy = np.insert(energy, 0, 0.0)
                trigger_energy = integrate.cumtrapz(dat['trigger_power'], x=dat['time'])
                trigger_energy = np.insert(trigger_energy, 0, 0.0)
                target_temp = ((np.sqrt(C2*C2 - 4.0*C1*(C3-1000.0*dat['target'][-1])) - C2) / (2.0 * C1)) + 273.15
                bulk_energy = specific_heat*density*volume*(target_temp - mean_initial_temp) + heat_transfer_coeff*surface_area*(target_temp - ambient_temp)*dat['time'] + trigger_energy
                ideal_energy = specific_heat*density*volume*(target_temp - mean_initial_temp) + trigger_energy
                
                # Store calculated energy and power data
                uncontrolled_power.append(power)
                uncontrolled_energy.append(energy)
                uncontrolled_bulk_energy.append(bulk_energy)
                uncontrolled_ideal_energy.append(ideal_energy)
    
                # Get the temperature and cure profiles at selected times after ignition
                steps = []
                if dat['time'][-1] > 8.0:
                    steps.append(np.argmin(np.abs(dat['time'] - 8.0)))
                if dat['time'][-1] > 18.0:
                    steps.append(np.argmin(np.abs(dat['time'] - 18.0)))
                if dat['time'][-1] > 28.0:
                    steps.append(np.argmin(np.abs(dat['time'] - 28.0)))
                uncontrolled_cure_profile_list = []
                uncontrolled_temp_profile_list = []
                for ind in range(len(steps)):
                    curr_step = steps[ind]
                    uncontrolled_cure_profile_list.append(dat['interpolated_cure_field'][curr_step][:,len(dat['interpolated_cure_field'][curr_step][0])//2])
                    uncontrolled_temp_profile_list.append(dat['interpolated_temp_field'][curr_step][:,len(dat['interpolated_temp_field'][curr_step][0])//2])
                    uncontrolled_profile_coords_list = dat['global_fine_mesh_x'][:,0]
                uncontrolled_cure_profile.append(uncontrolled_cure_profile_list)
                uncontrolled_temp_profile.append(uncontrolled_temp_profile_list)
                uncontrolled_profile_coords.append(uncontrolled_profile_coords_list)
                    
        # Standard deviation of uncontrolled simulation data
        std_uncontrolled_front_speed = np.std(np.array(uncontrolled_front_speed),axis=0)
        std_uncontrolled_max_cum_temp = np.std(np.array(uncontrolled_max_cum_temp),axis=0)
        std_uncontrolled_sorted_mean_front_x_locations = np.std(np.array(uncontrolled_sorted_mean_front_x_locations),axis=0)
        std_uncontrolled_sorted_front_temperature = np.std(np.array(uncontrolled_sorted_front_temperature),axis=0)
        std_uncontrolled_front_width = np.std(np.array(uncontrolled_front_width),axis=0)
        std_uncontrolled_front_skewness = np.std(np.array(uncontrolled_front_skewness),axis=0)
        std_uncontrolled_power = np.std(np.array(uncontrolled_power),axis=0)
        std_uncontrolled_energy = np.std(np.array(uncontrolled_energy),axis=0)
        std_uncontrolled_bulk_energy = np.std(np.array(uncontrolled_bulk_energy),axis=0)
        std_uncontrolled_ideal_energy = np.std(np.array(uncontrolled_ideal_energy),axis=0)
        std_uncontrolled_cure_profile = np.std(np.array(uncontrolled_cure_profile),axis=0)
        std_uncontrolled_temp_profile = np.std(np.array(uncontrolled_temp_profile),axis=0)
        std_uncontrolled_profile_coords = np.std(np.array(uncontrolled_profile_coords),axis=0)
        
        # Mean of uncontrolled simulation data
        uncontrolled_time = np.mean(np.array(uncontrolled_time),axis=0)
        uncontrolled_mesh_x_y0_z0 = np.mean(np.array(uncontrolled_mesh_x_y0_z0),axis=0)
        uncontrolled_mesh_width = np.mean(np.array(uncontrolled_mesh_width),axis=0)
        uncontrolled_target = np.mean(np.array(uncontrolled_target),axis=0)
        uncontrolled_front_speed = np.mean(np.array(uncontrolled_front_speed),axis=0)
        uncontrolled_max_cum_temp = np.mean(np.array(uncontrolled_max_cum_temp),axis=0)
        uncontrolled_sorted_mean_front_x_locations = np.mean(np.array(uncontrolled_sorted_mean_front_x_locations),axis=0)
        uncontrolled_sorted_front_temperature = np.mean(np.array(uncontrolled_sorted_front_temperature),axis=0)
        uncontrolled_front_width = np.mean(np.array(uncontrolled_front_width),axis=0)
        uncontrolled_front_skewness = np.mean(np.array(uncontrolled_front_skewness),axis=0)
        uncontrolled_power = np.mean(np.array(uncontrolled_power),axis=0)
        uncontrolled_energy = np.mean(np.array(uncontrolled_energy),axis=0)
        uncontrolled_bulk_energy = np.mean(np.array(uncontrolled_bulk_energy),axis=0)
        uncontrolled_ideal_energy = np.mean(np.array(uncontrolled_ideal_energy),axis=0)
        uncontrolled_cure_profile = np.mean(np.array(uncontrolled_cure_profile),axis=0)
        uncontrolled_temp_profile = np.mean(np.array(uncontrolled_temp_profile),axis=0)
        uncontrolled_profile_coords = np.mean(np.array(uncontrolled_profile_coords),axis=0)
    
    
    ## PLOT TRAJECTORIES ##
    ## ====================================================================================================================================================================================================== ##
    # Plot speed trajectory
    print("Plotting trajectories...")
    plt.clf()
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.title("(b) Mean front speed over " +  str(num_controlled) + " simulations",fontsize=20, y=-0.25)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Front Speed [mm/s]",fontsize=16)
    plt.fill_between(controlled_time, 1050.0*controlled_target,950.0*controlled_target,color='g',alpha=0.2,lw=0.0,label='Target')
    plt.fill_between(controlled_time, 1000.0*controlled_front_speed+500.0*std_controlled_front_speed,1000.0*controlled_front_speed-500.0*std_controlled_front_speed,color='r',alpha=0.2,lw=0.0)
    plt.plot(controlled_time, 1000.0*controlled_front_speed,c='r',lw=1.75,label='LQR')
    plt.fill_between(uncontrolled_time, 1000.0*uncontrolled_front_speed+500.0*std_uncontrolled_front_speed,1000.0*uncontrolled_front_speed-500.0*std_uncontrolled_front_speed,color='b',alpha=0.2,lw=0.0)
    plt.plot(uncontrolled_time, 1000.0*uncontrolled_front_speed,c='b',lw=1.75,label='PPO')
    plt.ylim(1.2, 1.7)
    plt.xlim(0.0, np.round(controlled_time[-1]))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper right',fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path+"/speed.svg", dpi = 500)
    plt.close()
    
    #Plot front temperature trajectory
    plt.clf()
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.title("Mean front temperature over " +  str(num_controlled) + " simulations",fontsize=20, y=-0.25)
    plt.xlabel("Location [mm]",fontsize=16)
    plt.ylabel("Front Temperature [C]",fontsize=16)
    plt.fill_between(controlled_sorted_mean_front_x_locations, controlled_sorted_front_temperature+0.5*std_controlled_sorted_front_temperature-273.15, controlled_sorted_front_temperature-0.5*std_controlled_sorted_front_temperature-273.15, color='r',alpha=0.2,lw=0.0)
    plt.plot(controlled_sorted_mean_front_x_locations, controlled_sorted_front_temperature-273.15, c='r', lw=1.75, label='LQR')
    plt.fill_between(uncontrolled_sorted_mean_front_x_locations, uncontrolled_sorted_front_temperature+0.5*std_uncontrolled_sorted_front_temperature-273.15, uncontrolled_sorted_front_temperature-0.5*std_uncontrolled_sorted_front_temperature-273.15, color='b',alpha=0.2,lw=0.0)
    plt.plot(uncontrolled_sorted_mean_front_x_locations, uncontrolled_sorted_front_temperature-273.15, c='b', lw=1.75, label='PPO')
    T1 = (uncontrolled_sorted_front_temperature-0.5*std_uncontrolled_sorted_front_temperature-273.15)
    T2 = (uncontrolled_sorted_front_temperature+0.5*std_uncontrolled_sorted_front_temperature-273.15)
    T3 = (controlled_sorted_front_temperature-0.5*std_controlled_sorted_front_temperature-273.15)
    T4 = (controlled_sorted_front_temperature+0.5*std_controlled_sorted_front_temperature-273.15)
    plt.ylim(210,230)
    plt.xlim(0.0, max(2.5*np.ceil(uncontrolled_sorted_mean_front_x_locations[-1]/2.5), 2.5*np.ceil(controlled_sorted_mean_front_x_locations[-1]/2.5)))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper right',fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path+"/temp.svg", dpi = 500)
    plt.close()
    
    
    ## PLOT FRONT SHAPE ##
    ## ====================================================================================================================================================================================================== ##
    # Set up front plot
    print("Plotting shapes...")
    # plt.cla()
    # plt.clf()
    # fig, ax1 = plt.subplots()
    # fig.set_size_inches(8.5,5.5)
    
    # # Plot skewness
    # ax1.set_xlabel("Time [s]",fontsize=16)
    # ax1.set_xlim(0.0, np.round(controlled_time[-1]))
    # ax1.set_ylabel("Nonparametric Skew of Front",fontsize=16,color='r')
    # ax1.set_ylim(0.0, ((max(np.max(uncontrolled_front_skewness+0.5*std_uncontrolled_front_skewness), np.max(controlled_front_skewness+0.5*std_controlled_front_skewness))//0.10)+1.0)*0.1)
    # ax1.fill_between(uncontrolled_time, uncontrolled_front_skewness+0.5*std_uncontrolled_front_skewness, uncontrolled_front_skewness-0.5*std_uncontrolled_front_skewness, color='magenta',alpha=0.2,lw=0.0)
    # plot1 = ax1.plot(uncontrolled_time,uncontrolled_front_skewness,lw=1.0,c='magenta',label="Open-Loop Skew")
    # ax1.fill_between(controlled_time, controlled_front_skewness+0.5*std_controlled_front_skewness, controlled_front_skewness-0.5*std_controlled_front_skewness, color='r',alpha=0.2,lw=0.0)
    # plot2 = ax1.plot(controlled_time,controlled_front_skewness,lw=1.0,c='r',label="Closed-Loop Skew")
    # ax1.tick_params(axis='x', labelsize=16)
    # ax1.tick_params(axis='y', labelsize=16, labelcolor='r')
    
    # # Plot front width
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("Normalized Front Width",fontsize=16,color='b')
    # ax2.set_ylim(0.0, ((max(np.max((uncontrolled_front_width+0.5*std_uncontrolled_front_width)/(1000.0*uncontrolled_mesh_width)), np.max((controlled_front_width+0.5*std_controlled_front_width)/(1000.0*controlled_mesh_width)))//0.10)+1.0)*0.1)
    # ax2.fill_between(uncontrolled_time, (uncontrolled_front_width+0.5*std_uncontrolled_front_width)/(1000.0*uncontrolled_mesh_width), (uncontrolled_front_width-0.5*std_uncontrolled_front_width)/(1000.0*uncontrolled_mesh_width), color='cyan',alpha=0.2,lw=0.0)
    # plot3 = ax2.plot(uncontrolled_time,uncontrolled_front_width / (1000.0*uncontrolled_mesh_width),lw=1.0,c='cyan',label="Open-Loop Width")
    # ax2.fill_between(controlled_time, (controlled_front_width+0.5*std_controlled_front_width)/(1000.0*controlled_mesh_width), (controlled_front_width-0.5*std_controlled_front_width)/(1000.0*controlled_mesh_width), color='b',alpha=0.2,lw=0.0)
    # plot4 = ax2.plot(controlled_time,controlled_front_width / (1000.0*controlled_mesh_width),lw=1.0,c='b',label="Closed-Loop Width")
    # ax2.tick_params(axis='x', labelsize=16)
    # ax2.tick_params(axis='y', labelsize=16, labelcolor='b')                
    # fig.suptitle("Mean Front Shape, n = " + str(num_controlled),fontsize=20)
    # plots = plot2+plot4+plot1+plot3
    # labels=[plot.get_label() for plot in plots]
    # ax1.legend(plots, labels, loc='upper left',fontsize=14)
    # plt.tight_layout()
    # plt.savefig(save_path+"/front_shape.svg", dpi = 500)
    # plt.close()
    
    # Plot front width
    plt.clf()
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.title("(d) Mean front width over " +  str(num_controlled) + " simulations",fontsize=20, y=-0.25)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Normalized Front Width [-]",fontsize=16)
    plt.fill_between(controlled_time, (controlled_front_width+0.5*std_controlled_front_width)/(1000.0*controlled_mesh_width), (controlled_front_width-0.5*std_controlled_front_width)/(1000.0*controlled_mesh_width), color='r',alpha=0.2,lw=0.0)
    plt.plot(controlled_time,controlled_front_width / (1000.0*controlled_mesh_width),lw=1.75,c='r',label="LQR")
    plt.fill_between(uncontrolled_time, (uncontrolled_front_width+0.5*std_uncontrolled_front_width)/(1000.0*uncontrolled_mesh_width), (uncontrolled_front_width-0.5*std_uncontrolled_front_width)/(1000.0*uncontrolled_mesh_width), color='b',alpha=0.2,lw=0.0)
    plt.plot(uncontrolled_time,uncontrolled_front_width / (1000.0*uncontrolled_mesh_width),lw=1.75,c='b',label="PPO")
    plt.ylim(0.0, 0.10*np.ceil(max(np.max((uncontrolled_front_width+0.5*std_uncontrolled_front_width)/(1000.0*uncontrolled_mesh_width)), np.max((controlled_front_width+0.5*std_controlled_front_width)/(1000.0*controlled_mesh_width)))/0.10))
    plt.xlim(0.0, np.round(controlled_time[-1]))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left',fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path+"/front_shape.svg", dpi = 500)
    plt.close()
    
    
    ## PLOT ENERGIES ##
    ## ====================================================================================================================================================================================================== ##
    print("Plotting energies...")
    # plt.clf()
    # fig, ax1 = plt.subplots()
    # fig.set_size_inches(8.5,5.5)
    # ax1.set_xlabel("Time [s]",fontsize='large')
    # ax1.set_ylabel("Power [W]",fontsize='large',color='b')
    # ax1.fill_between(controlled_time, controlled_power+0.5*std_controlled_power, controlled_power-0.5*std_controlled_power, color='b',alpha=0.2,lw=0.0)
    # ax1.plot(controlled_time, controlled_power,c='b',lw=1.0)
    # ax1.set_xlim(0.0, np.round(controlled_time[-1]))
    # ax1.set_ylim((np.min(controlled_power-0.5*std_controlled_power)//0.1-1.0)*0.1, (np.max(controlled_power+0.5*std_controlled_power)//0.1+1.0)*0.1)
    # ax1.tick_params(axis='x', labelsize=12)
    # ax1.tick_params(axis='y', labelsize=12, labelcolor='b')
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("Cumulative Energy Consumed [J]",fontsize='large',color='r')
    # ax2.fill_between(controlled_time, controlled_energy+0.5*std_controlled_energy, controlled_energy-0.5*std_controlled_energy, color='r',alpha=0.2,lw=0.0)
    # ax2.plot(controlled_time, controlled_energy,c='r',lw=1.0)
    # ax2.plot(controlled_time, controlled_bulk_energy,c='r',lw=1.0,ls='--',label='Bulk Heating')
    # ax2.plot(controlled_time, controlled_ideal_energy,c='r',lw=1.0,ls=':',label='Ideal Local Heating')
    # ax2.set_xlim(0.0, np.round(controlled_time[-1]))
    # ax2.set_ylim(0.0, (max(np.max(controlled_energy+0.5*std_controlled_energy), np.max(controlled_bulk_energy), np.max(controlled_ideal_energy))//1.0+1.0)*1.0)
    # ax2.tick_params(axis='x', labelsize=12)
    # ax2.tick_params(axis='y', labelsize=12, labelcolor='r')
    # fig.suptitle("Mean External Energy Input, n = " + str(num_controlled),fontsize='xx-large')
    # plt.legend(loc='upper left',fontsize='medium')
    # plt.tight_layout()
    # plt.savefig(save_path+"/energy.svg", dpi = 500)
    # plt.close()
    
    # Plot energies
    plt.clf()
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.title("Mean energy consumption over " +  str(num_controlled) + " simulations",fontsize=20, y=-0.25)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Cumulative Energy Use [J]",fontsize=16)
    plt.fill_between(controlled_time, controlled_energy+0.5*std_controlled_energy, controlled_energy-0.5*std_controlled_energy, color='r',alpha=0.2,lw=0.0)
    plt.plot(controlled_time, controlled_energy, lw=1.75,c='r',label="LQR")
    plt.plot(uncontrolled_time, uncontrolled_energy, lw=1.75,c='b',label="PPO")
    plt.plot(controlled_time, controlled_bulk_energy, lw=1.75,c='k',label="Bulk Heating")
    plt.ylim(np.floor(min(np.min(controlled_energy), np.min(uncontrolled_energy), np.min(controlled_bulk_energy))), np.ceil(max(np.max(controlled_energy), np.max(uncontrolled_energy), np.max(controlled_bulk_energy))))
    plt.xlim(0.0, np.round(controlled_time[-1]))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left',fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path+"/energy.svg", dpi = 500)
    plt.close()
    
    
    ## PLOT PROFILES ##
    ## ====================================================================================================================================================================================================== ##
    print("Plotting profiles...")
    
    # Calculate x bounds
    temp = abs(np.diff(controlled_temp_profile)) > 0.001
    temp = np.logical_or(np.logical_or(temp[0,:], temp[1,:]), temp[2,:])
    inds = np.arange(len(temp))
    max_x_value = controlled_profile_coords[np.max(inds[temp])+int(np.round(0.10*inds[-1]))]
    
    # Plot temperature profiles
    plt.clf()
    plt.gcf().set_size_inches(8.5,5.5)
    plt.xlabel("Location [mm]",fontsize=16)
    plt.ylabel("θ [-]",fontsize=16)
    for i in range(len(controlled_temp_profile)):
        plt.fill_between(1000.0*controlled_profile_coords, controlled_temp_profile[i]+0.5*std_controlled_temp_profile[i], controlled_temp_profile[i]-0.5*std_controlled_temp_profile[i], color='r',alpha=0.2,lw=0.0)
        if i == 0:
            plt.plot(1000.0*controlled_profile_coords, controlled_temp_profile[i],c='r',lw=1.75,label="LQR")
        else:
            plt.plot(1000.0*controlled_profile_coords, controlled_temp_profile[i],c='r',lw=1.75)
        plt.fill_between(1000.0*uncontrolled_profile_coords, uncontrolled_temp_profile[i]+0.5*std_uncontrolled_temp_profile[i], uncontrolled_temp_profile[i]-0.5*std_uncontrolled_temp_profile[i], color='b',alpha=0.2,lw=0.0)
        if i == 0:
            plt.plot(1000.0*uncontrolled_profile_coords, uncontrolled_temp_profile[i],c='b',lw=1.75,label="PPO")
        else:
            plt.plot(1000.0*uncontrolled_profile_coords, uncontrolled_temp_profile[i],c='b',lw=1.75)
    plt.xlim(0.0, 1000.0*max_x_value)
    plt.ylim((min(np.min(controlled_temp_profile-0.5*std_controlled_temp_profile), np.min(uncontrolled_temp_profile-0.5*std_uncontrolled_temp_profile)) // 0.10) * 0.10, (max(np.max(controlled_temp_profile+0.5*std_controlled_temp_profile), np.max(uncontrolled_temp_profile+0.5*std_uncontrolled_temp_profile)) // 0.10 + 1) * 0.10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Mean temperature profiles over " +  str(num_controlled) + " simulations",fontsize=20, y=-0.25)
    plt.legend(loc='upper right',fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path+"/temp_profiles.svg", dpi = 500)
    plt.close()
    
    # Plot cure profiles
    plt.clf()
    plt.gcf().set_size_inches(8.5,5.5)
    plt.xlabel("Location [mm]",fontsize=16)
    plt.ylabel("Degree of Cure (α) [-]",fontsize=16)
    for i in range(len(controlled_cure_profile)):
        plt.fill_between(1000.0*controlled_profile_coords, controlled_cure_profile[i]+0.5*std_controlled_cure_profile[i], controlled_cure_profile[i]-0.5*std_controlled_cure_profile[i], color='r',alpha=0.2,lw=0.0)
        if i == 0:
            plt.plot(1000.0*controlled_profile_coords, controlled_cure_profile[i],c='r',lw=1.75,label="LQR")
        else:
            plt.plot(1000.0*controlled_profile_coords, controlled_cure_profile[i],c='r',lw=1.75)
        plt.fill_between(1000.0*uncontrolled_profile_coords, uncontrolled_cure_profile[i]+0.5*std_uncontrolled_cure_profile[i], uncontrolled_cure_profile[i]-0.5*std_uncontrolled_cure_profile[i], color='b',alpha=0.2,lw=0.0)
        if i == 0:
            plt.plot(1000.0*uncontrolled_profile_coords, uncontrolled_cure_profile[i],c='b',lw=1.75,label="PPO")
        else:
            plt.plot(1000.0*uncontrolled_profile_coords, uncontrolled_cure_profile[i],c='b',lw=1.75)
    plt.xlim(0.0, 1000.0*max_x_value)
    plt.ylim(0.0, 1.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("(c) Mean cure profiles over " +  str(num_controlled) + " simulations",fontsize=20, y=-0.25)
    plt.legend(loc='upper right',fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path+"/cure_profiles.svg", dpi = 500)
    plt.close()
    
    
    ## PLOT CUMULATIVE MAXIMUM TEMPERATURE FIELDS ##
    ## ====================================================================================================================================================================================================== ##
    print("Plotting cumulative temperature fields...")
    min_temp = min(np.mean(controlled_max_cum_temp) - np.mean(controlled_max_cum_temp) % 0.05, np.mean(uncontrolled_max_cum_temp) - np.mean(uncontrolled_max_cum_temp) % 0.05)
    max_temp_1 = round(np.max(controlled_max_cum_temp) + 0.05 - (np.max(controlled_max_cum_temp) % 0.05),2)
    max_temp_1 = min(min_temp + 2*(np.std(controlled_max_cum_temp) - np.std(controlled_max_cum_temp) % 0.05), max_temp_1)
    max_temp_2 = round(np.max(uncontrolled_max_cum_temp) + 0.05 - (np.max(uncontrolled_max_cum_temp) % 0.05),2)
    max_temp_2 = min(min_temp + 2*(np.std(uncontrolled_max_cum_temp) - np.std(uncontrolled_max_cum_temp) % 0.05), max_temp_2)
    max_temp = max(max_temp_1, max_temp_2)
    
    min_temp = 0.8
    max_temp = 1.0
    
    plt.clf()
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=21, left=0.025, right=0.925, bottom=0.1, top=0.95, hspace=0.2, wspace=0.5)
    
    ax0 = fig.add_subplot(gs[0, 0:10])
    ax1 = fig.add_subplot(gs[0, 10:20])
    cax1 = fig.add_subplot(gs[0, -1])
    ax2 = fig.add_subplot(gs[1, 0:10])
    ax3 = fig.add_subplot(gs[1, 10:20])
    cax3 = fig.add_subplot(gs[1, -1])
    fig.set_size_inches(16.5,5.5)
    
    ax0.pcolormesh(1000.0*controlled_global_fine_mesh_x, 1000.0*controlled_global_fine_mesh_y, controlled_max_cum_temp, shading='gouraud', cmap='jet', vmin=min_temp, vmax=max_temp)
    ax0.set_axis_off()
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_title('Mean LQR',fontsize=16)
    
    c1 = ax1.pcolormesh(1000.0*uncontrolled_global_fine_mesh_x, 1000.0*uncontrolled_global_fine_mesh_y, uncontrolled_max_cum_temp, shading='gouraud', cmap='jet', vmin=min_temp, vmax=max_temp)
    cbar1 = fig.colorbar(c1, cax=cax1, ticks=[min_temp, max_temp], format=lambda x, _: f"{x:.1f}")
    cbar1.set_label("Mean θ [-]",labelpad=8,fontsize=16)
    cbar1.ax.tick_params(labelsize=16)
    ax1.set_axis_off()
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('Mean PPO',fontsize=16)
    
    min_temp = min(np.mean(std_controlled_max_cum_temp) - np.mean(std_controlled_max_cum_temp) % 0.05, np.mean(std_uncontrolled_max_cum_temp) - np.mean(std_uncontrolled_max_cum_temp) % 0.05)
    max_temp_1 = round(np.max(std_controlled_max_cum_temp) + 0.05 - (np.max(std_controlled_max_cum_temp) % 0.05),2)
    max_temp_1 = min(min_temp + 2*(np.std(std_controlled_max_cum_temp) - np.std(std_controlled_max_cum_temp) % 0.05), max_temp_1)
    max_temp_2 = round(np.max(std_uncontrolled_max_cum_temp) + 0.05 - (np.max(std_uncontrolled_max_cum_temp) % 0.05),2)
    max_temp_2 = min(min_temp + 2*(np.std(std_uncontrolled_max_cum_temp) - np.std(std_uncontrolled_max_cum_temp) % 0.05), max_temp_2)
    max_temp = max(max_temp_1, max_temp_2)
    
    min_temp = 0.0
    max_temp = 0.05
    
    ax2.pcolormesh(1000.0*controlled_global_fine_mesh_x, 1000.0*controlled_global_fine_mesh_y, std_controlled_max_cum_temp, shading='gouraud', cmap='jet', vmin=min_temp, vmax=max_temp)
    ax2.set_axis_off()
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title('Standard deviation LQR',fontsize=16)
    
    c3 = ax3.pcolormesh(1000.0*uncontrolled_global_fine_mesh_x, 1000.0*uncontrolled_global_fine_mesh_y, std_uncontrolled_max_cum_temp, shading='gouraud', cmap='jet', vmin=min_temp, vmax=max_temp)
    cbar3 = fig.colorbar(c3, cax=cax3, ticks=[min_temp, max_temp], format=lambda x, _: f"{x:.1f}")
    cbar3.set_label("Stdev. θ [-]",labelpad=8,fontsize=16)
    cbar3.ax.tick_params(labelsize=16)
    ax3.set_axis_off()
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title('Standard deviation PPO',fontsize=16)
    
    fig.suptitle("(e) Mean maximum cumulative temperature over " + str(num_controlled) + " simulations",fontsize=20, y=0.05)
    plt.savefig(save_path + "/max_cum_temp.png", dpi = 500)
    plt.close()
    
    
    ## SAVE ALL DATA USED TO GENERATE GRAPHS ##
    ## ====================================================================================================================================================================================================== ##
    print("Saving...")
    data = {
        'std_controlled_front_speed' : std_controlled_front_speed,
        'std_controlled_max_cum_temp' : std_controlled_max_cum_temp,
        'std_controlled_sorted_mean_front_x_locations' : std_controlled_sorted_mean_front_x_locations,
        'std_controlled_sorted_front_temperature' : std_controlled_sorted_front_temperature,
        'std_controlled_front_width' : std_controlled_front_width,
        'std_controlled_front_skewness' : std_controlled_front_skewness,
        'std_controlled_power' : std_controlled_power,
        'std_controlled_energy' : std_controlled_energy,
        'std_controlled_bulk_energy' : std_controlled_bulk_energy,
        'std_controlled_ideal_energy' : std_controlled_ideal_energy,
        'std_controlled_cure_profile' : std_controlled_cure_profile,
        'std_controlled_temp_profile' : std_controlled_temp_profile,
        'std_controlled_profile_coords' : std_controlled_profile_coords,
        'controlled_time' : controlled_time,
        'controlled_global_fine_mesh_x' : controlled_global_fine_mesh_x,
        'controlled_global_fine_mesh_y' : controlled_global_fine_mesh_y,
        'controlled_mesh_x_y0_z0' : controlled_mesh_x_y0_z0,
        'controlled_mesh_width' : controlled_mesh_width,
        'controlled_target' : controlled_target,
        'controlled_front_speed' : controlled_front_speed,
        'controlled_max_cum_temp' : controlled_max_cum_temp,
        'controlled_sorted_mean_front_x_locations' : controlled_sorted_mean_front_x_locations,
        'controlled_sorted_front_temperature' : controlled_sorted_front_temperature,
        'controlled_front_width' : controlled_front_width,
        'controlled_front_skewness' : controlled_front_skewness,
        'controlled_power' : controlled_power,
        'controlled_energy' : controlled_energy,
        'controlled_bulk_energy' : controlled_bulk_energy,
        'controlled_ideal_energy' : controlled_ideal_energy,
        'controlled_cure_profile' : controlled_cure_profile,
        'controlled_temp_profile' : controlled_temp_profile,
        'controlled_profile_coords' : controlled_profile_coords,
        'std_uncontrolled_front_speed' : std_uncontrolled_front_speed,
        'std_uncontrolled_max_cum_temp' : std_uncontrolled_max_cum_temp,
        'std_uncontrolled_sorted_mean_front_x_locations' : std_uncontrolled_sorted_mean_front_x_locations,
        'std_uncontrolled_sorted_front_temperature' : std_uncontrolled_sorted_front_temperature,
        'std_uncontrolled_front_width' : std_uncontrolled_front_width,
        'std_uncontrolled_front_skewness' : std_uncontrolled_front_skewness,
        'std_uncontrolled_power' : std_uncontrolled_power,
        'std_uncontrolled_energy' : std_uncontrolled_energy,
        'std_uncontrolled_bulk_energy' : std_uncontrolled_bulk_energy,
        'std_uncontrolled_ideal_energy' : std_uncontrolled_ideal_energy,
        'std_uncontrolled_cure_profile' : std_uncontrolled_cure_profile,
        'std_uncontrolled_temp_profile' : std_uncontrolled_temp_profile,
        'std_uncontrolled_profile_coords' : std_uncontrolled_profile_coords,
        'uncontrolled_time' : uncontrolled_time,
        'uncontrolled_global_fine_mesh_x' : uncontrolled_global_fine_mesh_x,
        'uncontrolled_global_fine_mesh_y' : uncontrolled_global_fine_mesh_y,
        'uncontrolled_mesh_x_y0_z0' : uncontrolled_mesh_x_y0_z0,
        'uncontrolled_mesh_width' : uncontrolled_mesh_width,
        'uncontrolled_target' : uncontrolled_target,
        'uncontrolled_front_speed' : uncontrolled_front_speed,
        'uncontrolled_max_cum_temp' : uncontrolled_max_cum_temp,
        'uncontrolled_sorted_mean_front_x_locations' : uncontrolled_sorted_mean_front_x_locations,
        'uncontrolled_sorted_front_temperature' : uncontrolled_sorted_front_temperature,
        'uncontrolled_front_width' : uncontrolled_front_width,
        'uncontrolled_front_skewness' : uncontrolled_front_skewness,
        'uncontrolled_power' : uncontrolled_power,
        'uncontrolled_energy' : uncontrolled_energy,
        'uncontrolled_bulk_energy' : uncontrolled_bulk_energy,
        'uncontrolled_ideal_energy' : uncontrolled_ideal_energy,
        'uncontrolled_cure_profile' : uncontrolled_cure_profile,
        'uncontrolled_temp_profile' : uncontrolled_temp_profile,
        'uncontrolled_profile_coords' : uncontrolled_profile_coords,
        'num_controlled' : num_controlled
    }
    with open(save_path + "/combined_data", 'wb') as file:
            pickle.dump(data, file)
            
    print("Done!")