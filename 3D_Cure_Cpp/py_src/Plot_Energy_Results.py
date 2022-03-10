# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:53:17 2021

@author: GKSch
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pickle
from scipy import integrate
from scipy.optimize import minimize_scalar

if __name__ == "__main__":
    
    
    ## INPUTS ##
    ## ====================================================================================================================================================================================================== ##
    # Define load and save paths
    path = "../results/Energy" 
    files = [["1.2_1", "1.2_2", "1.2_3", "1.2_4", "1.2_5", "1.2_6", "1.2_7", "1.2_8", "1.2_9", "1.2_10", "1.2_11", "1.2_12", "1.2_13", "1.2_14", "1.2_15", "1.2_16", "1.2_17", "1.2_18", "1.2_19", "1.2_20"], 
             ["1.3_1", "1.3_2", "1.3_3", "1.3_4", "1.3_5", "1.3_6", "1.3_7", "1.3_8", "1.3_9", "1.3_10", "1.3_11", "1.3_12", "1.3_13", "1.3_14", "1.3_15", "1.3_16", "1.3_17", "1.3_18", "1.3_19", "1.3_20"], 
             ["1.4_1", "1.4_2", "1.4_3", "1.4_4", "1.4_5", "1.4_6", "1.4_7", "1.4_8", "1.4_9", "1.4_10", "1.4_11", "1.4_12", "1.4_13", "1.4_14", "1.4_15", "1.4_16", "1.4_17", "1.4_18", "1.4_19", "1.4_20"], 
             ["1.5_1", "1.5_2", "1.5_3", "1.5_4", "1.5_5", "1.5_6", "1.5_7", "1.5_8", "1.5_9", "1.5_10", "1.5_11", "1.5_12", "1.5_13", "1.5_14", "1.5_15", "1.5_16", "1.5_17", "1.5_18", "1.5_19", "1.5_20"], 
             ["1.6_1", "1.6_2", "1.6_3", "1.6_4", "1.6_5", "1.6_6", "1.6_7", "1.6_8", "1.6_9", "1.6_10", "1.6_11", "1.6_12", "1.6_13", "1.6_14", "1.6_15", "1.6_16", "1.6_17", "1.6_18", "1.6_19", "1.6_20"],
             ["1.7_1", "1.7_2", "1.7_3", "1.7_4", "1.7_5", "1.7_6", "1.7_7", "1.7_8", "1.7_9", "1.7_10", "1.7_11", "1.7_12", "1.7_13", "1.7_14", "1.7_15", "1.7_16", "1.7_17", "1.7_18", "1.7_19", "1.7_20"]]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    labels = ["1.2 mm/s", "1.3 mm/s", "1.4 mm/s", "1.5 mm/s", "1.6 mm/s", "1.7 mm/s"]
        
    ## DEFINE HELPER FUNCTIONS ##
    ## ====================================================================================================================================================================================================== ##
    # Load all previous simulations, store their trajectory, front shape, and energy consumption
    def get_err(T, a0, v_target):
        return abs(v_target - (0.813723-5.12462*a0+0.0202144*T+9.54447*a0*a0-0.1646*a0*T+0.000808146*T*T+0.463609*a0*a0*T-0.00295865*a0*T*T))
    
    def get_speed(T0, a0):
        return (0.813723-5.12462*a0+0.0202144*T0+9.54447*a0*a0-0.1646*a0*T0+0.000808146*T0*T0+0.463609*a0*a0*T0-0.00295865*a0*T0*T0)
    
    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.
    
        Parameters
        ----------
        x, y : array-like, shape (n, )
            Input data.
    
        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.
    
        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.
    
        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`
    
        Returns
        -------
        matplotlib.patches.Ellipse
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
    
        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)
    
        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
    
        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
    
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
    
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    ## LOAD CONTROLLED SIMULATION DATA ##
    ## ====================================================================================================================================================================================================== ##
    # Load all previous simulations, store their trajectory, front shape, and energy consumption
    print("Loading simulation data...")

    ind = -1
    T0_list = []
    a0_list = []
    time_list = []
    front_speed_list = []
    target_list = []
    energy_list = []
    required_energy_list = []
    ideal_energy_list = []

    for target in files:
        ind = ind + 1
        T0_list.append([])
        a0_list.append([])
        time_list.append([])
        front_speed_list.append([])
        target_list.append([])
        energy_list.append([])
        required_energy_list.append([])
        ideal_energy_list.append([])
        
        for file in target:
            with open(path+"/"+file+"/output", 'rb') as load:
                print(path+"/"+file+"/output")
                dat = pickle.load(load)
                
                # Store time and frotn speed data
                time_list[ind].append(dat['time'])
                front_speed_list[ind].append(dat['front_velocity'])
                
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
                interpolated_cure_field = dat['interpolated_cure_field']
                
                # Energy and power calculations
                T0 = np.mean(interpolated_temp_field[0]) * (adiabatic_rxn_temp - initial_temperature) + initial_temperature
                a0 = np.mean(interpolated_cure_field[0])
                T0_list[ind].append(T0)
                a0_list[ind].append(a0)
                v_target = 1000.0*np.mean(dat['target'])
                target_temp = minimize_scalar(get_err, args=(a0, v_target), bounds=(T0-273.15, 2.0*(T0-273.15)), method='bounded', options={'xatol':0.001}).x + 273.15
                v_actual = 1000.0*np.mean(dat['front_velocity'][np.argmin(abs(dat['time']-5.0)):np.argmin(abs(dat['time']-25.0))])
                actual_temp = minimize_scalar(get_err, args=(a0, v_actual), bounds=(T0-273.15, 2.0*(T0-273.15)), method='bounded', options={'xatol':0.001}).x + 273.15
                power = dat['source_power'] + dat['trigger_power']
                energy = integrate.trapz(power, x=dat['time'])
                trigger_energy = integrate.trapz(dat['trigger_power'], x=dat['time'])
                required_energy = specific_heat*density*volume*(actual_temp - T0) + heat_transfer_coeff*surface_area*(actual_temp - ambient_temp)*(dat['time'][-1]-dat['time'][0]) + trigger_energy
                ideal_energy = specific_heat*density*volume*(actual_temp - T0) + trigger_energy
                
                # Store calculated energy and power data
                target_list[ind].append(v_target)
                energy_list[ind].append(energy)
                required_energy_list[ind].append(required_energy)
                ideal_energy_list[ind].append(ideal_energy)
    
    # Time data calculation and formatting
    time = np.mean(np.array(time_list),axis=(0,1))
    
    # Energy data calculation and formatting
    energy = np.array(energy_list)
    required_energy = np.array(required_energy_list)
    ideal_energy = np.array(ideal_energy_list)
    energy_reduction = 100.0*(required_energy - energy) / required_energy
    ideal_energy_reduction = 100.0*(required_energy - ideal_energy) / required_energy
    efficiency = energy_reduction/ideal_energy_reduction
    
    # Time reduction data calculation and formatting
    a0 = np.array(a0_list)
    T0 = np.array(T0_list)
    open_loop_speed = get_speed(T0-273.15, a0)
    closed_loop_speed = 1000.0*np.mean(np.array(front_speed_list)[:,:,np.argmin(abs(time-5.0)):np.argmin(abs(time-25.0))], axis=2)
    time_reduction = 100.0*open_loop_speed * (open_loop_speed**-1 - closed_loop_speed**-1)
    ideal_time_reduction = 100.0*open_loop_speed * (open_loop_speed**-1 - np.array(target_list)**-1)
    
    # Front speed data calculation and formatting
    mean_front_speed = np.mean(np.array(front_speed_list),axis=1)
    std_front_speed = np.std(np.array(front_speed_list),axis=1)
    targets = np.array(target_list)
    mean_targets = np.mean(np.array(target_list),axis=1)
    mean_front_speed_err = np.mean(closed_loop_speed - targets,axis=1)
    std_front_speed_err = np.std(closed_loop_speed - targets,axis=1)
    
    
    ## PLOT TRAJECTORIES ##
    ## ====================================================================================================================================================================================================== ##
    # Plot speed trajectory
    print("Plotting trajectories...")
    plt.clf()
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.title("Mean Controlled Front Speed, n = " + str(len(files[0])),fontsize='xx-large')
    plt.xlabel("Time [s]",fontsize='large')
    plt.ylabel("Front Speed [mm/s]",fontsize='large')
    
    for i in range(len(mean_front_speed)):
        plt.fill_between(time, 1000.0*mean_front_speed[i]+500.0*std_front_speed[i],1000.0*mean_front_speed[i]-500.0*std_front_speed[i],color=colors[i],alpha=0.2,lw=0.0)
        plt.plot(time, 1000.0*mean_front_speed[i],c=colors[i], lw=1.0, label=labels[i])
    plt.ylim(1.0, (np.max(1000.0*mean_front_speed+500.0*std_front_speed)//0.10+1.0)*0.10)
    plt.xlim(0.0, np.round(time[-1]))
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.legend(loc='upper left',fontsize='medium')
    plt.tight_layout()
    plt.savefig("../results/speed.svg", dpi = 500)
    plt.close()
    
    # Plot speed aherence
    plt.clf()
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.title("Controlled Front Speed Error, n = " + str(len(files[0])),fontsize='xx-large')
    plt.xlabel("Target Front Speed [mm/s]",fontsize='large')
    plt.ylabel("Absolute Error [mm/s]",fontsize='large')
    plt.errorbar(mean_targets, mean_front_speed_err, yerr=std_front_speed_err, c='r',lw=2.0,elinewidth=2.0,marker="o",ms=5.0,capsize=4.0)
    plt.axhline(y=0, c='k', lw=0.75)
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.tight_layout()
    plt.savefig("../results/speed_error.svg", dpi = 500)
    plt.close()
    
    ## PLOT ENERGIES ##
    ## ====================================================================================================================================================================================================== ##
    print("Plotting energies...")
    plt.clf()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_title("Mean Energy Consumption, n = " + str(len(files[0])),fontsize='xx-large')
    ax.set_xlabel("Front Speed [mm/s]",fontsize='large')
    ax.set_ylabel("Energy Consumed [J]",fontsize='large')
    ax.plot(np.mean(closed_loop_speed,axis=1), np.mean(required_energy,axis=1),c='r',lw=2.0,label='Bulk Heating')
    ax.plot(np.mean(closed_loop_speed,axis=1), np.mean(ideal_energy,axis=1),c='b',lw=2.0,label='Ideal Local Heating')
    for i in range(len(closed_loop_speed)):
        if i==0:
            ax.scatter(closed_loop_speed[i], energy[i], c='k', marker=".", s=2.0, label='Actual Local Heating')
        else:
            ax.scatter(closed_loop_speed[i], energy[i], c='k', marker=".", s=2.0)
        confidence_ellipse(closed_loop_speed[i], energy[i], ax, n_std=3, edgecolor='k')
    ax.tick_params(axis='x',labelsize=12)
    ax.tick_params(axis='y',labelsize=12)
    ax.legend(loc='upper left',fontsize='medium')
    plt.tight_layout()
    plt.savefig("../results/energy.svg", dpi = 500)
    plt.close()
    
    # Plot energy savings as function of cure time reduction
    plt.clf()
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8.5,5.5)
    ax1.set_xlabel("Cure Time Reduction Compared to Open-Loop [%]",fontsize='large')
    ax1.set_ylabel("Energy Reduction Compared to Bulk Heating [%]",fontsize='large', c='r')
    ax1.plot(np.mean(ideal_time_reduction,axis=1), np.mean(ideal_energy_reduction,axis=1), c='r', lw=2.0, ls='--', label="Ideal Reduction")
    for i in range(len(time_reduction)):
        if i==0:
            ax1.scatter(time_reduction[i], energy_reduction[i], c='r', marker=".", s=2.0, label='Closed-Loop Reduction')
        else:
            ax1.scatter(time_reduction[i], energy_reduction[i], c='r', marker=".", s=2.0)
        confidence_ellipse(time_reduction[i], energy_reduction[i], ax1, n_std=3, edgecolor='r')
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12, labelcolor='r')
    ax1.set_ylim(0, 70)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Efficiency [-]",fontsize='large', c='b')
    for i in range(len(time_reduction)):
        if i==0:
            ax2.scatter(time_reduction[i], efficiency[i], c='b', marker=".", s=2.0, label='Closed-Loop Reduction')
        else:
            ax2.scatter(time_reduction[i], efficiency[i], c='b', marker=".", s=2.0)
        confidence_ellipse(time_reduction[i], efficiency[i], ax2, n_std=3, edgecolor='b')
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12, labelcolor='b')
    ax2.set_ylim(0.0, 0.8)
    fig.suptitle("Mean Energy Reduction, n = " + str(len(files[0])),fontsize='xx-large')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',fontsize='medium')
    plt.tight_layout()
    plt.savefig("../results/energy_saving.svg", dpi = 500)
    plt.close()
    
    print("Done!")