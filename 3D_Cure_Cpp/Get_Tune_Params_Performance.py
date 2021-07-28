# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:53:17 2021

@author: GKSch
"""
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    
    # Define load paths
    paths = [
        'config_files/DCPD_GC2/log.dat', 
        ]
    
    # Define save path
    save_path = "config_files/DCPD_GC2"
    
    # Load log files
    files = []
    for i in range(len(paths)):
        read_file = open(paths[i],'r')
        files.append(read_file.readlines())
        read_file.close()
        
    # Define the strings we are looking for
    start_sub_strs = [
        "Num Updates: ",
        "Fine X Len: ",
        "X Step Mult: ",
        "Time Step: ",
        "Time Step Mult: ",
        "Trans Cure: ",
        "Avg Dur: ",
        "Max Std: ",
        "Avg Std: ",
        "Loss: ",
        ]
    end_sub_strs = [
        " | Fine X Len: ",
        " mm | X Step Mult: ",
        " | Time Step: ",
        " ms | Time Step Mult: ",
        " | Trans Cure: ",
        " | \n",
        " | Max Std: ",
        " | Avg Std: ",
        " | Loss: ",
        " |\n",
        ]
    units = [
        "[-]",
        "[mm]",
        "[-]",
        "[ms]",
        "[-]",
        "[-]",
        "[-]",
        "[-]",
        "[-]",
        "[-]",
        ]
        
    # Generate data list
    data = []
    for i in range(len(paths)):
        data.append([])
        for j in range(len(start_sub_strs)):
            data[i].append([])
    
    # Gather training data from loaded files
    for i in range(len(paths)):
        for j in range(len(files[i])):
            curr_line = files[i][j]
            for k in range(len(start_sub_strs)):
                start_index = curr_line.find(start_sub_strs[k])
                if start_index != -1:
                    start_index = start_index + len(start_sub_strs[k])
                    end_index = curr_line.find(end_sub_strs[k])
                    data[i][k].append(float(curr_line[start_index:end_index]))
                    
    # Plot data
    for i in range(len(paths)):
        for j in range(1,len(start_sub_strs)):
            plt.clf()
            plt.title(start_sub_strs[j][0:-2], fontsize='xx-large')
            plt.xlabel("Num Updates [-]",fontsize='large')
            plt.ylabel(start_sub_strs[j][0:-2] + " " + units[j],fontsize='large')
            plt.plot(data[i][0], data[i][j], c='r', lw=2.5)
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(7.5, 5)
            folder = (save_path + "/" + paths[i][paths[i].rfind('/')+1:-4]).replace(" ", "_").lower()
            if not os.path.isdir(folder):
                os.mkdir(folder)
            curr_save_path = folder + "/" + start_sub_strs[j][0:-2] + ".png"
            curr_save_path = curr_save_path.replace(" ", "_").lower()
            plt.savefig(curr_save_path, dpi = 500)
            plt.close()