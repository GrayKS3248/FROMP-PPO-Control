# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:53:17 2021

@author: GKSch
"""
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    # Define load paths
    paths = [
        '../results/tune_log.dat',
        ]
    
    # Load log files
    files = []
    for i in range(len(paths)):
        read_file = open(paths[i],'r')
        files.append(read_file.readlines())
        read_file.close()
        
    # Read the log files
    names = []
    units = []
    data = []
    for i in range(len(files)):
        names.append(files[i][27].replace("\t ",",").replace("\n","").split(","))
        units.append(files[i][28].replace("\t ",",").replace("\n","").replace("---","-").split(","))
        data.append([])
        
        for j in range(30, len(files[i])):
            data[i].append(files[i][j].replace("\t",",").replace("\n","").split(","))
    data = np.array(data)
                    
    # Plot data
    for i in range(len(paths)):
        for j in range(1, len(names[i])):
            plt.clf()
            plt.title(names[i][j], fontsize='xx-large')
            plt.xlabel("Num Updates [-]",fontsize='large')
            plt.ylabel(names[i][j] + " [" + units[i][j] + "]",fontsize='large')
            plt.plot(np.int32(data[i,:,0]), np.double(data[i,:,j]), c='r', lw=2.5)
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(7.5, 5)
            save_path = paths[i][0:paths[i].rfind('/')+1] + names[i][j].replace(" ","_") + ".png"
            plt.savefig(save_path, dpi = 500)
            plt.close()