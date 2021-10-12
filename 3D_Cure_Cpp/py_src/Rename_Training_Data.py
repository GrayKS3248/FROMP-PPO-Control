# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 07:39:16 2021

@author: GKSch
"""

import os
import shutil

if __name__ == "__main__":
    
    path = "../training_data/DCPD_GC2"
    directories = [
        "294_5", 
        "294_7", 
        "294_9", 
        "298_5", 
        "298_7", 
        "298_9", 
        "302_5", 
        "302_7", 
        "302_9", 
        "306_5", 
        "306_7", 
        "306_9"
        ]
    file_types = [
        "cure_data",
        "temp_data",
        "loc_data",
        "ftemp_data",
        ]
    settings_name = "__settings__"
    new_dir = "Autoencoder"
    
    if not os.path.isdir(path+"/"+new_dir):
        os.mkdir(path+"/"+new_dir)
        
    batch = []
    setting_num = 0
    for i in range(len(file_types)):
        batch.append(0)
    for directory in directories:
        with os.scandir(path=path+"/"+directory) as it:
            for entry in it:
                for file_type in range(len(file_types)):
                    if entry.name[0:len(file_types[file_type])] == file_types[file_type]:
                        shutil.copyfile(path+"/"+directory+"/"+entry.name, path+"/"+new_dir+"/"+file_types[file_type]+"_"+str(batch[file_type])+entry.name[entry.name.rfind("."):])
                        batch[file_type] = batch[file_type] + 1
                if entry.name[0:len(settings_name)] == settings_name:
                    shutil.copyfile(path+"/"+directory+"/"+entry.name, path+"/"+new_dir+"/"+settings_name+str(setting_num)+entry.name[entry.name.rfind("."):])
                    setting_num = setting_num + 1
                    