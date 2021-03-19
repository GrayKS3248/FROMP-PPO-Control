# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:52:58 2021

@author: GKSch
"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    with open('results/1-4-8_64/output', 'rb') as file:
        load_data_1 = pickle.load(file)
    
    with open('results/1-8-16_64/output', 'rb') as file:
        load_data_2 = pickle.load(file)
    
    with open('results/1-12-24_64/output', 'rb') as file:
        load_data_3 = pickle.load(file)
        
    load_data_1=load_data_1['data']
    load_data_2=load_data_2['data']
    load_data_3=load_data_3['data']
    
    ts_1 = pd.Series(load_data_1['MSE_loss'])
    ts_2 = pd.Series(load_data_2['MSE_loss'])
    ts_3 = pd.Series(load_data_3['MSE_loss'])
    
    avg_1 = ts_1.rolling(window=1000).mean()
    avg_2 = ts_2.rolling(window=1000).mean()
    avg_3 = ts_3.rolling(window=1000).mean()
    
    plt.clf()
    title_str = "Autoencoder Learning Curve for DCPD GC2"
    plt.title(title_str,fontsize='xx-large')
    plt.xlabel("Optimization Frame",fontsize='large')
    plt.ylabel("MSE Loss",fontsize='large')
    plt.plot([*range(len(avg_1))],avg_1,lw=2.5,c='r')
    plt.plot([*range(len(avg_2))],avg_2,lw=2.5,c='g')
    plt.plot([*range(len(avg_3))],avg_3,lw=2.5,c='b')
    plt.yscale("log")
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.gcf().set_size_inches(8.5, 5.5)
    plt.legend(('1-4-8','1-8-16','1-12-24'),loc='best',fontsize='large')
    plt.savefig('results/autoencoder_learning.png', dpi = 500)
    plt.close()