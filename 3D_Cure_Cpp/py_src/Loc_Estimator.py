# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

class Estimator:

    def __init__(self, x_dim, y_dim, min_x_loc, max_x_loc, alpha_zero, decay_rate, path=""):

        if (path == ""):
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.max_x_loc = max_x_loc
            self.min_x_loc = min_x_loc
            self.x_loc_range = np.linspace(min_x_loc, max_x_loc, x_dim-1)   
            self.bias = 0.0
            self.weight = 1.0
            self.training_curve = []
            self.weight_curve = []
            self.bias_curve = []
            self.lr_curve = []
        else:
            self.load(path)
        
        self.alpha = alpha_zero
        self.decay = decay_rate

 
    # Loads a given saved estimator
    # @param the path from which the estimator will be loaded
    def load(self, path):
        
        if not os.path.isdir(path):
            raise RuntimeError("Could not find " + path)
        else:
            with open(path+"/output", 'rb') as file:
                loaded_data = pickle.load(file)
        
        self.x_dim = loaded_data['x_dim']
        self.y_dim = loaded_data['y_dim']
        self.max_x_loc = loaded_data['max_x_loc']
        self.min_x_loc = loaded_data['min_x_loc']
        self.x_loc_range = loaded_data['x_loc_range']   
        
        self.bias = loaded_data['bias']
        self.weight = loaded_data['weight']
 
        self.training_curve = loaded_data['training_curve']
        self.weight_curve = loaded_data['weight_curve']
        self.bias_curve = loaded_data['bias_curve']
        self.lr_curve = loaded_data['lr_curve']
 
    # Estimates the front's x location based on the temperature image
    # @param the input temperature image
    # @return the estimated x location of the front
    def get_x_loc(self, state):

        x_loc = np.diff(state,axis=0)
        x_loc = abs(x_loc)
        x_loc = np.argmax(x_loc,axis=0)
        x_loc = self.x_loc_range[x_loc]
        x_loc = np.mean(x_loc)
        x_loc = x_loc * self.weight + self.bias
        
        return x_loc
    
    # Updates the weight and bias of the x location estimator
    # @param A collection of input temperature images
    # @param A collection of associated true front x locations
    # @return Normalized error, weight, and bias after update
    def learn(self, states, targets):
        
        # Estimate the front location given the states and current weights and biases
        estimated_x_loc = []
        for i in range(len(states)):
            estimated_x_loc.append(self.get_x_loc(states[i]))
        estimated_x_loc = np.array(estimated_x_loc)
        
        # Calculate varaibles used to determine update gradient
        error = np.mean(abs(estimated_x_loc - targets))
        norm_error = error / (self.max_x_loc - self.min_x_loc) 
        
        # Calculate the gradient
        error_minus_weight = np.mean(abs(0.99*estimated_x_loc - targets))
        error_plus_weight = np.mean(abs(1.01*estimated_x_loc - targets))
        error_minus_bias = np.mean(abs((estimated_x_loc - 0.01*(self.max_x_loc-self.min_x_loc)) - targets))
        error_plus_bias = np.mean(abs((estimated_x_loc + 0.01*(self.max_x_loc-self.min_x_loc)) - targets))
        gradient = np.array([(error_plus_weight - error_minus_weight), (error_plus_bias - error_minus_bias)])
        
        # Update the weights, biases, and learning rates
        self.weight = self.weight - gradient[0] * self.alpha
        self.bias = self.bias - gradient[1] * self.alpha
        self.alpha = self.alpha * self.decay
        
        # return normalized error, weight, and bias
        return norm_error, self.weight, self.bias, self.alpha
    
    # Saves the training data and trained model
    # @param training loss curve
    # @param weight curve
    # @param bias curve
    # @param learning rate curve
    # @return path at which data has been saved
    def save(self, training_curve, weight_curve, bias_curve, lr_curve):
        print("\nSaving state estimator results...")

        # Store data to dictionary
        data = {
            'x_dim' : self.x_dim,
            'y_dim' : self.y_dim,
            'max_x_loc' : self.max_x_loc,
            'min_x_loc' : self.min_x_loc,
            'bias' : self.bias,
            'weight' : self.weight,
            'x_loc_range' : self.x_loc_range,
            'training_curve' : training_curve,
            'weight_curve' : weight_curve,
            'bias_curve' : bias_curve,
            'lr_curve' : lr_curve,
        }
        
        self.training_curve = training_curve
        self.weight_curve = weight_curve
        self.bias_curve = bias_curve
        self.lr_curve = lr_curve

        # Find save paths
        initial_path = "../results/x_loc_estimator"
        path = initial_path
        done = False
        curr_dir_num = 1
        while not done:
            if not os.path.isdir(path):
                os.mkdir(path)
                done = True
            else:
                curr_dir_num = curr_dir_num + 1
                path = initial_path + "(" + str(curr_dir_num) + ")"

        # Pickle all important outputs
        save_file = path + "/output"
        with open(save_file, 'wb') as file:
            pickle.dump(data, file)

        return path

    # Draw and save the training curve
    # @param the window over wich the rolling average is taken
    # @param path at which training curve is saved
    def draw_training_curve(self, window, path):
        print("Plotting estimator training curves...")
        
        # Format data
        training_curve = pd.Series(self.training_curve)
        training_curve_mean = np.array(training_curve.rolling(window).mean())
        training_curve_mean[0:window-1] = training_curve_mean[window-1]
        training_curve_std = np.array(training_curve.rolling(window).std())
        training_curve_std[0:window-1] = training_curve_std[window-1]
        lower = training_curve_mean-training_curve_std
        lower[lower<0.0] = 0.0
        upper = training_curve_mean+training_curve_std
        
        plt.clf()
        plt.title("X Location Estimator Learning Curve",fontsize='xx-large')
        plt.xlabel("Batch",fontsize='large')
        plt.ylabel("Range Normalized RMS Error",fontsize='large')
        plt.plot([*range(len(training_curve))],training_curve,lw=2.5,c='r')
        plt.yscale("log")
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/estimator_training_1.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()
        
        plt.clf()
        plt.title("X Location Estimator Averaged Learning Curve",fontsize='xx-large')
        plt.xlabel("Batch",fontsize='large')
        plt.ylabel("Range Normalized RMS Error W="+str(window),fontsize='large')
        plt.plot([*range(len(training_curve_mean))],training_curve_mean,lw=2.5,c='r')
        plt.fill_between([*range(len(training_curve_mean))], upper, lower, alpha=0.25, color='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/estimator_training_2.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()
        
        plt.clf()
        plt.title("X Location Estimator Weight",fontsize='xx-large')
        plt.xlabel("Batch",fontsize='large')
        plt.ylabel("Weight",fontsize='large')
        plt.plot([*range(len(self.weight_curve))],self.weight_curve,lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/estimator_weight.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()
        
        plt.clf()
        plt.title("X Location Estimator Bias",fontsize='xx-large')
        plt.xlabel("Batch",fontsize='large')
        plt.ylabel("Bias",fontsize='large')
        plt.plot([*range(len(self.bias_curve))],self.bias_curve,lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/estimator_bias.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()
        
        plt.clf()
        plt.title("X Location Estimator Learning Rate",fontsize='xx-large')
        plt.xlabel("Batch",fontsize='large')
        plt.ylabel("Learning Rate",fontsize='large')
        plt.plot([*range(len(self.lr_curve))],self.lr_curve,lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/estimator_lr.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()