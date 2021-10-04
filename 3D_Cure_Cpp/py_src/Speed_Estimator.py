# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:39 2021

@author: GKSch
"""

from Loc_Estimator import Estimator as loc_est
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

class Estimator:

    def __init__(self, loc_load_path, num_images, image_spacing, speed_min, speed_max, alpha_zero, decay_rate, load_path=""):

        if (load_path == ""):
            self.weight = 1.0
            self.bias = 0.0
            
            self.num_images = num_images
            self.image_spacing = image_spacing
            
            self.speed_min = speed_min
            self.speed_max = speed_max
            
            self.location_estimator = loc_est(0, 0, 0, 0, 0, 0, path=loc_load_path)
            
            self.training_curve = []
            self.weight_curve = []
            self.bias_curve = []
            self.lr_curve = []
            
        else:
            self.load(load_path)
        
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
        
        self.weight = loaded_data['weight']
        self.bias = loaded_data['bias']
        
        self.num_images = loaded_data['num_images']
        self.image_spacing = loaded_data['image_spacing']
        
        self.speed_min = loaded_data['speed_min']
        self.speed_max = loaded_data['speed_max']
        
        self.location_estimator = loaded_data['location_estimator']

        self.training_curve = loaded_data['training_curve']
        self.weight_curve = loaded_data['weight_curve']
        self.bias_curve = loaded_data['bias_curve']
        self.lr_curve = loaded_data['lr_curve']
    
    # Returns some important parameters
    def get_params(self):
        return self.num_images, self.image_spacing
    
    # Estimates the front's x speed based on a set temperature images
    # @param the input temperature images
    # @return the estimated x location of the front
    def get_estimate(self, states):
        
        estimated_x_loc = []
        for i in range(self.num_images):
                estimated_x_loc.append(self.location_estimator.get_x_loc(states[i]))
        estimated_x_loc = np.array(estimated_x_loc)
        
        estimated_speed = np.diff(estimated_x_loc)
        estimated_speed = estimated_speed / self.image_spacing
        estimated_speed = np.mean(estimated_speed)
        estimated_speed = estimated_speed * self.weight + self.bias
        
        if estimated_speed < 0.0:
            estimated_speed = 0.0
        
        return estimated_speed
    
    # Updates the weight and bias of the x location estimator
    # @param A collection of input temperature image sequences
    # @param A collection of associated true front speeds
    # @return Normalized error, weight, and bias after update
    def learn(self, states, targets):
        
        # Estimate the front location and speed of all the image sequences
        estimated_speed = []
        for i in range(len(states)):
            estimated_speed.append(self.get_estimate(states[i]))
        estimated_speed = np.array(estimated_speed)
        
        # Calculate varaibles used to determine update gradient
        error = np.mean(abs(estimated_speed - targets))
        norm_error = error / (self.speed_max - self.speed_min) 
        
        # Calculate the gradient
        error_minus_weight = np.mean(abs(0.999*estimated_speed - targets))
        error_plus_weight = np.mean(abs(1.001*estimated_speed - targets))
        error_minus_bias = np.mean(abs((estimated_speed - 0.001*(self.speed_max-self.speed_min)) - targets))
        error_plus_bias = np.mean(abs((estimated_speed + 0.001*(self.speed_max-self.speed_min)) - targets))
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
            'weight' : self.weight,
            'bias' : self.bias,
            'num_images' : self.num_images,
            'image_spacing' : self.image_spacing,
            'speed_min' : self.speed_min,
            'speed_max' : self.speed_max,
            'location_estimator' : self.location_estimator,
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
        initial_path = "../results/speed_estimator"
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
        print("Plotting estimator training curve...")
        
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
        plt.title("Speed Estimator Learning Curve",fontsize='xx-large')
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
        plt.title("Speed Estimator Averaged Learning Curve",fontsize='xx-large')
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
        plt.title("Speed Estimator Weight",fontsize='xx-large')
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
        plt.title("Speed Estimator Bias",fontsize='xx-large')
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
        plt.title("Speed Estimator Learning Rate",fontsize='xx-large')
        plt.xlabel("Batch",fontsize='large')
        plt.ylabel("Learning Rate",fontsize='large')
        plt.plot([*range(len(self.lr_curve))],self.lr_curve,lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/estimator_lr.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()