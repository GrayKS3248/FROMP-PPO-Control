# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 18:36:06 2020

@author: Grayson
"""
import numpy as np

class PD_Controller:
    
    def __init__(self, length, panels, kp_m, kd_m, kp_p, kd_p, kp_t=0.75, kd_t=-0.1, random_vel=False):
        
        # Environment data
        self.length = length
        self.panels = panels
        self.contracted_panels = np.mean(np.resize(self.panels,(len(self.panels)//10,10)),axis=1)
        self.random_vel = random_vel
        
        # Desired state information
        self.initial_temperature = 306.0
        self.desired_temperature = self.initial_temperature
        self.desired_temperature_rate = 0.0
        self.lead_distance = 0.09*length
        
        # Control gains
        self.kp_m = kp_m
        self.kd_m = kd_m
        self.kp_p = kp_p
        self.kd_p = kd_p
        self.kp_t = kp_t
        self.kd_t = kd_t
        
        # Memory
        self.previous_front_rate = 0.0
        
    def get_action(self, state):
        
        # Collect and organize state data
        indices_for_field = int(0.50 * (len(state) - 5.0))
        temperature_state = state[0:indices_for_field]
        temperature_rate = state[indices_for_field:-5]
        front_location = state[-5]
        front_rate = state[-4]
        input_location = state[-3]
        input_location_rate = state[-2]
        target_front_rate = state[-1]
        
        # Calculate the desired input location
        desired_input_location = front_location + self.lead_distance
        desired_input_rate = front_rate
        
        # Determine the part of the temperature state that corresponds to the desired input location
        desired_location_index = np.argmin(abs(desired_input_location - self.contracted_panels))
        
        # Calculate the state error
        temperature_error = temperature_state[desired_location_index] - self.desired_temperature
        temperature_rate_error = temperature_rate[desired_location_index] - self.desired_temperature_rate
        input_error = input_location - desired_input_location
        input_rate_error = input_location_rate - desired_input_rate
        
        # Use the state error and gains to calculate the desired magnitude rate and position rate
        new_input_location_rate = (self.kp_p * input_error + self.kd_p * input_rate_error)
        new_input_magnitude_rate = (self.kp_m * temperature_error + self.kd_m * temperature_rate_error)
        
        # If the front is done propogating, turn off laser
        if front_location >= 0.85 * self.length:
            new_input_magnitude_rate = -100.0
        
        # Adjust the desired temperature if the front speed is not good
        if self.random_vel and abs(front_rate - target_front_rate)/target_front_rate >= 0.075:
            delta = self.kp_t*(target_front_rate - front_rate)/target_front_rate + self.kd_t*(front_rate - self.previous_front_rate)
            self.desired_temperature = self.desired_temperature + delta
            
        # Update the memory
        self.previous_front_rate = front_rate
        
        # Return the calculated action
        return new_input_location_rate, new_input_magnitude_rate
    
    def reset(self):
        self.desired_temperature = self.initial_temperature