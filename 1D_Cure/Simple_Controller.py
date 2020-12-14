# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 18:36:06 2020

@author: Grayson
"""
import numpy as np

class Simple_Controller:
    
    def __init__(self, field_length, spacial_field, kp_m, kd_m, kp_p, kd_p):
        
        # Environment data
        self.field_length = field_length
        self.spacial_field = spacial_field
        
        # Desired state information
        self.desired_temperature = 300.0
        self.desired_temperature_rate = 0.0
        self.lead_distance = 0.10*field_length
        
        # Control gains
        self.kp_m = kp_m
        self.kd_m = kd_m
        self.kp_p = kp_p
        self.kd_p = kd_p
        
    def get_action(self, state):
        
        # Collect and organize state data
        indices_for_field = int(0.50 * (len(state) - 5.0))
        temperature_state = state[0:indices_for_field]
        temperature_rate = state[indices_for_field:-5]
        front_location = state[-5]
        front_rate = state[-4]
        input_location = state[-3]
        input_location_rate = state[-2]
        
        # Calculate the desired input location
        desired_input_location = front_location + self.lead_distance
        desired_input_rate = front_rate
        
        # Determine the part of the temperature state that corresponds to the desired input location
        contracted_spacial_field = np.concatenate((self.spacial_field[0::10], [self.spacial_field[-1]]))
        desired_location_index = np.argmin(abs(desired_input_location - contracted_spacial_field))
        
        # Calculate the state error
        temperature_error = temperature_state[desired_location_index] - self.desired_temperature
        temperature_rate_error = temperature_rate[desired_location_index] - self.desired_temperature_rate
        input_error = input_location - desired_input_location
        input_rate_error = input_location_rate - desired_input_rate
        
        # Use the state error and gains to calculate the desired magnitude rate and position rate
        new_input_location_rate = (self.kp_p * input_error + self.kd_p * input_rate_error)
        new_input_magnitude_rate = (self.kp_m * temperature_error + self.kd_m * temperature_rate_error)
        
        # If the front is done propogating, turn off laser
        if front_location >= 0.85 * self.field_length:
            new_input_magnitude_rate = -100.0
        
        # Return the calculated action
        return new_input_location_rate, new_input_magnitude_rate