#pragma once
#include <math.h>
#include <deque>
#include <vector>
#include "Config_Handler.h"
#include "Finite_Difference_Solver.h"

using namespace std;

/**
* Speed estimator class takes observations of front locations and times and converts these to front speed estimates
**/
class Speed_Estimator
{
	public:
		// public constructor and destructor
		Speed_Estimator(Finite_Difference_Solver* FDS);
		
		// public functions
		int observe(vector<vector<double>> temperature_image, double time);
		double estimate();
		int reset();
		
	private:
		// private variables
		int sequence_length;
		double filter_time_const;
		vector<vector<double>> coarse_x_mesh_z0;
		double coarse_x_step;
		deque<double> front_location_history;
		deque<double> observation_time_history;
		double x_loc_estimate;
		double speed_estimate;
		
		// private functions
		double get_avg(deque<double> input);
		double estimate_front_location(vector<vector<double>> temperature_image);
};

/**
* Constructor for speed estimator
* @param Configuration handler object that contains all loaded and calculated configuration data
*/
Speed_Estimator::Speed_Estimator(Finite_Difference_Solver* FDS)
{	
	// Initialize configuration handler class
	Config_Handler speed_estimator_cfg("../config_files", "speed_estimator.cfg");

	// Set sequence length and filter time constant
	speed_estimator_cfg.get_var("sequence_length", sequence_length);
	speed_estimator_cfg.get_var("filter_time_const", filter_time_const);
	coarse_x_mesh_z0 = FDS->get_coarse_x_mesh_z0();
	
	// Populate observation histories
	for( int i = 0; i < sequence_length; i++ )
	{
		front_location_history.push_back(0.0);
		observation_time_history.push_back( double(sequence_length - i)*(-0.02) );
	}
	
	// Set location and speed estimate to 0
	x_loc_estimate = 0.0;
	speed_estimate = 0.0;
}

/**
* Gets the average value of a deque object
* @param The deque object for which the average is being calculated
* @return The average of the deque object
**/
double Speed_Estimator::get_avg(deque<double> input)
{
	double sum = 0.0;
	for( unsigned int i = 0; i < input.size(); i++ )
	{
		sum += input[i];
	}
	double avg = sum / (double)(input.size());
	return avg;
}

/**
* Adds front location point to location history used to estimate speed
* @param Mean front x location at time of observation
* @param Simulated time at observation
* @return 0 on success, 1 on failure
*/
int Speed_Estimator::observe(vector<vector<double>> temperature_image, double time)
{
	// Estimate the mean x location of the leading edge of the front
	x_loc_estimate = estimate_front_location(temperature_image);
	
	//  Add newest observation
	front_location_history.push_back(x_loc_estimate);
	observation_time_history.push_back(time);
	
	// Remove oldest observation
	front_location_history.pop_front();
	observation_time_history.pop_front();
	
	return 0;
}

/** 
* Estimates the mean x location of the front's leading edge given temperature field observation
* @param Normalized image of temperature field
* @return X location of the mean x location of the front's leading edge based on dT/dx (NON NORMALIZED)
*/
double Speed_Estimator::estimate_front_location(vector<vector<double>> temperature_image)
{
	// Find the x location that corresponds to the max amplitude temperature derivative in the x direction for each column j
	vector<double> dt_dx_max_x_loc = vector<double>(temperature_image[0].size(), -1.0);
	for(unsigned int j = 0; j < temperature_image[0].size(); j++)
	{
		// Tracks the maximum observed temperature derivative in the x direction in column j
		double max_abs_dt_dx_j = 0.0;
		
		for(unsigned int i = 0; i < temperature_image.size(); i++)
		{
			// Store the magnitude of the temperature derivative in the x direction at point (i,j)
			double abs_dt_dx_ij = 0.0;
			
			// Left boundary condition
			if( i==0 )
			{
				abs_dt_dx_ij = abs(-1.5*temperature_image[i][j] + 2.0*temperature_image[i+1][j] + -0.5*temperature_image[i+2][j]);
			}
			
			// Right boundary condition
			else if( i==temperature_image.size()-1 )
			{
				abs_dt_dx_ij = abs(0.5*temperature_image[i-2][j] + -2.0*temperature_image[i-1][j] + 1.5*temperature_image[i][j]);
			}
			
			// Bulk condition
			else
			{
				abs_dt_dx_ij = abs(-0.5*temperature_image[i-1][j] + 0.5*temperature_image[i+1][j]);
			}
			
			// Save max derivative x location so long as the derivate is greater than some threshold
			if ( abs_dt_dx_ij >= max_abs_dt_dx_j || abs_dt_dx_ij > 0.15 )
			{
				dt_dx_max_x_loc[j] = coarse_x_mesh_z0[i][j];
				max_abs_dt_dx_j = abs_dt_dx_ij;
			}
		}
	}
	
	// Sum the admissable front x locations
	double x_loc_sum = 0.0;
	double count = 0.0;
	for(unsigned int j = 0; j < temperature_image[0].size(); j++)
	{
		if( dt_dx_max_x_loc[j] > 0.0 )
		{
			x_loc_sum += dt_dx_max_x_loc[j];
			count = count + 1.0;
		}
	}
	
	// If there are more than 0 instances of fronts, take the average and update the front location estimate, other do not update the front x location
	double curr_estimate = x_loc_estimate;
	if(count > 0.0)
	{
		// Average
		curr_estimate = x_loc_sum / count;
	}
	else
	{
		curr_estimate = x_loc_estimate;
	}
	
	return curr_estimate;
	
}

/**
* Estimates the front velocity based on previous observations of the temperature field
* @return Estiamte of the front velocity (NON NORMALIZED)
*/
double Speed_Estimator::estimate()
{
	// Calculate the average mean x location and sim time from front location history
	double front_location_history_avg = get_avg(front_location_history);
	double observation_time_history_avg = get_avg(observation_time_history);
	
	
	// Apply a simple linear regression to the front mean x location history to determine the front velocity
	double sample_covariance = 0.0;
	double sample_variance = 0.0;
	double observation_time_step_avg = 0.0;
	for(int i = 0; i < sequence_length; i++)
	{
		double delta_x = observation_time_history[i] - observation_time_history_avg;
		double delta_y = front_location_history[i] - front_location_history_avg;
		
		sample_covariance += delta_x*delta_y;
		sample_variance += delta_x*delta_x;
		if (i != sequence_length-1)
		{
			observation_time_step_avg += observation_time_history[i+1] - observation_time_history[i];
		}
	}
	
	// Calculate filter alpha based on time constant and average time step
	observation_time_step_avg = observation_time_step_avg / (double)(sequence_length-1);
	double front_filter_alpha = 1.0 - exp(-observation_time_step_avg/filter_time_const);
	
	// Pass the front velocity signal through a SPLP filter
	double curr_speed_estimate = sample_variance==0.0 ? 0.0 : sample_covariance / sample_variance;
	speed_estimate += front_filter_alpha * ( curr_speed_estimate - speed_estimate );
	
	return speed_estimate;
}

/** 
* Resets the speed estimator for new trajecotry
* @return 0 on success, 1 on failure
*/
int Speed_Estimator::reset()
{
	// Clear the memory
	front_location_history.clear();
	observation_time_history.clear();
	
	// Populate observation histories
	for( int i = 0; i < sequence_length; i++ )
	{
		front_location_history.push_back(0.0);
		observation_time_history.push_back( double(sequence_length - i)*(-0.02) );
	}
	
	// Set x location and speed estimate to 0
	x_loc_estimate = 0.0;
	speed_estimate = 0.0;
	
	return 0;
}