#pragma once
#include <deque>
#include <vector>
#include <math.h>
#include "Config_Handler.hpp"

using namespace std;

/**
* Speed estimator class takes observations of front locations and times and converts these to front speed estimates
**/
class Speed_Estimator
{
	public:
		// public constructor
		Speed_Estimator(vector<vector<double>> fds_coarse_x_grid_z0);
		
		// public functions
		double get_observation_delta_t();
		int observe(vector<vector<double>> temperature_image, double time);
		double estimate();
		int reset();
		
	private:
		// from cfg and construction
		int sequence_length;
		double observation_delta_t;
		double filter_time_const;
		vector<vector<double>> coarse_x_grid_z0;
		
		// private variables for speed estimation
		deque<double> front_location_history;
		deque<double> observation_time_history;
		double x_loc_estimate;
		double speed_estimate;
		
		// private functions
		double get_avg(deque<double> input);
		double estimate_front_location(vector<vector<double>> temperature_image);
};