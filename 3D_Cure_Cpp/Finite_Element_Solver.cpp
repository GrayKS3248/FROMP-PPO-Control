#include "Finite_Element_Solver.h"
#define M_PI   3.14159265358979323846264338327950288

using namespace std;

/**
* Default constructor
* @param length of encoded state
* @param boolean flag that indicates whether front location and velocity data are appended to encoded state
* @param decimal percent of length sent to autoencoder
*/
Finite_Element_Solver::Finite_Element_Solver(int encoded_size_in, bool get_extended_state_in, double x_crop)
{
	// Set randomization seed
	srand(time(NULL));

	// Simulation time and target velocity index
	current_time = 0.0;  // Seconds
	current_index = 0;   // Unitless
	
	// State information
	encoded_size = encoded_size_in;
	get_extended_state = get_extended_state_in;
	if (x_crop > 0.0 && x_crop < 1.0)
	{
		num_vert_sub_length = (int)round(x_crop*(double)num_vert_length);
	}
	else
	{
		num_vert_sub_length = num_vert_length;
	}	
	
	// Monomer physical parameters
	if (use_DCPD_GC1)
	{
		thermal_diffusivity = DCPD_GC1_thermal_conductivity / (DCPD_GC1_specific_heat * DCPD_GC1_density);
		thermal_conductivity = DCPD_GC1_thermal_conductivity;
		enthalpy_of_reaction = DCPD_GC1_enthalpy_of_reaction;
		specific_heat = DCPD_GC1_specific_heat;
	}
	else if (use_DCPD_GC2)
	{
		thermal_diffusivity = DCPD_GC2_thermal_conductivity / (DCPD_GC2_specific_heat * DCPD_GC2_density);
		thermal_conductivity = DCPD_GC2_thermal_conductivity;
		enthalpy_of_reaction = DCPD_GC2_enthalpy_of_reaction;
		specific_heat = DCPD_GC2_specific_heat;
	}
	else if (use_COD)
	{
		thermal_diffusivity = COD_thermal_conductivity / (COD_specific_heat * COD_density);
		thermal_conductivity = COD_thermal_conductivity;
		enthalpy_of_reaction = COD_enthalpy_of_reaction;
		specific_heat = COD_specific_heat;
	}

	// Calculate the target temporal vector and define the current target
	int sim_steps = (int)round(sim_duration / time_step);
	double target = 0.0;
	double randomizing_scale = 0.0;
	if(control_speed)
	{
		if(use_DCPD_GC1)
		{
			target = DCPD_GC1_target_vel;
			randomizing_scale = DCPD_GC1_vel_rand_scale;
		}
		else if(use_DCPD_GC2)
		{
			target = DCPD_GC2_target_vel;
			randomizing_scale = DCPD_GC2_vel_rand_scale;
		}
		else if(use_COD)
		{
			target = COD_target_vel;
			randomizing_scale = COD_vel_rand_scale;
		}
	}
	else if(control_temperature)
	{
		if(use_DCPD_GC1)
		{
			target = DCPD_GC1_target_temp;
			randomizing_scale = DCPD_GC1_temp_rand_scale;
		}
		else if(use_DCPD_GC2)
		{
			target = DCPD_GC2_target_temp;
			randomizing_scale = DCPD_GC2_temp_rand_scale;
		}
		else if(use_COD)
		{
			target = COD_target_temp;
			randomizing_scale = COD_temp_rand_scale;
		}
	}
	target_vector = vector<double>(sim_steps, target);
	if (random_target)
	{
		double new_target = target - 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
		for (unsigned int i = 0; i < target_vector.size(); i++)
		{
			target_vector[i] = new_target;
		}
	}
	else if (target_switch)
	{
		int switch_location = (int) floor((0.20 * (double)rand()/(double)RAND_MAX + 0.40) * (double)(target_vector.size() - 1));
		double switch_vel = target_vector[switch_location] + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
		for (unsigned int i = switch_location; i < target_vector.size(); i++)
		{
			target_vector[i] = switch_vel;
		}
	}
	current_target = target_vector[current_index];

	// Set trigger conditions
	if (!trigger)
	{
		trigger_flux = 0.0;      // Watts / Meter ^ 2
		trigger_time = 0.0;      // Seconds
		trigger_duration = 0.0;  // Seconds
	}
	else
	{
		if (use_DCPD_GC1)
		{
			trigger_flux = DCPD_GC1_trigger_flux_ref;          // Watts / Meter ^ 2
			trigger_time = trigger_time_ref;          // Seconds
			trigger_duration = DCPD_GC1_trigger_duration_ref;  // Seconds
		}
		else if (use_DCPD_GC2)
		{
			trigger_flux = DCPD_GC2_trigger_flux_ref;          // Watts / Meter ^ 2
			trigger_time = trigger_time_ref;          // Seconds
			trigger_duration = DCPD_GC2_trigger_duration_ref;  // Seconds
		}
		else if (use_COD)
		{
			trigger_flux = COD_trigger_flux_ref;          // Watts / Meter ^ 2
			trigger_time = trigger_time_ref;          // Seconds
			trigger_duration = COD_trigger_duration_ref;  // Seconds
		}
		

	}

	// Create mesh and calculate step size
	vector<double> x_range(num_vert_length, 0.0);
	vector<double> y_range(num_vert_width, 0.0);
	vector<double> z_range(num_vert_depth, 0.0);
	mesh_x = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth)));
	mesh_y = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth)));
	mesh_z = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth)));
	for (int i = 0; i < max(num_vert_length, max(num_vert_width, num_vert_depth)); i++)
	{
		if (i < num_vert_length)
		{
			x_range[i] = ((double)i / (double)(num_vert_length - 1)) * length;
		}
		if (i < num_vert_width)
		{
			y_range[i] = ((double)i / (double)(num_vert_width - 1)) * width;
		}
		if (i < num_vert_depth)
		{
			z_range[i] = ((double)i / (double)(num_vert_depth - 1)) * depth;
		}
	}
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			for (int k = 0; k < num_vert_depth; k++)
			{
				mesh_x[i][j][k] = x_range[i];
				mesh_y[i][j][k] = y_range[j];
				mesh_z[i][j][k] = z_range[k];
			}
		}
	}
	x_step = mesh_x[1][0][0];
	y_step = mesh_y[0][1][0];
	z_step = mesh_z[0][0][1];

	// Init and perturb temperature and cure meshes
	temp_mesh = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, initial_temperature)));
	temp_mesh = get_perturbation(temp_mesh, initial_temp_delta);
	cure_mesh = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, initial_cure)));
	cure_mesh = get_perturbation(cure_mesh, initial_cure_delta);

	// Init front mesh and parameters
	front_loc = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
	front_vel = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
	time_front_last_moved = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
	front_temp = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, initial_temperature));

	// Input magnitude parameters
	double sigma = 0.329505114491 * radius_of_input;
	exp_const = -1.0 / (2.0 * sigma * sigma);
	double delta_x = (1.0 / 999.0) * radius_of_input;
	double x = 0.0;
	max_input_mag = 1.01;
	for (int i = 1; i < 999; i++)
	{
		x = ((double)i / 999.0) * radius_of_input;
		max_input_mag += 2.0 * pow(0.01, ((x * x) / (radius_of_input * radius_of_input)));
	}
	max_input_mag = laser_power / (4.0 * (max_input_mag * delta_x / 2.0) * (max_input_mag * delta_x / 2.0));
	input_percent = (double)rand()/(double)RAND_MAX;

	// Input location parameters
	min_input_x_loc = 0.0;
	max_input_x_loc = length;
	min_input_y_loc = 0.0;
	max_input_y_loc = width;
	input_location = vector<double>(2, 0.0);
	
	// Select a random input location
	int length_pos = (int)floor((double)rand()/(double)RAND_MAX * num_vert_length);
	length_pos = length_pos >= num_vert_length ? num_vert_length - 1 : length_pos;
	int width_pos = (int)floor((double)rand()/(double)RAND_MAX * num_vert_width);
	width_pos = width_pos >= num_vert_width ? num_vert_width - 1 : width_pos;

	// Assign the random input locationj
	input_location[0] = mesh_x[length_pos][0][0];
	input_location[1] = mesh_y[0][width_pos][0];
		
	// Apply correction if in control mode
	if (control)
	{
		max_input_mag = 0.0;
		min_input_x_loc = mesh_x[(int)floor(0.5 * (double)num_vert_length)][0][0];
		max_input_x_loc = min_input_x_loc;
		min_input_y_loc = mesh_y[0][(int)floor(0.5 * (double)num_vert_width)][0];
		max_input_y_loc = min_input_y_loc;
		input_location[0] = min_input_x_loc;
		input_location[1] = min_input_y_loc;
	}

	// Initiate input wattage mesh
	input_mesh = vector<vector<double>>(num_vert_length, vector<double>(num_vert_width, 0.0));
	double local_input_power = 0.0;
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			local_input_power = input_percent * max_input_mag * exp(pow((mesh_x[i][j][0] - input_location[0]), 2.0) * exp_const + pow((mesh_y[i][j][0] - input_location[1]), 2.0) * exp_const);
			if (local_input_power < 0.01 * max_input_mag)
			{
				input_mesh[i][j] = 0.0;
			}
			else
			{
				input_mesh[i][j] = local_input_power;
			}
		}
	}
}

/**
* Gets the number of vertices in lengthwise direction
* @return The number of vertices in the lengthwise direction
*/
int Finite_Element_Solver::get_num_vert_length()
{
	return num_vert_length;
}

/**
* Gets the number of vertices in widthwise direction
* @return The number of vertices in the widthwise direction
*/
int Finite_Element_Solver::get_num_vert_width()
{
	return num_vert_width;
}

/**
* Gets the number of vertices in depthwise direction
* @return The number of vertices in the depthwise direction
*/
int Finite_Element_Solver::get_num_vert_depth()
{
	return num_vert_depth;
}

/**
* Gets the duration of the simulation
* @return The duration of the simulation in seconds
*/
double Finite_Element_Solver::get_sim_duration()
{
	return sim_duration;
}

/**
* Gets the time step used in the simulation
* @return The simulation time step in seconds
*/
double Finite_Element_Solver::get_time_step()
{
	return time_step;
}

/**
* Gets the current input power percent
* @return The power level of the input in percent
*/
double Finite_Element_Solver::get_input_percent()
{
	return input_percent;
}

/**
* Gets the location rate scaling factor
* @return The location rate scaling factor that takes NN raw to FES
*/
double Finite_Element_Solver::get_loc_rate_scale()
{
	return loc_rate_scale;
}

/**
* Gets the magnitude scaling factor
* @return The magnitude scaling factor that takes NN raw to FES
*/
double Finite_Element_Solver::get_mag_scale()
{
	return mag_scale;
}

/**
* Gets the maximum magnitude of the input in W/m^2
* @return The peak magnitude of the input in W/m^2
*/
double Finite_Element_Solver::get_max_input_mag()
{
	return max_input_mag;
}

/**
* Gets the exponent constant used to calculate input mesh
* @return The exponent constant used to calculate input mesh in W/m^2
*/
double Finite_Element_Solver::get_exp_const()
{
	return exp_const;
}

/**
* Gets the current target velocity
* @return The current target front velocity in m/s
*/
double Finite_Element_Solver::get_current_target()
{
	return current_target;
}

/**
* Gets the length of the target velocity array
* @return The length of the target velocity array
*/
int Finite_Element_Solver::get_target_vector_arr_size()
{
	return target_vector.size();
}

/**
* Gets the current time
* @return The time in seconds
*/
double Finite_Element_Solver::get_current_time()
{
	return current_time;
}

/**
* Gets the length of the state vector
*/
int Finite_Element_Solver::get_num_state()
{
	// Temperature field view
	int num_states = encoded_size;

	// Add front location and veloicty views
	if (get_extended_state)
	{
		num_states += (int)((double)num_vert_width/4.0) + (int)((double)num_vert_width/4.0);	
	}

	// Add input location and magnitude states
	num_states += 3;

	// Return calculated number of states
	return num_states;
}

/**
* Gets whether the speed is controlled or not
* @return Whether the speed is controlled or not
*/
bool Finite_Element_Solver::get_control_speed()
{
	return control_speed;
}

/**
* Gets the number of vertices in the x dim sent to the autoencoder
* @return the number of vertices in the x dim sent to the autoencoder
*/
int Finite_Element_Solver::get_cropped_x_dim()
{
	return num_vert_sub_length;
}

/**
* Prints the finite element solver parameters to std out
*/
void Finite_Element_Solver::print_params()
{
	// Input parameters
	cout << "\nInput(\n";
	if (control)
	{
		cout << "  No input.\n";
	}
	else
	{ 
		cout << "  (Radius): " << 1000.0 * radius_of_input << " mm\n";
		cout << "  (Power): " << 1000.0 * laser_power << " mW\n";
		cout << "  (Power Rate): " << 1000.0 * laser_power * input_mag_percent_rate << " mW/s\n";
		cout << "  (Slew Rate): " << 1000.0 * max_input_loc_rate << " mm/s\n";
	}
	cout << ")\n";
	
	// Trigger parameters
	cout << "\nTrigger(\n";
	if (!trigger)
	{
		cout << "  No trigger.\n";
	}
	else
	{ 
		cout << "  (Flux): " << trigger_flux << " W/m^2\n";
		cout << "  (Time): " << trigger_time << " s\n";
		cout << "  (Duration): " << trigger_duration  << " s\n";
	}
	cout << ")\n";
	
	// Monomer
	cout << "\nMaterial(\n";
	if (use_DCPD_GC1)
	{
		cout << "  (Monomer): DCPD\n";
		cout << "  (Catalyst): GC1\n";
	}
	else if (use_DCPD_GC2)
	{ 
		cout << "  (Monomer): DCPD\n";
		cout << "  (Catalyst): GC2\n";
	}
	else if (use_COD)
	{
		cout << "  (Monomer): COD\n";
		cout << "  (Catalyst): GC2\n";
	}
	cout << "  (Initial Temperature): " << initial_temperature-273.15 << "C +- " << initial_temp_delta << " C\n";
	cout << "  (Initial Cure): " << initial_cure << " +- " << initial_cure_delta << "\n";
	cout << "  (HTC): " << htc << " W/m^2-K\n";
	cout << ")\n";
	
	// Environment
	cout << "\nEnvironment(\n";
	cout << "  (Dimensions): " << 1000.0*length << " x " << 1000.0*width << " x " << 1000.0*depth << " mm\n";
	cout << "  (Grid): " << num_vert_length << " x " << num_vert_width << " x " << num_vert_depth << "\n";
	cout << "  (State): " << get_num_state() << "\n";
	cout << "  (Duration): " << sim_duration << " s\n";
	cout << "  (Time Step): " << 1000.0*time_step << " ms\n";
	cout << "  (Ambient Temperature): " << ambient_temperature-273.15 << " C\n";
	cout << ")\n";
}

/**
* Gets the input location
* @return The input location as a vector {x,y}
*/
vector<double> Finite_Element_Solver::get_input_location()
{
	return input_location;
}

/**
* Gets the top layer of the temperature mesh
* @return The top layer of the temperature mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Element_Solver::get_temp_mesh()
{
	vector<vector<double>> ret_val(num_vert_length, vector<double>(num_vert_width, 0.0));
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			ret_val[i][j] = temp_mesh[i][j][0];
		}
	}
	return ret_val;
}

/**
* Gets the normalized top layer of the temperature mesh around the front
* @return The top layer of the temperature mesh around the front as a 2D vector in x,y normalized against in 0.90*T0 to 1.10*Tmax
*/
vector<vector<double>> Finite_Element_Solver::get_norm_temp_mesh()
{
	// If you are returning a cropped image
	if (num_vert_sub_length != num_vert_length && num_vert_sub_length >= 3)
	{
		// Get the mean front x location on the top layer of the mesh
		double mean_front_loc = 0.0;
		for (int i = 0; i < num_vert_width; i++)
		{
			mean_front_loc += front_loc[i][0];
		}
		mean_front_loc = mean_front_loc / num_vert_width;
		
		// Determine the index of the mean front location
		int mean_front_index = 0;
		double error = 2.0*length;
		bool getting_colder = false;
		while(!getting_colder)
		{
			if (mean_front_index > num_vert_length-1)
			{
				getting_colder = true;
				mean_front_index--;
			}
			else
			{
				double new_error = abs(mean_front_loc - mesh_x[mean_front_index][0][0]);
				if (new_error > error)
				{
					getting_colder = true;
					mean_front_index--;
				}
				else
				{
					error = new_error;
					mean_front_index++;
				}
			}
		}
		
		// Determine the starting and ending vertices based on the mean front index
		int half_num_vert_sub_length = (int)round((double)num_vert_sub_length/2.0);
		int start_index = mean_front_index - half_num_vert_sub_length;
		start_index = start_index < 0 ? 0 : start_index;
		int end_index = start_index + num_vert_sub_length - 1;
		if(end_index >= num_vert_length)
		{
			end_index = num_vert_length - 1;
			start_index = end_index - num_vert_sub_length + 1;
		}
		
		// Get the temperature in the range determined above
		vector<vector<double>> ret_val(num_vert_sub_length, vector<double>(num_vert_width, 0.0));
		for (int i = start_index; i <= end_index; i++)
		{
			for (int j = 0; j < num_vert_width; j++)
			{
				int i_index = i - start_index;
				ret_val[i_index][j] = (temp_mesh[i][j][0] - 0.90*initial_temperature) / (1.1*temperature_limit - 0.90*initial_temperature);
				ret_val[i_index][j] = ret_val[i_index][j] > 1.0 ? 1.0 : ret_val[i_index][j];
				ret_val[i_index][j] = ret_val[i_index][j] < 0.0 ? 0.0 : ret_val[i_index][j];
			}
		}
		return ret_val;
	}
	
	// If you are returning an uncropped image
	// Get the temperature 
	vector<vector<double>> ret_val(num_vert_length, vector<double>(num_vert_width, 0.0));
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			ret_val[i][j] = (temp_mesh[i][j][0] - 0.90*initial_temperature) / (1.1*temperature_limit - 0.90*initial_temperature);
			ret_val[i][j] = ret_val[i][j] > 1.0 ? 1.0 : ret_val[i][j];
			ret_val[i][j] = ret_val[i][j] < 0.0 ? 0.0 : ret_val[i][j];
		}
	}
	return ret_val;

}

/**
* Gets the top layer of the cure mesh
* @return The top layer of the cure mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Element_Solver::get_cure_mesh()
{
	vector<vector<double>> ret_val(num_vert_length, vector<double>(num_vert_width, 0.0));
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			ret_val[i][j] = cure_mesh[i][j][0];
		}
	}
	return ret_val;
}

/**
* Gets the input mesh
* @return The input mesh as a 2D vector in x,y of watts/m^2
*/
vector<vector<double>> Finite_Element_Solver::get_input_mesh()
{
	return input_mesh;
}

/**
* Gets the current front location
* @return The current front location as a 2D vector in y,z
*/
vector<vector<double>> Finite_Element_Solver::get_front_loc()
{
	return front_loc;
}

/**
* Gets the current front velocity
* @return The current front velocity as a 2D vector in y,z
*/
vector<vector<double>> Finite_Element_Solver::get_front_vel()
{
	return front_vel;
}

/**
* Gets the current front temperature
* @return The current front temperature as a 2D vector in y,z
*/
vector<vector<double>> Finite_Element_Solver::get_front_temp()
{
	return front_temp;
}

/**
* Gets the top layer of the x mesh
* @return The top layer of the x mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Element_Solver::get_mesh_x_z0()
{
	vector<vector<double>> ret_val(num_vert_length, vector<double>(num_vert_width, 0.0));
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			ret_val[i][j] = mesh_x[i][j][0];
		}
	}
	return ret_val;
}

/**
* Gets the top layer of the y mesh
* @return The top layer of the y mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Element_Solver::get_mesh_y_z0()
{
	vector<vector<double>> ret_val(num_vert_length, vector<double>(num_vert_width, 0.0));
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			ret_val[i][j] = mesh_y[i][j][0];
		}
	}
	return ret_val;
}

/**
* Gets the left layer of the y mesh
* @return The left layer of the y mesh as a 2D vector in y,z
*/
vector<vector<double>> Finite_Element_Solver::get_mesh_y_x0()
{
	vector<vector<double>> ret_val(num_vert_width, vector<double>(num_vert_depth, 0.0));
	for (int i = 0; i < num_vert_width; i++)
	{
		for (int j = 0; j < num_vert_depth; j++)
		{
			ret_val[i][j] = mesh_y[0][i][j];
		}
	}
	return ret_val;
}

/**
* Gets the left layer of the z mesh
* @return The left layer of the z mesh as a 2D vector in y,z
*/
vector<vector<double>> Finite_Element_Solver::get_mesh_z_x0()
{
	vector<vector<double>> ret_val(num_vert_width, vector<double>(num_vert_depth, 0.0));
	for (int i = 0; i < num_vert_width; i++)
	{
		for (int j = 0; j < num_vert_depth; j++)
		{
			ret_val[i][j] = mesh_z[0][i][j];
		}
	}
	return ret_val;
}

/**
* Steps the environment forward one time step
* @param The raw NN x location rate command
* @param The raw NN y location rate command
* @param The raw NN magnitude command
* @return Whether the sim is done or not
*/
bool Finite_Element_Solver::step(double x_loc_rate_action, double y_loc_rate_action, double mag_action)
{
	// Step the input, cure, front, and temperature
	step_input(x_loc_rate_action, y_loc_rate_action, mag_action);
	step_meshes();

	// Step time_step
	bool done = step_time();
	return done;
}

/**
* Resets the environment to initial conditions
* @return the initial state vector
*/
void Finite_Element_Solver::reset()
{
	// Simulation time and target velocity index
	current_time = 0.0;      // Seconds
	current_index = 0;       // Unitless

	// Calculate the target temporal vector and define the current target
	int sim_steps = (int)round(sim_duration / time_step);
	double target = 0.0;
	double randomizing_scale = 0.0;
	if(control_speed)
	{
		if(use_DCPD_GC1)
		{
			target = DCPD_GC1_target_vel;
			randomizing_scale = DCPD_GC1_vel_rand_scale;
		}
		else if(use_DCPD_GC2)
		{
			target = DCPD_GC2_target_vel;
			randomizing_scale = DCPD_GC2_vel_rand_scale;
		}
		else if(use_COD)
		{
			target = COD_target_vel;
			randomizing_scale = COD_vel_rand_scale;
		}
	}
	else if(control_temperature)
	{
		if(use_DCPD_GC1)
		{
			target = DCPD_GC1_target_temp;
			randomizing_scale = DCPD_GC1_temp_rand_scale;
		}
		else if(use_DCPD_GC2)
		{
			target = DCPD_GC2_target_temp;
			randomizing_scale = DCPD_GC2_temp_rand_scale;
		}
		else if(use_COD)
		{
			target = COD_target_temp;
			randomizing_scale = COD_temp_rand_scale;
		}
	}
	target_vector = vector<double>(sim_steps, target);
	if (random_target)
	{
		double new_target = target - 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
		for (unsigned int i = 0; i < target_vector.size(); i++)
		{
			target_vector[i] = new_target;
		}
	}
	else if (target_switch)
	{
		int switch_location = (int) floor((0.20 * (double)rand()/(double)RAND_MAX + 0.40) * (double)(target_vector.size() - 1));
		double switch_vel = target_vector[switch_location] + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
		for (unsigned int i = switch_location; i < target_vector.size(); i++)
		{
			target_vector[i] = switch_vel;
		}
	}
	current_target = target_vector[current_index];

	// Init and perturb temperature and cure meshes
	temp_mesh = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, initial_temperature)));
	temp_mesh = get_perturbation(temp_mesh, initial_temp_delta);
	cure_mesh = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, initial_cure)));
	cure_mesh = get_perturbation(cure_mesh, initial_cure_delta);

	// Init front mesh and parameters
	front_loc = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
	front_vel = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
	time_front_last_moved = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
	front_temp = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, initial_temperature));

	// Input magnitude parameters
	input_percent = (double)rand()/(double)RAND_MAX;

	// Input location parameters
	input_location = vector<double>(2, 0.0);
	if (control)
	{
		input_location[0] = min_input_x_loc;
		input_location[1] = min_input_y_loc;
	}
	else
	{
		// Select a random input location
		int length_pos = (int)floor((double)rand()/(double)RAND_MAX * num_vert_length);
		length_pos = length_pos >= num_vert_length ? num_vert_length - 1 : length_pos;
		int width_pos = (int)floor((double)rand()/(double)RAND_MAX * num_vert_width);
		width_pos = width_pos >= num_vert_width ? num_vert_width - 1 : width_pos;
		
		// Assign the random input locationj
		input_location[0] = mesh_x[length_pos][0][0];
		input_location[1] = mesh_y[0][width_pos][0];
	}

	// Initiate input wattage mesh
	double local_input_power = 0.0;
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			local_input_power = input_percent * max_input_mag * exp(pow((mesh_x[i][j][0] - input_location[0]), 2.0) * exp_const + pow((mesh_y[i][j][0] - input_location[1]), 2.0) * exp_const);
			if (local_input_power < 0.01 * max_input_mag)
			{
				input_mesh[i][j] = 0.0;
			}
			else
			{
				input_mesh[i][j] = local_input_power;
			}
		}
	}
}

/** Get smooth 3D perturbation over input fields
* @ param array used to determine size of output mesh
* @ param maximum magnitude of perturbation
* @ return sum of size_array and smooth continuous perturbation of magnitude delta
*/
vector<vector<vector<double> > > Finite_Element_Solver::get_perturbation(vector<vector<vector<double> > > size_array, double delta)
{
	// Get magnitude and biases
	double mag_1 = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;
	double mag_2 = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;
	double mag_3 = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;
	double bias_1 = 4.0 * M_PI * (double)rand()/(double)RAND_MAX - 2.0 * M_PI;
	double bias_2 = 4.0 * M_PI * (double)rand()/(double)RAND_MAX - 2.0 * M_PI;
	double bias_3 = 4.0 * M_PI * (double)rand()/(double)RAND_MAX - 2.0 * M_PI;
	double min_mag = (double)rand()/(double)RAND_MAX;
	double max_mag = (double)rand()/(double)RAND_MAX;
	double min_x_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;
	double max_x_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;
	double min_y_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;
	double max_y_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;
	double min_z_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;
	double max_z_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;

	// Get x*y*z over perturbation field
	double x, y, z, xyz;
	double scale = 0.0;
	vector<vector<vector<double> > > perturbation = vector<vector<vector<double> > >(size_array.size(), vector<vector<double> >(size_array[0].size(), vector<double>(size_array[0][0].size(), 0.0)));
	for (unsigned int i = 0; i < size_array.size(); i++)
	{
		x = -2.0*min_mag+min_x_bias + (2.0*max_mag+max_x_bias + 2.0*min_mag-min_x_bias) * ((double)i / (double)size_array.size());
		for (unsigned int j = 0; j < size_array[0].size(); j++)
		{
			y = -2.0*min_mag+min_y_bias + (2.0*max_mag+max_y_bias + 2.0*min_mag-min_y_bias) * ((double)j / (double)size_array[0].size());
			for (unsigned int k = 0; k < size_array[0][0].size(); k++)
			{
				z =-2.0*min_mag+min_z_bias + (2.0*max_mag+max_z_bias + 2.0*min_mag-min_z_bias) * ((double)k / (double)size_array[0][0].size());
				xyz = x * y * z;
				perturbation[i][j][k] = mag_1 * sin(xyz + bias_1) + mag_2 * sin(2.0*xyz + bias_2) + mag_3 * sin(3.0*xyz + bias_3);
				if (abs(perturbation[i][j][k]) > scale)
				{
					scale = abs(perturbation[i][j][k]);
				}
			}
		}
	}

	// Scale the perturbations and sum them to the original array
	for (unsigned int i = 0; i < size_array.size(); i++)
	{
		for (unsigned int j = 0; j < size_array[0].size(); j++)
		{
			for (unsigned int k = 0; k < size_array[0][0].size(); k++)
			{
				perturbation[i][j][k] = size_array[i][j][k] + (delta * perturbation[i][j][k]) / scale;
			}
		}
	}

	// Return perturbed array
	return perturbation;
}

/** Step the input through time
* @param The raw NN x location rate command
* @param The raw NN y location rate command
* @param The raw NN magnitude command
*/
void Finite_Element_Solver::step_input(double x_loc_rate_action, double y_loc_rate_action, double mag_action)
{
	// Convert the raw PPO x command to usable, clipped x location rate command
	double cmd_x = loc_rate_offset + loc_rate_scale * x_loc_rate_action;
	cmd_x = cmd_x > max_input_loc_rate ? max_input_loc_rate : cmd_x;
	cmd_x = cmd_x < -max_input_loc_rate ? -max_input_loc_rate : cmd_x;

	// Convert the raw PPO y command to usable, clipped y location rate command
	double cmd_y = loc_rate_offset + loc_rate_scale * y_loc_rate_action;
	cmd_y = cmd_y > max_input_loc_rate ? max_input_loc_rate : cmd_y;
	cmd_y = cmd_y < -max_input_loc_rate ? -max_input_loc_rate : cmd_y;

	// Update the input's x location from the converted location rate commands
	input_location[0] = input_location[0] + cmd_x * time_step;
	input_location[0] = input_location[0] > max_input_x_loc ? max_input_x_loc : input_location[0];
	input_location[0] = input_location[0] < min_input_x_loc ? min_input_x_loc : input_location[0];

	// Update the input's y location from the converted location rate commands
	input_location[1] = input_location[1] + cmd_y * time_step;
	input_location[1] = input_location[1] > max_input_y_loc ? max_input_y_loc : input_location[1];
	input_location[1] = input_location[1] < min_input_y_loc ? min_input_y_loc : input_location[1];

	// Convert the raw PPO command to a usable, clipped input percent command
	double input_percent_command = mag_offset + mag_scale * mag_action;
	input_percent_command = input_percent_command > 1.0 ? 1.0 : input_percent_command;
	input_percent_command = input_percent_command < 0.0 ? 0.0 : input_percent_command;

	// Update the input's magnitude from the converted input percent command
	if (input_percent_command > input_percent)
	{
		input_percent = input_percent + input_mag_percent_rate * time_step;
		input_percent = input_percent > input_percent_command ? input_percent_command : input_percent;
	}
	else if (input_percent_command < input_percent)
	{
		input_percent = input_percent - input_mag_percent_rate * time_step;
		input_percent = input_percent < input_percent_command ? input_percent_command : input_percent;
	}

	// Update the input wattage mesh
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < num_vert_length; i++)
	for (int j = 0; j < num_vert_width; j++)
	{
		double local_input_power = input_percent * max_input_mag * exp(pow((mesh_x[i][j][0] - input_location[0]), 2.0) * exp_const + pow((mesh_y[i][j][0] - input_location[1]), 2.0) * exp_const);
		if (local_input_power < 0.01 * max_input_mag)
		{
			input_mesh[i][j] = 0.0;
		}
		else
		{
			input_mesh[i][j] = local_input_power;
		}
	}
}

/** Calculates the cure rate at every point in the 3D mesh and uses this data to update the cure, temperature, and front meshes
*/
void Finite_Element_Solver::step_meshes()
{
	// Front mesh variables
	const vector<vector<double> > prev_front_loc(front_loc);
	const vector<vector<double> > prev_last_move(time_front_last_moved);

	// Temperature mesh variables
	const vector<vector<vector<double> > > prev_temp(temp_mesh);

	// Update the mesh
	#pragma omp parallel for collapse(3)
	for (unsigned int i = 0; i < mesh_x.size(); i++)
	for (unsigned int j = 0; j < mesh_x[0].size(); j++)
	for (unsigned int k = 0; k < mesh_x[0][0].size(); k++)
	{
		// Calculate the cure rate
		double cure_rate = 0.0;
		if (use_DCPD_GC1)
		{
			cure_rate = DCPD_GC1_pre_exponential * exp(-DCPD_GC1_activiation_energy / (gas_const * prev_temp[i][j][k])) *
			pow((1.0 - cure_mesh[i][j][k]), DCPD_GC1_model_fit_order) * 
			(1.0 + DCPD_GC1_autocatalysis_const * cure_mesh[i][j][k]);
		}
		else if (use_DCPD_GC2)
		{
			cure_rate = DCPD_GC2_pre_exponential * exp(-DCPD_GC2_activiation_energy / (gas_const * prev_temp[i][j][k])) *  
			pow((1.0 - cure_mesh[i][j][k]), DCPD_GC2_model_fit_order) * 
			pow(cure_mesh[i][j][k], DCPD_GC2_m_fit) * 
			(1.0 / 1+(1.0 + exp(DCPD_GC2_diffusion_const*(cure_mesh[i][j][k] - DCPD_GC2_critical_cure))));
		}
		else if (use_COD)
		{
			cure_rate = COD_pre_exponential * exp(-COD_activiation_energy / (gas_const * prev_temp[i][j][k])) *  
			pow((1.0 - cure_mesh[i][j][k]), COD_model_fit_order) * 
			pow(cure_mesh[i][j][k], COD_m_fit);
		}

		// Update the cure mesh
		cure_rate = isnan(cure_rate) ? ((1.0 - cure_mesh[i][j][k]) / time_step) : cure_rate;
		cure_rate = (isinf(cure_rate) || (cure_rate > ((1.0 - cure_mesh[i][j][k]) / time_step))) ? ((1.0 - cure_mesh[i][j][k]) / time_step) : cure_rate;
		cure_rate = cure_rate < 1.0e-7 ? 0.0 : cure_rate;
		cure_mesh[i][j][k] = cure_mesh[i][j][k] + cure_rate * time_step;
		cure_mesh[i][j][k] = cure_mesh[i][j][k] > 1.0 ? 1.0 : cure_mesh[i][j][k];

		// Update the front location and either temperature or velocity
		if ((cure_mesh[i][j][k] >= 0.80) && (front_loc[j][k] <= mesh_x[i][j][k]))
		{
			front_loc[j][k] = mesh_x[i][j][k];
			int search_diameter = (int) round((double) num_vert_length * 0.025);
			int min_search_ind = i - search_diameter + 1;
			int max_search_ind = i;
			if (min_search_ind < 0)
			{
				max_search_ind -= min_search_ind;
				min_search_ind = 0;
			}
			front_temp[j][k] = 0.0;
			for (int ind = min_search_ind; ind <= max_search_ind; ind++)
			{
				front_temp[j][k] += temp_mesh[ind][j][k];
			}
			front_temp[j][k] = front_temp[j][k] / search_diameter;
			if (front_loc[j][k] >= 0.99*length)
			{
				front_vel[j][k] = 0.0;
			}
			else if (prev_last_move[j][k] != 0.0 && front_loc[j][k] != prev_front_loc[j][k])
			{
				front_vel[j][k] = (front_loc[j][k] - prev_front_loc[j][k]) / (current_time - prev_last_move[j][k]);
				time_front_last_moved[j][k] = current_time;
			}
			else if (prev_last_move[j][k] == 0.0 && front_loc[j][k] != prev_front_loc[j][k])
			{
				time_front_last_moved[j][k] = current_time;
			}
		}

		// Temperature variables
		double dT2_dx2;
		double dT2_dy2;
		double dT2_dz2;
		double left_flux;
		double right_flux;
		double front_flux;
		double back_flux;
		double top_flux;
		double bottom_flux;

		// Calculate the second derivative of temperature wrt x
		if (i != 0 && i != mesh_x.size()-1)
		{
			dT2_dx2 = (prev_temp[i+1][j][k] - 2.0*prev_temp[i][j][k] + prev_temp[i-1][j][k]) / (x_step*x_step);
		}
		else
		{
			// Boundary conditions
			if (i == 0)
			{
				if (current_time >= trigger_time && current_time < trigger_time + trigger_duration)
				{
					left_flux = htc*(prev_temp[i][j][k]-ambient_temperature) - trigger_flux;
				}
				else
				{
					left_flux = htc*(prev_temp[i][j][k]-ambient_temperature);
				}
				dT2_dx2 = 2.0*(prev_temp[i+1][j][k]-prev_temp[i][j][k]-(x_step*left_flux/thermal_conductivity))/(x_step*x_step);
			}
			if (i == mesh_x.size()-1)
			{
				right_flux = htc*(prev_temp[i][j][k]-ambient_temperature);
				dT2_dx2 = 2.0*(prev_temp[i-1][j][k]-prev_temp[i][j][k]-(x_step*right_flux/thermal_conductivity))/(x_step*x_step);
			}
		}

		// Calculate the second derivative of temperature wrt y
		if (j != 0 && j != mesh_x[0].size()-1)
		{
			dT2_dy2 = (prev_temp[i][j+1][k] - 2.0*prev_temp[i][j][k] + prev_temp[i][j-1][k]) / (y_step*y_step);
		}
		else
		{
			// Boundary conditions
			if (j == 0)
			{
				front_flux = htc*(prev_temp[i][j][k]-ambient_temperature);
				dT2_dy2 = 2.0*(prev_temp[i][j+1][k]-prev_temp[i][j][k]-(y_step*front_flux/thermal_conductivity))/(y_step*y_step);
			}
			if (j == mesh_x[0].size()-1)
			{
				back_flux = htc*(prev_temp[i][j][k]-ambient_temperature);
				dT2_dy2 = 2.0*(prev_temp[i][j-1][k]-prev_temp[i][j][k]-(y_step*back_flux/thermal_conductivity))/(y_step*y_step);
			}
		}

		// Calculate the second derivative of temperature wrt z
		if (k != 0 && k != mesh_x[0][0].size()-1)
		{
			dT2_dz2 = (prev_temp[i][j][k+1] - 2.0*prev_temp[i][j][k] + prev_temp[i][j][k-1]) / (z_step*z_step);
		}
		else
		{
			// Boundary conditions
			if (k == 0)
			{
				top_flux = htc*(prev_temp[i][j][k]-ambient_temperature) - input_mesh[i][j];
				dT2_dz2 = 2.0*(prev_temp[i][j][k+1]-prev_temp[i][j][k]-(z_step*top_flux/thermal_conductivity))/(z_step*z_step);
			}
			if (k == mesh_x[0][0].size()-1)
			{
				bottom_flux = htc*(prev_temp[i][j][k]-ambient_temperature);
				dT2_dz2 = 2.0*(prev_temp[i][j][k-1]-prev_temp[i][j][k]-(z_step*bottom_flux/thermal_conductivity))/(z_step*z_step);
			}
		}

		// Update the temperature field
		double temp_rate = thermal_diffusivity*(dT2_dx2+dT2_dy2+dT2_dz2)+(enthalpy_of_reaction*cure_rate)/specific_heat;
		temp_mesh[i][j][k] = temp_mesh[i][j][k] + temp_rate * time_step;
	}
}

/**
* Gets the state fed to PPO agent based on temperature, front location, front velocity, and the input
* @param The autoencoder encoded temperature field as a vector of doubles
* @return The normalized state array
*/
vector<double> Finite_Element_Solver::get_state(vector<double> encoded_temp)
{
	// Init state variables
	vector<double> state(get_num_state(), 0.0);

	// Copy the encoded state
	for (int i = 0 ; i  < encoded_size; i++)
	{
		state[i] = encoded_temp[i];
	}

	// Get the coarse front location and velocity data
	if (get_extended_state)
	{
		int curr_state_index = encoded_size;
		int y_width = 0;
		int y_max_index = num_vert_width-1;
		int y_curr_index = 0;
		double avg_loc = 0.0;
		double avg_vel = 0.0;
		for (int j = 0; j < (int)((double)num_vert_width/4.0); j++)
		{
			y_width = (int) floor((y_max_index-y_curr_index) / ((int)((double)num_vert_width/4.0)-j));
			for (int y = y_curr_index; y <= (y_curr_index + y_width); y++)
			{
				avg_loc += front_loc[y][0];
				avg_vel += front_vel[y][0];
			}
			state[curr_state_index] = avg_loc / ((double)(y_width+1)*length);
			state[curr_state_index+(int)((double)num_vert_width/4.0)] = avg_vel / ((double)(y_width+1)*current_target);
			curr_state_index++;
			y_curr_index += y_width;
		}
	}

	// Append the input location and magnitude parameters
	state[get_num_state()-3] = input_location[0] / length;
	state[get_num_state()-2] = input_location[1] / width;
	state[get_num_state()-1] = input_percent;

	// Return the state
	return state;
}

/**
* Solves for the reward fed to the PPO agent based on the reward function parameters, temperature, and front velocity
* @return The calculated reward
*/
double Finite_Element_Solver::get_reward()
{
	// Initialize reward and punishment variables
	double input_punishment;
	double dist_punishment;
	double overage_punishment;
	double integral_punishment;
	double front_shape_punishment;
	double punishment;
	double reward = 0.0;

	// Integrate the temp mesh and get the mean front x location
	double temp_integral = 0.0;
	double mean_location = 0.0;
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			for (int k = 0; k < num_vert_depth; k++)
			{
				temp_integral += temp_mesh[i][j][k];
				mean_location = i == 0 ? mean_location + front_loc[j][k] : mean_location;
			}
		}
	}
	temp_integral = temp_integral * x_step * y_step * z_step;
	mean_location = mean_location / ((double)num_vert_width * (double)num_vert_depth);

	// Find the front's location and velocity mean deviation and max temperature
	double max_front_temp = 0.0;
	double mean_loc_deviation = 0.0;
	double mean_deviation = 0.0;
	for (int j = 0; j < num_vert_width; j++)
	{
		for (int k = 0; k < num_vert_depth; k++)
		{
			mean_loc_deviation += abs(front_loc[j][k] - mean_location);
			if (control_temperature)
			{
				mean_deviation += abs(front_temp[j][k] - current_target);
			}
			else if (control_speed)
			{
				mean_deviation += abs(front_vel[j][k] - current_target);
			}
			max_front_temp = front_temp[j][k] > max_front_temp ? front_temp[j][k] : max_front_temp;
		}
	}
	mean_loc_deviation = mean_loc_deviation / ((double)num_vert_width * (double)num_vert_depth);
	mean_deviation = mean_deviation / ((double)num_vert_width * (double)num_vert_depth * current_target);
	mean_deviation = mean_deviation > 1.0 ? 1.0 : mean_deviation;

	if (control)
	{
		input_punishment = 0.0;
		dist_punishment = 0.0;
	}
	else
	{
		input_punishment = -input_punishment_const * max_reward * input_percent;

		// Calculate dist from front punishment
		double mean_front_loc = 0.0;
		for (int j = 0; j < num_vert_width; j++)
		{
			mean_front_loc += front_loc[j][0];
		}
		mean_front_loc = mean_front_loc / num_vert_width;
		double dist_from_front = abs(mean_front_loc - input_location[0]);
		dist_from_front = dist_from_front <= (1.25 * radius_of_input) ? 0.0 : dist_from_front/length;
		dist_punishment = -dist_punishment_const * max_reward * dist_from_front;
	}

	// Get the integral punishment
	integral_punishment = -integral_punishment_const * max_reward * (1.0 - (max_integral - temp_integral) / integral_delta);

	// Get the front shape punishment
	front_shape_punishment = -front_shape_const * mean_loc_deviation;

	// Get the overage punishment
	overage_punishment = max_front_temp > temperature_limit ? -overage_punishment_const * max_reward * max_front_temp / temperature_limit : 0.0;

	// Get the punishment
	punishment = input_punishment + dist_punishment + integral_punishment + front_shape_punishment + overage_punishment;

	// Get the total reward
	if (control_temperature)
	{
		reward = pow((1.0 - mean_deviation) * temperature_reward_const, 3.0) + punishment;
	}
	else if (control_speed)
	{
		reward = pow((1.0 - mean_deviation) * front_rate_reward_const, 3.0) + punishment;
	}

	return reward;
}

/**
* Steps the environments time and updates the target velocity
* Boolean that determines whether simulation is complete or not
*/
bool Finite_Element_Solver::step_time()
{
	// Update the current time and check for simulation completion
	bool done = (current_index == (int)target_vector.size() - 1);
	if (!done)
	{
		current_time = current_time + time_step;
		current_index = current_index + 1;
		current_target = target_vector[current_index];
	}

	return done;
}
