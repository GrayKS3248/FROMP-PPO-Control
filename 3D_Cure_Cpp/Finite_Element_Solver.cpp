#include "Finite_Element_Solver.h"
#define M_PI   3.14159265358979323846264338327950288

using namespace std;


//******************************************************************** CONSTRUCTOR ********************************************************************//

/**
* Default constructor
*/
Finite_Element_Solver::Finite_Element_Solver()
{
	// Set randomization seed
	srand(time(NULL));

	// Simulation time and target velocity index
	current_time = 0.0;  // Seconds
	current_index = 0;   // Unitless
	
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
	
	// Calculate peak irradiance (W/m^2) given total laser power (W)
	double x = 0.0;
	double delta_x = 0.001*radius_of_input;
	double integral = 0.0;
	while(x <= radius_of_input)
	{
		integral += pow(0.01, (x*x)/(radius_of_input*radius_of_input)) * delta_x;
		x += delta_x;
	}
	max_input_mag = laser_power / (4.0 * integral * integral);
	
	// Randomly select laser power percentage
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


//******************************************************************** MESH GETTERS ********************************************************************//

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


//******************************************************************** TIME GETTERS ********************************************************************//

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
* Gets the current time
* @return The time in seconds
*/
double Finite_Element_Solver::get_current_time()
{
	return current_time;
}

//******************************************************************** INPUT PARAMETERS GETTERS ********************************************************************//

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
* Gets the maximum possible input magnitude percent rate
* @return maximum possible input magnitude percent rate in decimal percent per second
*/
const double Finite_Element_Solver::get_max_input_mag_percent_rate()
{
	return max_input_mag_percent_rate;
}

/**
* Gets the maximum possible input single axis movement rate
* @return the maximum possible input single axis movement rate in m/s
*/
const double Finite_Element_Solver::get_max_input_loc_rate()
{
	return max_input_loc_rate;
}


//******************************************************************** INPUT STATE GETTERS ********************************************************************//

/**
* Gets the current input power percent
* @return The power level of the input in percent
*/
double Finite_Element_Solver::get_input_percent()
{
	return input_percent;
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
* Gets the input mesh
* @return The input mesh as a 2D vector in x,y of watts/m^2
*/
vector<vector<double>> Finite_Element_Solver::get_input_mesh()
{
	return input_mesh;
}


//******************************************************************** TARGET GETTERS ********************************************************************//

/**
* Gets the current target
* @return The current target front
*/
double Finite_Element_Solver::get_current_target()
{
	return current_target;
}

/**
* Gets the length of the target array
* @return The length of the target array
*/
int Finite_Element_Solver::get_target_vector_arr_size()
{
	return target_vector.size();
}


//******************************************************************** SIM OPTION GETTERS ********************************************************************//

/**
* Gets whether the speed is controlled or not
* @return Whether the speed is controlled or not
*/
bool Finite_Element_Solver::get_control_speed()
{
	return control_speed;
}


//******************************************************************** TEMP + CURE GETTERS ********************************************************************//

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
	// Get the cure
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


//******************************************************************** FRONT STATE GETTERS ********************************************************************//

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


//******************************************************************** PUBLIC FUNCTIONS ********************************************************************//

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
		cout << "  (Power Rate): " << 1000.0 * laser_power * max_input_mag_percent_rate << " mW/s\n";
		cout << "  (Slew Rate): " << 1000.0 * max_input_loc_rate << " mm/s\n";
	}
	cout << ")\n";
	
	// Target parameters
	double mean = 0.0;
	for (unsigned int i = 0; i < target_vector.size(); i++)
	{
		mean += target_vector[i];
	}
	mean = mean / (double)target_vector.size();
	cout << "\nTarget(\n";
	if (control_speed)
	{
		cout << "  (Type): Front speed\n";
		if (const_target){cout << "  (Style): Constant target\n";}
		else if (random_target){cout << "  (Style): Random target\n";}
		else{cout << "  (Style): Switch target\n";}
		cout << "  (Mean Target): " << mean  << " m/s\n";
	}
	else
	{
		cout << "  (Type): Front temperature\n";
		if (const_target){cout << "  (Style): Constant target\n";}
		else if (random_target){cout << "  (Style): Random target\n";}
		else{cout << "  (Style): Switch target\n";}
		cout << "  (Mean Target): " << mean  << " K\n";
	}

	
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
	cout << "  (Duration): " << sim_duration << " s\n";
	cout << "  (Time Step): " << 1000.0*time_step << " ms\n";
	cout << "  (Ambient Temperature): " << ambient_temperature-273.15 << " C\n";
	cout << ")\n\n";
}

/**
* Resets the environment to initial conditions
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
* Solves for the reward fed to the PPO agent based on the reward function parameters, temperature, and front velocity
* @return The calculated reward
*/
double Finite_Element_Solver::get_reward()
{
	// Initialize reward and punishment variables
	double input_reward;
	double overage_reward;
	double front_shape_reward;
	double target_reward;

	// Find the mean front location and the maximum temperature over the entire mesh
	double mean_front_location = 0.0;
	double max_temperature = 0.0;
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			for (int k = 0; k < num_vert_depth; k++)
			{
				if(i==0) {mean_front_location += front_loc[j][k];}
				if(temp_mesh[i][j][k] > max_temperature) {max_temperature = temp_mesh[i][j][k];}
			}
		}
	}
	mean_front_location = mean_front_location / ((double)num_vert_width * (double)num_vert_depth);
	
	// Find the normalized standard deviation of the front location, mean front speed, mean front temperature
	double stdev_front_location = 0.0;
	double mean_front_speed = 0.0;
	double mean_front_temperature = 0.0;
	for (int j = 0; j < num_vert_width; j++)
	{
		for (int k = 0; k < num_vert_depth; k++)
		{
			stdev_front_location += (front_loc[j][k] - mean_front_location)*(front_loc[j][k] - mean_front_location);
			mean_front_speed +=  front_vel[j][k];
			mean_front_temperature += front_temp[j][k];
		}
	}
	stdev_front_location = sqrt(stdev_front_location / ((double)num_vert_width * (double)num_vert_depth)) / width;
	stdev_front_location = stdev_front_location > 1.0 ? 1.0 : stdev_front_location;
	mean_front_speed = mean_front_speed / ((double)num_vert_width * (double)num_vert_depth);
	mean_front_temperature = mean_front_temperature / ((double)num_vert_width * (double)num_vert_depth);

	// Get the input reward
	input_reward = input_reward_const * (1.0 - input_percent);

	// Get the overage reward
	overage_reward = max_temperature > temperature_limit ? 0.0 : overage_reward_const;

	// Get the front shape reward
	front_shape_reward = front_shape_reward_const * (1.0 - stdev_front_location);

	// Get the total reward
	if (control_temperature)
	{
		target_reward = target_reward_const * exp(-0.5 * pow(((mean_front_temperature-current_target)/(0.30*current_target)), 2.0));
	}
	else
	{
		target_reward = target_reward_const * exp(-0.5 * pow(((mean_front_speed-current_target)/(0.03*current_target)), 2.0));
	}

	return input_reward+overage_reward+front_shape_reward+target_reward;
}


//******************************************************************** PRIVATE FUNCTIONS ********************************************************************//

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
* @param The raw NN magnitude percent rate command
*/
void Finite_Element_Solver::step_input(double x_loc_rate_action, double y_loc_rate_action, double mag_percent_rate_action)
{
	// Convert the raw PPO x command to usable, clipped x location rate command
	double cmd_x = x_loc_rate_action * max_input_loc_rate;
	cmd_x = cmd_x > max_input_loc_rate ? max_input_loc_rate : cmd_x;
	cmd_x = cmd_x < -max_input_loc_rate ? -max_input_loc_rate : cmd_x;

	// Update the input's x location from the converted location rate commands
	input_location[0] = input_location[0] + cmd_x * time_step;
	input_location[0] = input_location[0] > max_input_x_loc ? max_input_x_loc : input_location[0];
	input_location[0] = input_location[0] < min_input_x_loc ? min_input_x_loc : input_location[0];

	// Convert the raw PPO y command to usable, clipped y location rate command
	double cmd_y = y_loc_rate_action * max_input_loc_rate;
	cmd_y = cmd_y > max_input_loc_rate ? max_input_loc_rate : cmd_y;
	cmd_y = cmd_y < -max_input_loc_rate ? -max_input_loc_rate : cmd_y;

	// Update the input's y location from the converted location rate commands
	input_location[1] = input_location[1] + cmd_y * time_step;
	input_location[1] = input_location[1] > max_input_y_loc ? max_input_y_loc : input_location[1];
	input_location[1] = input_location[1] < min_input_y_loc ? min_input_y_loc : input_location[1];

	// Convert the raw PPO magnitude percent rate command to usable, clipped magnitude percent rate command
	double cmd_mag = mag_percent_rate_action * max_input_mag_percent_rate;
	cmd_mag = cmd_mag > max_input_mag_percent_rate ? max_input_mag_percent_rate : cmd_mag;
	cmd_mag = cmd_mag < -max_input_mag_percent_rate ? -max_input_mag_percent_rate : cmd_mag;

	// Update the input's y location from the converted location rate commands
	input_percent = input_percent + cmd_mag * time_step;
	input_percent = input_percent > 1.0 ? 1.0 : input_percent;
	input_percent = input_percent < 0.0 ? 0.0 : input_percent;

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
