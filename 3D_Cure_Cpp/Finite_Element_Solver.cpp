#include "Finite_Element_Solver.h"
#define M_PI   3.14159265358979323846264338327950288

using namespace std;

//******************************************************************** CONSTRUCTOR/DESTRUCTOR ********************************************************************//

/**
* Default constructor
*/
Finite_Element_Solver::Finite_Element_Solver()
{
	// Load configuration data
	if (load_config() == 1)
	{
		throw 1;
	}
	
	// Set randomization seed
	srand(time(NULL));

	// Simulation time and target velocity index
	current_time = 0.0;  // Seconds
	current_index = 0;   // Unitless
	
	// Randomize htc and ambient temperature
	htc = default_htc + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * htc_delta;
	ambient_temperature = default_ambient_temperature + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * ambient_temperature_delta;
	htc = htc < 0.0 ? 0.0 : htc;
	ambient_temperature = ambient_temperature < 0.0 ? 0.0 : ambient_temperature;
	
	// Monomer physical parameters
	if (use_DCPD_GC1)
	{
		thermal_diffusivity = DCPD_GC1_thermal_conductivity / (DCPD_GC1_specific_heat * DCPD_GC1_density);
		thermal_conductivity = DCPD_GC1_thermal_conductivity;
		enthalpy_of_reaction = DCPD_GC1_enthalpy_of_reaction;
		specific_heat = DCPD_GC1_specific_heat;
		
		// Calcualte the critical temperature
		cure_critical_temperature = -DCPD_GC1_activiation_energy / (log((pow(1.0-initial_cure,-DCPD_GC1_model_fit_order)*1.0e-3) / ((1.0+DCPD_GC1_autocatalysis_const*initial_cure)*DCPD_GC1_pre_exponential)) * gas_const);
	}
	else if (use_DCPD_GC2)
	{
		thermal_diffusivity = DCPD_GC2_thermal_conductivity / (DCPD_GC2_specific_heat * DCPD_GC2_density);
		thermal_conductivity = DCPD_GC2_thermal_conductivity;
		enthalpy_of_reaction = DCPD_GC2_enthalpy_of_reaction;
		specific_heat = DCPD_GC2_specific_heat;
		
		// Calcualte the critical temperature
		cure_critical_temperature = -DCPD_GC2_activiation_energy / ((log((exp(initial_cure*DCPD_GC2_diffusion_const)/DCPD_GC2_pre_exponential + exp(DCPD_GC2_diffusion_const*DCPD_GC2_critical_cure)/DCPD_GC2_pre_exponential) * pow((1.0-initial_cure), -DCPD_GC2_model_fit_order) * pow(initial_cure, -DCPD_GC2_m_fit))- DCPD_GC2_diffusion_const*DCPD_GC2_critical_cure + log(1.0e-3)) * gas_const);
	}
	else if (use_COD)
	{
		thermal_diffusivity = COD_thermal_conductivity / (COD_specific_heat * COD_density);
		thermal_conductivity = COD_thermal_conductivity;
		enthalpy_of_reaction = COD_enthalpy_of_reaction;
		specific_heat = COD_specific_heat;
		
		// Calcualte the critical temperature
		cure_critical_temperature = 0.0;
	}

	// Determine target value
	int sim_steps = get_steps_per_episode();
	double target = 0.0;
	double randomizing_scale = 0.0;
	if(control_speed)
	{
		target = target_vel;
		randomizing_scale = vel_rand_scale;
	}
	else if(control_temperature)
	{
		target = target_temp;
		randomizing_scale = temp_rand_scale;
	}
	
	// Allocate memory for target vector
	target_vector = new double[sim_steps];
	for(int i = 0; i < sim_steps; i++)
	{
		target_vector[i] = target;
	}
	
	// Modify target vector based on user parameters
	if (random_target)
	{
		double new_target = target + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
		for (int i = 0; i < sim_steps; i++)
		{
			target_vector[i] = new_target;
		}
	}
	else if (target_switch)
	{
		int switch_location = (int) floor((0.20 * (double)rand()/(double)RAND_MAX + 0.40) * (double)(sim_steps - 1));
		double switch_target_1 = target + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
		double switch_target_2 = target + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
		for (int i = 0; i < sim_steps; i++)
		{
			if(i < switch_location)
			{
				target_vector[i] = switch_target_1;
			}
			else
			{
				target_vector[i] = switch_target_2;
			}
		}
	}
	current_target = target_vector[current_index];

	// Disengage trigger if set to off
	if (!trigger)
	{
		trigger_flux = 0.0;
		trigger_time = 0.0;
		trigger_duration = 0.0;
	}

	// Allocate memory space for position mesh and assign mesh values
	mesh_x = new double**[num_vert_length];
	mesh_y = new double**[num_vert_length];
	mesh_z = new double**[num_vert_length];
	for(int i = 0; i < num_vert_length; i++)
	{
		mesh_x[i] = new double*[num_vert_width];
		mesh_y[i] = new double*[num_vert_width];
		mesh_z[i] = new double*[num_vert_width];
		for(int j = 0; j < num_vert_width; j++)
		{
			mesh_x[i][j] = new double[num_vert_depth];
			mesh_y[i][j] = new double[num_vert_depth];
			mesh_z[i][j] = new double[num_vert_depth];
			for(int k = 0; k < num_vert_depth; k++)
			{
				mesh_x[i][j][k] = ((double)i / (double)(num_vert_length - 1)) * length;
				mesh_y[i][j][k] = ((double)j / (double)(num_vert_width - 1)) * width;
				mesh_z[i][j][k] = ((double)k / (double)(num_vert_depth - 1)) * depth;
			}
		}
	}

	// Determine step sizes
	x_step = mesh_x[1][0][0];
	y_step = mesh_y[0][1][0];
	z_step = mesh_z[0][0][1];

	// Allocate memory space for temp and cure mesh
	temp_mesh = new double**[num_vert_length];
	laplacian_mesh = new double**[num_vert_length];
	cure_mesh = new double**[num_vert_length];
	for(int i = 0; i < num_vert_length; i++)
	{
		temp_mesh[i] = new double*[num_vert_width];
		laplacian_mesh[i] = new double*[num_vert_width];
		cure_mesh[i] = new double*[num_vert_width];
		for(int j = 0; j < num_vert_width; j++)
		{
			temp_mesh[i][j] = new double[num_vert_depth];
			laplacian_mesh[i][j] = new double[num_vert_depth];
			cure_mesh[i][j] = new double[num_vert_depth];
			for(int k = 0; k < num_vert_depth; k++)
			{
				temp_mesh[i][j][k] = initial_temperature;
				laplacian_mesh[i][j][k] = 0.0;
				cure_mesh[i][j][k] = initial_cure;
			}
		}
	}

	// Perturb temperature and cure meshes
	perturb_mesh(temp_mesh, initial_temp_delta);
	perturb_mesh(cure_mesh, initial_cure_delta);
	
	// Ensure fine mesh parameters are acceptable (coarse_steps_per_fine_mesh_x must be whole number)
	coarse_steps_per_fine_mesh_x = (int)ceil((length_fine * (double)num_vert_length) / length);
	coarse_steps_per_fine_mesh_y = num_vert_width;
	coarse_steps_per_fine_mesh_z = num_vert_depth;
	length_fine = ((double)coarse_steps_per_fine_mesh_x * length) / ((double)num_vert_length);
	
	// Determine fine mesh resolution
	num_vert_length_fine = coarse_steps_per_fine_mesh_x * fine_steps_per_coarse_step_x;
	num_vert_width_fine = coarse_steps_per_fine_mesh_y * fine_steps_per_coarse_step_y;
	num_vert_depth_fine = coarse_steps_per_fine_mesh_z * fine_steps_per_coarse_step_z;
	 
	// Set front detection parameters
	front_location_indicies_length = 10 * num_vert_width_fine;
	front_filter_alpha = 1.0 - exp(-time_step/front_time_const);
	 
	// Allocate memory space for fine temperature and cure mesh
	temp_mesh_fine = new double**[num_vert_length_fine];
	laplacian_mesh_fine = new double**[num_vert_length_fine];
	cure_mesh_fine = new double**[num_vert_length_fine];
	for(int i = 0; i < num_vert_length_fine; i++)
	{
		temp_mesh_fine[i] = new double*[num_vert_width_fine];
		laplacian_mesh_fine[i] = new double*[num_vert_width_fine];
		cure_mesh_fine[i] = new double*[num_vert_width_fine];
		for(int j = 0; j < num_vert_width_fine; j++)
		{
			temp_mesh_fine[i][j] = new double[num_vert_depth_fine];
			laplacian_mesh_fine[i][j] = new double[num_vert_depth_fine];
			cure_mesh_fine[i][j] = new double[num_vert_depth_fine];
		}
	}
	
	// Get the step size of the fine mesh
	x_step_fine = (1.0 / (double)(num_vert_length_fine - 1)) * length_fine;
	y_step_fine = (1.0 / (double)(num_vert_width_fine - 1)) * width;
	z_step_fine = (1.0 / (double)(num_vert_depth_fine - 1)) * depth;

	// Copy over the coarse mesh to the fine mesh
	copy_coarse_to_fine();
	
	// Handle fine time stepping parameters
	time_step_fine = time_step / (double)fine_time_steps_per_coarse;

	// Init front mesh and parameters
	front_curve = new double*[2];
	front_curve[0] = new double[front_location_indicies_length];
	front_curve[1] = new double[front_location_indicies_length];
	for(int i = 0; i < front_location_indicies_length; i++)
	{
		front_curve[0][i] = -1.0;
		front_curve[1][i] = -1.0;
	}
	
	threadwise_front_curve = new double**[omp_get_max_threads()];
	for(int i = 0; i < omp_get_max_threads(); i++)
	{
		threadwise_front_curve[i] = new double*[2];
		threadwise_front_curve[i][0] = new double[front_location_indicies_length];
		threadwise_front_curve[i][1] = new double[front_location_indicies_length];
	}
	
	front_mean_x_loc = 0.0;
	front_shape_param = 1.0;
	front_vel = 0.0;
	front_temp = initial_temperature;

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
	input_location = new double[2];
	input_location[0] = 0.0;
	input_location[1] = 0.0;
	
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

	// Allocate memory space for input mesh
	input_mesh = new double*[num_vert_length];
	for(int i = 0; i < num_vert_length; i++)
	{
		input_mesh[i] = new double[num_vert_width];
		for(int j = 0; j < num_vert_width; j++)
		{
			input_mesh[i][j] = 0.0;
		}
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
	
	// Allocate memory for coarse BCs
	lr_bc_temps = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		lr_bc_temps[i] = new double*[num_vert_width];
		for(int j = 0; j < num_vert_width; j++)
		{
			lr_bc_temps[i][j] = new double[num_vert_depth];
		}
	}
	fb_bc_temps = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		fb_bc_temps[i] = new double*[num_vert_length];
		for(int j = 0; j < num_vert_length; j++)
		{
			fb_bc_temps[i][j] = new double[num_vert_depth];
		}
	}
	tb_bc_temps = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		tb_bc_temps[i] = new double*[num_vert_length];
		for(int j = 0; j < num_vert_length; j++)
		{
			tb_bc_temps[i][j] = new double[num_vert_width];
		}
	}
	
	// Allocate memory for fine BCs
	lr_bc_temps_fine = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		lr_bc_temps_fine[i] = new double*[num_vert_width_fine];
		for(int j = 0; j < num_vert_width_fine; j++)
		{
			lr_bc_temps_fine[i][j] = new double[num_vert_depth_fine];
		}
	}
	fb_bc_temps_fine = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		fb_bc_temps_fine[i] = new double*[num_vert_length_fine];
		for(int j = 0; j < num_vert_length_fine; j++)
		{
			fb_bc_temps_fine[i][j] = new double[num_vert_depth_fine];
		}
	}
	tb_bc_temps_fine = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		tb_bc_temps_fine[i] = new double*[num_vert_length_fine];
		for(int j = 0; j < num_vert_length_fine; j++)
		{
			tb_bc_temps_fine[i][j] = new double[num_vert_width_fine];
		}
	}
}

/**
* Destructor
*/
Finite_Element_Solver::~Finite_Element_Solver()
{
	// 1D arrays
	delete[] input_location;
	delete[] target_vector;
	
	// Input mesh
	for(int i = 0; i != num_vert_length; ++i)
	{
		delete[] input_mesh[i];
	}
	delete[] input_mesh;
	
	// Front curve data
	for(int i = 0; i != 2; ++i)
	{
		delete[] front_curve[i];
	}
	delete[] front_curve;
	
	// Threadwise front curve
	for(int i = 0; i != omp_get_max_threads(); ++i)
	{
		for(int j = 0; j != 2; ++j)
		{
			delete[] threadwise_front_curve[i][j];
		}
		delete[] threadwise_front_curve[i];
	}
	delete[] threadwise_front_curve;
	
	// Coarse mesh
	for(int i = 0; i != num_vert_length; ++i)
	{
		for(int j = 0; j != num_vert_width; ++j)
		{
			delete[] cure_mesh[i][j];
			delete[] temp_mesh[i][j];
			delete[] laplacian_mesh[i][j];
			delete[] mesh_x[i][j];
			delete[] mesh_y[i][j];
			delete[] mesh_z[i][j];
		}
		delete[] cure_mesh[i];
		delete[] temp_mesh[i];
		delete[] laplacian_mesh[i];
		delete[] mesh_x[i];
		delete[] mesh_y[i];
		delete[] mesh_z[i];
	}
	delete[] cure_mesh;
	delete[] temp_mesh;
	delete[] laplacian_mesh;
	delete[] mesh_x;
	delete[] mesh_y;
	delete[] mesh_z;

	// Fine mesh
	for(int i = 0; i != num_vert_length_fine; ++i)
	{
		for(int j = 0; j != num_vert_width_fine; ++j)
		{
			delete[] cure_mesh_fine[i][j];
			delete[] temp_mesh_fine[i][j];
			delete[] laplacian_mesh_fine[i][j];
		}
		delete[] cure_mesh_fine[i];
		delete[] temp_mesh_fine[i];
		delete[] laplacian_mesh_fine[i];
	}
	delete[] cure_mesh_fine;
	delete[] temp_mesh_fine;
	delete[] laplacian_mesh_fine;
	
	// Coarse BCs
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_vert_width; ++j)
		{
			delete[] lr_bc_temps[i][j];
		}
		delete[] lr_bc_temps[i];
	}
	delete[] lr_bc_temps;
	
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_vert_length; ++j)
		{
			delete[] fb_bc_temps[i][j];
		}
		delete[] fb_bc_temps[i];
	}
	delete[] fb_bc_temps;
	
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_vert_length; ++j)
		{
			delete[] tb_bc_temps[i][j];
		}
		delete[] tb_bc_temps[i];
	}
	delete[] tb_bc_temps;
	
	// Fine BCs
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_vert_width_fine; ++j)
		{
			delete[] lr_bc_temps_fine[i][j];
		}
		delete[] lr_bc_temps_fine[i];
	}
	delete[] lr_bc_temps_fine;
	
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_vert_length_fine; ++j)
		{
			delete[] fb_bc_temps_fine[i][j];
		}
		delete[] fb_bc_temps_fine[i];
	}
	delete[] fb_bc_temps_fine;
	
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_vert_length_fine; ++j)
		{
			delete[] tb_bc_temps_fine[i][j];
		}
		delete[] tb_bc_temps_fine[i];
	}
	delete[] tb_bc_temps_fine;
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
	vector<vector<double>> ret_val = vector<vector<double>>(num_vert_length, vector<double>(num_vert_width, 0.0));
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
	vector<vector<double>> ret_val = vector<vector<double>>(num_vert_length, vector<double>(num_vert_width, 0.0));
	for (int i = 0; i < num_vert_length; i++)
	{
		for (int j = 0; j < num_vert_width; j++)
		{
			ret_val[i][j] = mesh_y[i][j][0];
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
double Finite_Element_Solver::get_max_input_mag_percent_rate()
{
	return max_input_mag_percent_rate;
}

/**
* Gets the maximum possible input single axis movement rate
* @return the maximum possible input single axis movement rate in m/s
*/
double Finite_Element_Solver::get_max_input_loc_rate()
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
	vector<double> ret_val = vector<double>(2, 0.0);
	ret_val[0] = input_location[0];
	ret_val[1] = input_location[1];
	
	return ret_val;
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
* Gets the number of discrete time steps per episode
* @return The length of the target array
*/
int Finite_Element_Solver::get_steps_per_episode()
{
	return (int)round(sim_duration / time_step);
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
	vector<vector<double>> ret_val = vector<vector<double>>(num_vert_length, vector<double>(num_vert_width, 0.0));
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
	vector<vector<double>> ret_val = vector<vector<double>>(num_vert_length, vector<double>(num_vert_width, 0.0));
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
	vector<vector<double>> ret_val = vector<vector<double>>(num_vert_length, vector<double>(num_vert_width, 0.0));
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
* Gets the current front curve
* @return Front curve of standard length. If front does not populate entire vector, end padded with -1.0
*/
vector<vector<double>> Finite_Element_Solver::get_front_curve()
{
	vector<vector<double>> ret_val = vector<vector<double>>(2, vector<double>(front_location_indicies_length, 0.0));
	bool done = false;
	for(int i = 0; i < front_location_indicies_length; i++)
	{
		if(!done)
		{
			ret_val[0][i] = front_curve[0][i];
			ret_val[1][i] = front_curve[1][i];
			done = (front_curve[0][i] < 0.0) || (front_curve[1][i] < 0.0);
		}
		else
		{
			ret_val[0][i] = -1.0;
			ret_val[1][i] = -1.0;
		}
	}
	return ret_val;
}

/**
* Gets the current front velocity
* @return The current mean front velocity
*/
double Finite_Element_Solver::get_front_vel()
{
	return front_vel;
}

/**
* Gets the current front temperature
* @return The current front mean temperature
*/
double Finite_Element_Solver::get_front_temp()
{
	return front_temp;
}

/**
* Gets the current front shape parameters
* @return The current front shape parameter (x stdev normalized against quarter fine mesh length)
*/
double Finite_Element_Solver::get_front_shape_param()
{
	return front_shape_param;
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
	
	cout << "\nTarget(\n";
	if (control_speed)
	{
		cout << "  (Type): Front speed\n";
		if (const_target)
		{
			cout << "  (Style): Constant target\n";
			cout << "  (Target): " << 1000.0*target_vel << " mm/s\n";
		}
		else if (random_target)
		{
			cout << "  (Style): Random target\n";
			cout << "  (Target): " << 1000.0*target_vel << " mm/s +- " << 1000.0*vel_rand_scale << "mm/s\n";
		}
		else
		{
			cout << "  (Style): Switch target\n";
			cout << "  (Target): " << 1000.0*target_vel << " mm/s +- " << 1000.0*vel_rand_scale << "mm/s\n";
		}
	}
	else
	{
		cout << "  (Type): Front temperature\n";
		if (const_target)
		{
			cout << "  (Style): Constant target\n";
			cout << "  (Target): " << target_temp-273.15 << " C\n";
		}
		else if (random_target)
		{
			cout << "  (Style): Random target\n";
			cout << "  (Target): " << target_temp-273.15 << " C +- " << temp_rand_scale << "C\n";
		}
		else
		{
			cout << "  (Style): Switch target\n";
			cout << "  (Target): " << target_temp-273.15 << " C +- " << temp_rand_scale << "C\n";
		}
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
	cout << "  (Initial Temperature): " << initial_temperature-273.15 << " C +- " << initial_temp_delta << " C\n";
	cout << "  (Initial Cure): " << initial_cure << " +- " << initial_cure_delta << "\n";
	cout << ")\n";
	
	// Coarse Mesh
	cout << "\nCoarse Mesh(\n";
	cout << "  (Dimensions): " << 1000.0*length << " x " << 1000.0*width << " x " << 1000.0*depth << " mm\n";
	cout << "  (Grid): " << num_vert_length << " x " << num_vert_width << " x " << num_vert_depth << "\n";
	cout << "  (Time Step): " << 1000.0*time_step << " ms\n";
	cout << ")\n\n";
	
	// Fine Mesh
	cout << "\nFine Mesh(\n";
	cout << "  (Fine Dimensions): " << 1000.0*length_fine << " x " << 1000.0*width << " x " << 1000.0*depth << " mm\n";
	cout << "  (Fine Grid): " << num_vert_length_fine << " x " << num_vert_width_fine << " x " << num_vert_depth_fine << "\n";
	cout << "  (Fine Time Step): " << 1000.0*time_step_fine << " ms\n";
	cout << ")\n\n";
	
	// Environment
	cout << "\nEnvironment(\n";
	cout << "  (Duration): " << sim_duration << " s\n";
	cout << "  (Ambient Temperature): " << default_ambient_temperature-273.15 << " C +- " << ambient_temperature_delta << " C\n";
	cout << "  (HTC): " << default_htc << " W/m^2-K +- " << htc_delta << " W/m^2-K\n";
	cout << ")\n\n";
}

/**
* Resets the environment to initial conditions
*/
void Finite_Element_Solver::reset()
{
	// Simulation time and target velocity index
	current_time = 0.0;
	current_index = 0;

	// Randomize htc and ambient temperature
	htc = default_htc + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * htc_delta;
	ambient_temperature = default_ambient_temperature + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * ambient_temperature_delta;
	htc = htc < 0.0 ? 0.0 : htc;
	ambient_temperature = ambient_temperature < 0.0 ? 0.0 : ambient_temperature;

	// Calculate the target temporal vector and define the current target
	int sim_steps = get_steps_per_episode();
	double target = 0.0;
	double randomizing_scale = 0.0;
	if(control_speed)
	{
		target = target_vel;
		randomizing_scale = vel_rand_scale;
	}
	else if(control_temperature)
	{
		target = target_temp;
		randomizing_scale = temp_rand_scale;
	}
	for(int i = 0; i < sim_steps; i++)
	{
		target_vector[i] = target;
	}
	if (random_target)
	{
		double new_target = target - 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
		for (int i = 0; i < sim_steps; i++)
		{
			target_vector[i] = new_target;
		}
	}
	else if (target_switch)
	{
		int switch_location = (int) floor((0.20 * (double)rand()/(double)RAND_MAX + 0.40) * (double)(sim_steps - 1));
		double switch_target_1 = target + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
		double switch_target_2 = target + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
		for (int i = 0; i < sim_steps; i++)
		{
			if(i < switch_location)
			{
				target_vector[i] = switch_target_1;
			}
			else
			{
				target_vector[i] = switch_target_2;
			}
		}
	}
	current_target = target_vector[current_index];

	// Perturb temp and cure mesh
	for(int i = 0; i < num_vert_length; i++)
	for(int j = 0; j < num_vert_width; j++)
	for(int k = 0; k < num_vert_depth; k++)
	{
		temp_mesh[i][j][k] = initial_temperature;
		laplacian_mesh[i][j][k] = 0.0;
		cure_mesh[i][j][k] = initial_cure;
	}
	perturb_mesh(temp_mesh, initial_temp_delta);
	perturb_mesh(cure_mesh, initial_cure_delta);
	
	// Copy over the coarse mesh to the fine mesh
	copy_coarse_to_fine();
	
	// Init front mesh and parameters
	for(int i = 0; i < front_location_indicies_length; i++)
	{
		front_curve[0][i] = -1.0;
		front_curve[1][i] = -1.0;
	}
	front_mean_x_loc = 0.0;
	front_shape_param = 1.0;
	front_vel = 0.0;
	front_temp = initial_temperature;

	// Input magnitude parameters
	input_percent = (double)rand()/(double)RAND_MAX;

	// Input location parameters
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

	// Determine maximum mesh temperature
	double max_temp = 0.0;
	
	// Get the input reward
	input_reward = input_reward_const * (1.0 - input_percent);

	// Get the overage reward
	overage_reward = max_temp > temperature_limit ? 0.0 : overage_reward_const;

	// Get the front shape reward
	front_shape_reward = front_shape_reward_const * (1.0 - front_shape_param);

	// Get the total reward
	if (control_temperature)
	{
		target_reward = target_reward_const * exp(-0.5 * pow(((front_temp-current_target)/(0.30*current_target)), 2.0));
	}
	else
	{
		target_reward = target_reward_const * exp(-0.5 * pow(((front_vel-current_target)/(0.03*current_target)), 2.0));
	}

	return input_reward+overage_reward+front_shape_reward+target_reward;
}

//******************************************************************** PRIVATE FUNCTIONS ********************************************************************//

/**
* Loads FES parameters from .cfg file
* @return 0 on success, 1 on failure
*/
int Finite_Element_Solver::load_config()
{
	// Load from config file
	ifstream config_file;
	config_file.open("config_files/fes.cfg");
	string config_dump;
	string bool_dump;
	string string_dump;
	if (config_file.is_open())
	{
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> bool_dump;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		if (bool_dump.compare("true")==0)
		{
			control = false;
		}
		else if (bool_dump.compare("false")==0)
		{
			control = true;
		}
		else
		{
			cout << "\nInput configuration not recognized.";
		}
		
		config_file >> config_dump >> bool_dump;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		if (bool_dump.compare("true")==0)
		{
			trigger = true;
		}
		else if (bool_dump.compare("false")==0)
		{
			trigger = false;
		}
		else
		{
			cout << "\nTrigger configuration not recognized.";
		}
		
		config_file >> config_dump >> string_dump;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		if (string_dump.compare("dcpd_gc1")==0)
		{
			use_DCPD_GC1 = true;
			use_DCPD_GC2 = false;
			use_COD = false;
		}
		else if (string_dump.compare("dcpd_gc2")==0)
		{
			use_DCPD_GC1 = false;
			use_DCPD_GC2 = true;
			use_COD = false;
		}
		else if (string_dump.compare("cod")==0)
		{
			use_DCPD_GC1 = false;
			use_DCPD_GC2 = false;
			use_COD = true;
		}
		else
		{
			cout << "\nMaterial configuration not recognized.";
		}
		
		config_file >> config_dump >> string_dump;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		if (string_dump.compare("speed")==0)
		{
			control_speed = true;
			control_temperature = false;
		}
		else if (string_dump.compare("temp")==0)
		{
			control_speed = false;
			control_temperature = true;
		}
		else
		{
			cout << "\nControl configuration not recognized.";
		}
		
		config_file >> config_dump >> string_dump;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		if (string_dump.compare("const")==0)
		{
			const_target = true;
			random_target = false;
			target_switch = false;
		}
		else if (string_dump.compare("rand")==0)
		{
			const_target = false;
			random_target = true;
			target_switch = false;
		}
		else if (string_dump.compare("switch")==0)
		{
			const_target = false;
			random_target = false;
			target_switch = true;
		}
		else
		{
			cout << "\nTarget configuration not recognized.";
		}
		
		config_file >> config_dump >> num_vert_length;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> num_vert_width;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> num_vert_depth;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> length;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> width;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> depth;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> sim_duration;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> time_step;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> length_fine;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> fine_time_steps_per_coarse;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> fine_steps_per_coarse_step_x;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> fine_steps_per_coarse_step_y;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> fine_steps_per_coarse_step_z;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');

		config_file >> config_dump >> temperature_limit;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> target_vel;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> vel_rand_scale;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> target_temp;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> temp_rand_scale;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> front_time_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');

		config_file >> config_dump >> front_min_cure;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');

		config_file >> config_dump >> front_max_cure;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> initial_temperature;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> initial_cure;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> initial_temp_delta;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> initial_cure_delta;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> default_htc;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> default_ambient_temperature;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> htc_delta;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> ambient_temperature_delta;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> trigger_flux;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> trigger_time;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> trigger_duration;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> radius_of_input;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> laser_power;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> max_input_mag_percent_rate;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> max_input_loc_rate;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> input_reward_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> overage_reward_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> front_shape_reward_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		config_file >> config_dump >> target_reward_const;
	}
	else
	{
		cout << "Unable to open config_files/fes.cfg." << endl;
		return 1;
	}
	config_file.close();
	return 0;
}

/** Perturb mesh with smooth, 3D noise
* @ param array being perturbed
* @ param maximum magnitude of perturbation
* @ return sum of size_array and smooth continuous perturbation of magnitude delta
*/
void Finite_Element_Solver::perturb_mesh(double*** arr, double delta)
{
	// Get magnitude and biases
	double mag_1 = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;			// -1.0 to 1.0
	double mag_2 = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;			// -1.0 to 1.0
	double mag_3 = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;			// -1.0 to 1.0
	double bias_1 = 4.0 * M_PI * (double)rand()/(double)RAND_MAX - 2.0 * M_PI;	// -2pi to 2pi
	double bias_2 = 4.0 * M_PI * (double)rand()/(double)RAND_MAX - 2.0 * M_PI;	// -2pi to 2pi
	double bias_3 = 4.0 * M_PI * (double)rand()/(double)RAND_MAX - 2.0 * M_PI;	// -2pi to 2pi
	double min_mag = (double)rand()/(double)RAND_MAX;				// 0.0 to 1.0
	double max_mag = (double)rand()/(double)RAND_MAX;				// 0.0 to 1.0
	double min_x_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;			// -1.0 to 1.0
	double max_x_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;			// -1.0 to 1.0
	double min_y_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;			// -1.0 to 1.0
	double max_y_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;			// -1.0 to 1.0
	double min_z_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;			// -1.0 to 1.0
	double max_z_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;			// -1.0 to 1.0

	// Get x*y*z over perturbation field
	double x, y, z, xyz, perturbation;
	double scale = abs(mag_1) + abs(mag_2) + abs(mag_3);
	
	for (int i = 0; i < num_vert_length; i++)
	for (int j = 0; j < num_vert_width; j++)
	for (int k = 0; k < num_vert_depth; k++)
	{
		x = -2.0*min_mag + min_x_bias + (2.0*max_mag + max_x_bias + 2.0*min_mag - min_x_bias) * ((double)i / (num_vert_length-1));
		y = -2.0*min_mag + min_y_bias + (2.0*max_mag + max_y_bias + 2.0*min_mag - min_y_bias) * ((double)j / (num_vert_width-1));
		z = -2.0*min_mag + min_z_bias + (2.0*max_mag + max_z_bias + 2.0*min_mag - min_z_bias) * ((double)k / (num_vert_depth-1));
		xyz = x * y * z;
		
		perturbation = mag_1 * sin(xyz + bias_1) + mag_2 * sin(2.0*xyz + bias_2) + mag_3 * sin(3.0*xyz + bias_3);
		arr[i][j][k] = arr[i][j][k] + (delta * perturbation) / scale;
	}
}

/** Copies the coarse mesh data to the fine mesh with the fine mesh starting at the right most point.
*/
void Finite_Element_Solver::copy_coarse_to_fine()
{
	// Assign the starting index of the fine and coarse meshes
	fine_mesh_start_x_index = 0;
	coarse_mesh_start_x_index = 0;
	
	// Assign coarse mesh values to their respective fine mesh counterparts
	for(int i = 0; i < num_vert_length_fine; i++)
	for(int j = 0; j < num_vert_width_fine; j++)
	for(int k = 0; k < num_vert_depth_fine; k++)
	{	
		// Determine location in coarse mesh
		int curr_coarse_x_index = (int)floor((double)i / (double)fine_steps_per_coarse_step_x) + coarse_mesh_start_x_index;
		int curr_coarse_y_index = (int)floor((double)j / (double)fine_steps_per_coarse_step_y);
		int curr_coarse_z_index = (int)floor((double)k / (double)fine_steps_per_coarse_step_z);
		
		// Assign coarse values to fine mesh
		temp_mesh_fine[i][j][k] = temp_mesh[curr_coarse_x_index][curr_coarse_y_index][curr_coarse_z_index];
		cure_mesh_fine[i][j][k] = cure_mesh[curr_coarse_x_index][curr_coarse_y_index][curr_coarse_z_index];
	}
	
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
	
	// Determine the approximate mesh x index of the input location
	int input_x_index = (int)round(input_location[0] / x_step) + 1;

	// Convert the raw PPO y command to usable, clipped y location rate command
	double cmd_y = y_loc_rate_action * max_input_loc_rate;
	cmd_y = cmd_y > max_input_loc_rate ? max_input_loc_rate : cmd_y;
	cmd_y = cmd_y < -max_input_loc_rate ? -max_input_loc_rate : cmd_y;

	// Update the input's y location from the converted location rate commands
	input_location[1] = input_location[1] + cmd_y * time_step;
	input_location[1] = input_location[1] > max_input_y_loc ? max_input_y_loc : input_location[1];
	input_location[1] = input_location[1] < min_input_y_loc ? min_input_y_loc : input_location[1];

	// Determine the approximate mesh y index of the input location
	int input_y_index = (int)round(input_location[1] / y_step) + 1;

	// Convert the raw PPO magnitude percent rate command to usable, clipped magnitude percent rate command
	double cmd_mag = mag_percent_rate_action * max_input_mag_percent_rate;
	cmd_mag = cmd_mag > max_input_mag_percent_rate ? max_input_mag_percent_rate : cmd_mag;
	cmd_mag = cmd_mag < -max_input_mag_percent_rate ? -max_input_mag_percent_rate : cmd_mag;

	// Update the input's magnitude from the converted magnitude rate commands
	input_percent = input_percent + cmd_mag * time_step;
	input_percent = input_percent > 1.0 ? 1.0 : input_percent;
	input_percent = input_percent < 0.0 ? 0.0 : input_percent;
	
	// Detemine the range of x and y indices within which the input resides
	int start_x = input_x_index - (int)round((double)radius_of_input/x_step) - 1;
	start_x = start_x < 0 ? 0 : start_x;
	int end_x = input_x_index + (int)round((double)radius_of_input/x_step) + 1;
	end_x = end_x > num_vert_length ? num_vert_length : end_x;
	int start_y = input_y_index - (int)round((double)radius_of_input/y_step) - 1;
	start_y = start_y < 0 ? 0 : start_y;
	int end_y = input_y_index + (int)round((double)radius_of_input/y_step) + 1;
	end_y = end_y > num_vert_width ? num_vert_width : end_y;
	
	// Update the input wattage mesh
	for (int i = start_x; i < end_x; i++)
	for (int j = start_y; j < end_y; j++)
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

/** Slides the fine mesh right by one corase mesh element
*/
void Finite_Element_Solver::slide_fine_mesh_right()
{
	// Ensure fine mesh is not slid off of simulation domain
	int coarse_mesh_x_slice_being_added = coarse_mesh_start_x_index + coarse_steps_per_fine_mesh_x;
	if(coarse_mesh_x_slice_being_added >= num_vert_length)
	{
		return;
	}
	coarse_mesh_start_x_index++;
	
	int i = 0;
	for(int ind = fine_mesh_start_x_index; ind < fine_mesh_start_x_index + fine_steps_per_coarse_step_x; ind++)
	{
		// Wrap the i index around the fine mesh
		if(ind >= num_vert_length_fine)
		{
			i = ind - num_vert_length_fine;
		}
		else
		{
			i = ind;
		}
			
		for(int j = 0; j < num_vert_width_fine; j++)
		for(int k = 0; k < num_vert_depth_fine; k++)
		{
			// Determine location in coarse mesh
			int curr_coarse_y_index = (int)floor((double)j / (double)fine_steps_per_coarse_step_y);
			int curr_coarse_z_index = (int)floor((double)k / (double)fine_steps_per_coarse_step_z);
			
			// Assign coarse values to fine mesh
			temp_mesh_fine[i][j][k] = temp_mesh[coarse_mesh_x_slice_being_added][curr_coarse_y_index][curr_coarse_z_index];
			cure_mesh_fine[i][j][k] = cure_mesh[coarse_mesh_x_slice_being_added][curr_coarse_y_index][curr_coarse_z_index];
		}
	}

	
	// Update the fine mesh starting index
	fine_mesh_start_x_index = i + 1;
}

/** Copies the fine mesh data to the coarse mesh
*/
void Finite_Element_Solver::copy_fine_to_coarse()
{	
	#pragma omp parallel for collapse(3)
	for(int i = 0; i < coarse_steps_per_fine_mesh_x; i++)
	for(int j = 0; j < coarse_steps_per_fine_mesh_y; j++)
	for(int k = 0; k < coarse_steps_per_fine_mesh_z; k++)
	{
		double temp_avg = 0.0;
		double cure_avg = 0.0;
		
		int start_fine_x_index = i * fine_steps_per_coarse_step_x + fine_mesh_start_x_index;
		int start_fine_y_index = j * fine_steps_per_coarse_step_y;
		int start_fine_z_index = k * fine_steps_per_coarse_step_z;
		
		int p = 0;
		for(int ind = start_fine_x_index; ind < start_fine_x_index + fine_steps_per_coarse_step_x; ind++)
		for(int q = start_fine_y_index; q < start_fine_y_index + fine_steps_per_coarse_step_y; q++)
		for(int r = start_fine_z_index; r < start_fine_z_index + fine_steps_per_coarse_step_z; r++)
		{
			// Wrap the i index around the fine mesh
			if(ind >= num_vert_length_fine)
			{
				p = ind - num_vert_length_fine;
			}
			else
			{
				p = ind;
			}
			
			temp_avg += temp_mesh_fine[p][q][r];
			cure_avg += cure_mesh_fine[p][q][r];
		}
		
		temp_mesh[coarse_mesh_start_x_index+i][j][k] = temp_avg / ((double)(fine_steps_per_coarse_step_x*fine_steps_per_coarse_step_y*fine_steps_per_coarse_step_z));
		cure_mesh[coarse_mesh_start_x_index+i][j][k] = cure_avg / ((double)(fine_steps_per_coarse_step_x*fine_steps_per_coarse_step_y*fine_steps_per_coarse_step_z));
	}
}

/** Compute the location in the fine temperature mesh at which the ith temperature value is stored
* @param x location associated index
* @return Index of fine mesh at which the x location index value is stored
*/
int Finite_Element_Solver::get_ind(int i)
{
	int i_access = i + fine_mesh_start_x_index;
	if( i_access >= num_vert_length_fine )
	{
		i_access = i_access - num_vert_length_fine;
	}
	
	return i_access;
}

/** Updates the virtual temperatures outside of the mesh on the left and right faces based on the boundary conditions
*/
void Finite_Element_Solver::update_lr_bc_temps()
{
	// Coarse mesh BCs
	for(int j = 0; j < num_vert_width; j++)
	for(int k = 0; k < num_vert_depth; k++)
	{
		if(coarse_mesh_start_x_index != 0)
		{
			if ((current_time >= trigger_time) && (current_time < trigger_time + trigger_duration))
			{
				lr_bc_temps[0][j][k] = temp_mesh[0][j][k] - (x_step/thermal_conductivity)*(htc*(temp_mesh[0][j][k]-ambient_temperature)-trigger_flux);
			}
			else
			{
				lr_bc_temps[0][j][k] = temp_mesh[0][j][k] - (x_step*htc/thermal_conductivity)*(temp_mesh[0][j][k]-ambient_temperature);
			}
		}
		
		if(coarse_mesh_start_x_index + coarse_steps_per_fine_mesh_x != num_vert_length)
		{
			lr_bc_temps[1][j][k] = temp_mesh[num_vert_length-1][j][k] - (x_step*htc/thermal_conductivity)*(temp_mesh[num_vert_length-1][j][k]-ambient_temperature);
		}
	}
	
	// Fine mesh BCs
	for(int j = 0; j < num_vert_width_fine; j++)
	for(int k = 0; k < num_vert_depth_fine; k++)
	{
		// Determine location in coarse mesh
		int curr_coarse_y_index = (int)floor((double)j / (double)fine_steps_per_coarse_step_y);
		int curr_coarse_z_index = (int)floor((double)k / (double)fine_steps_per_coarse_step_z);
		
		// Left BC if fine mesh is on left edge of domain
		if(coarse_mesh_start_x_index == 0)
		{
			if ((current_time >= trigger_time) && (current_time < trigger_time + trigger_duration))
			{
				lr_bc_temps_fine[0][j][k] = temp_mesh_fine[get_ind(0)][j][k] - (x_step_fine/thermal_conductivity)*(htc*(temp_mesh_fine[get_ind(0)][j][k]-ambient_temperature)-trigger_flux);
			}
			else
			{
				lr_bc_temps_fine[0][j][k] = temp_mesh_fine[get_ind(0)][j][k] - (x_step_fine*htc/thermal_conductivity)*(temp_mesh_fine[get_ind(0)][j][k]-ambient_temperature);
			}
		}
		// Left BC if fine mesh is in middle of domain
		else
		{
			
			lr_bc_temps_fine[0][j][k] = temp_mesh[coarse_mesh_start_x_index-1][curr_coarse_y_index][curr_coarse_z_index];
		}
		
		// Right BC if fine mesh is on right edge of domain
		if(coarse_mesh_start_x_index + coarse_steps_per_fine_mesh_x == num_vert_length)
		{
			lr_bc_temps_fine[1][j][k] = temp_mesh_fine[get_ind(num_vert_length_fine-1)][j][k] - (x_step_fine*htc/thermal_conductivity)*(temp_mesh_fine[get_ind(num_vert_length_fine-1)][j][k]-ambient_temperature);
		}
		// Right BC if fine mesh is in middle of domain
		else
		{
			lr_bc_temps_fine[1][j][k] = temp_mesh[coarse_mesh_start_x_index + coarse_steps_per_fine_mesh_x][curr_coarse_y_index][curr_coarse_z_index];
		}
		
	}
}

/** Updates the virtual temperatures outside of the mesh on the front and back faces based on the boundary conditions
*/
void Finite_Element_Solver::update_fb_bc_temps()
{
	// Coarse mesh BCs
	for(int j = 0; j < num_vert_length; j++)
	for(int k = 0; k < num_vert_depth; k++)
	{
		if( !((j > coarse_mesh_start_x_index) && (j < coarse_mesh_start_x_index+coarse_steps_per_fine_mesh_x-1)) )
		{
			fb_bc_temps[0][j][k] = temp_mesh[j][0][k] - (y_step*htc/thermal_conductivity)*(temp_mesh[j][0][k]-ambient_temperature);
			fb_bc_temps[1][j][k] = temp_mesh[j][num_vert_width-1][k] - (y_step*htc/thermal_conductivity)*(temp_mesh[j][num_vert_width-1][k]-ambient_temperature);
		}
	}
	
	// Fine mesh BCs
	for(int j = 0; j < num_vert_length_fine; j++)
	for(int k = 0; k < num_vert_depth_fine; k++)
	{
		fb_bc_temps_fine[0][get_ind(j)][k] = temp_mesh_fine[get_ind(j)][0][k] - (y_step_fine*htc/thermal_conductivity)*(temp_mesh_fine[get_ind(j)][0][k]-ambient_temperature);
		fb_bc_temps_fine[1][get_ind(j)][k] = temp_mesh_fine[get_ind(j)][num_vert_width_fine-1][k] - (y_step_fine*htc/thermal_conductivity)*(temp_mesh_fine[get_ind(j)][num_vert_width_fine-1][k]-ambient_temperature);
	}
}

/** Updates the virtual temperatures outside of the mesh on the top and bottom faces based on the boundary conditions
*/
void Finite_Element_Solver::update_tb_bc_temps()
{
	// Coarse mesh BCs
	for(int j = 0; j < num_vert_length; j++)
	for(int k = 0; k < num_vert_width; k++)
	{
		if( !((j > coarse_mesh_start_x_index) && (j < coarse_mesh_start_x_index+coarse_steps_per_fine_mesh_x-1)) )
		{
			tb_bc_temps[0][j][k] = temp_mesh[j][k][0] - (z_step/thermal_conductivity)*(htc*(temp_mesh[j][k][0]-ambient_temperature)-input_mesh[j][k]);
			tb_bc_temps[1][j][k] = temp_mesh[j][k][num_vert_depth-1] - (z_step*htc/thermal_conductivity)*(temp_mesh[j][k][num_vert_depth-1]-ambient_temperature);
		}
	}
	
	// Fine mesh BCs
	for(int j = 0; j < num_vert_length_fine; j++)
	for(int k = 0; k < num_vert_width_fine; k++)
	{
		int curr_coarse_x_index = (int)floor((double)j / (double)fine_steps_per_coarse_step_x) + coarse_mesh_start_x_index;
		int curr_coarse_y_index = (int)floor((double)k / (double)fine_steps_per_coarse_step_y);
		
		tb_bc_temps_fine[0][get_ind(j)][k] = temp_mesh_fine[get_ind(j)][k][0] - (z_step_fine/thermal_conductivity)*(htc*(temp_mesh_fine[get_ind(j)][k][0]-ambient_temperature)-input_mesh[curr_coarse_x_index][curr_coarse_y_index]);
		tb_bc_temps_fine[1][get_ind(j)][k] = temp_mesh_fine[get_ind(j)][k][num_vert_depth_fine-1] - (z_step_fine*htc/thermal_conductivity)*(temp_mesh_fine[get_ind(j)][k][num_vert_depth_fine-1]-ambient_temperature);
	}
}

/** Calculates the 7-point stencil 3D laplacian of the coarse mesh. 1st order in z direction, 2nd order in y direction, and 3rd order in x direction
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @return 7-point stencil, 3rd order, 3D laplacian at (i,j,k). 1st order in z direction, 2nd order in y direction, and 3rd order in x direction
*/
double Finite_Element_Solver::get_laplacian(int i, int j, int k)
{
	double T_000 = temp_mesh[i][j][k];
	double d2t_dx2 = 0.0;
	double d2t_dy2 = 0.0;
	double d2t_dz2 = 0.0;
	
	// Right face BC
	if (i==0)
	{
		d2t_dx2 = lr_bc_temps[0][j][k] - 2.0*T_000 + temp_mesh[i+1][j][k];
	}
	// Left face BC
	else if(i==num_vert_length-1)
	{
		d2t_dx2 = temp_mesh[i-1][j][k] - 2.0*T_000 + lr_bc_temps[1][j][k];
	}
	// Bulk material
	else
	{
		int start_p = -3;
		start_p = (i==1) ? -1 : start_p;
		start_p = (i==2) ? -2 : start_p;
		start_p = (i==num_vert_length-3) ? -4 : start_p;
		start_p = (i==num_vert_length-2) ? -5 : start_p;
		for (int p = start_p; p < start_p + 7; p++)
		{
			d2t_dx2 += laplacian_consts_3rd[abs(start_p)-1][p-start_p] * temp_mesh[i+p][j][k];
		}
	}
	d2t_dx2 = d2t_dx2 / (x_step*x_step);
	
	
	// Front face BC
	if (j==0)
	{
		d2t_dy2 = fb_bc_temps[0][i][k] - 2.0*T_000 + temp_mesh[i][j+1][k];
	}
	// Back face BC
	else if(j==num_vert_width-1)
	{
		d2t_dy2 = temp_mesh[i][j-1][k] - 2.0*T_000 + fb_bc_temps[1][i][k];
	}
	// Bulk material
	else
	{
		int start_q = -2;
		start_q = (j==1) ? -1 : start_q;
		start_q = (j==num_vert_width-2) ? -3 : start_q;
		for (int q = start_q; q < start_q + 5; q++)
		{
			d2t_dy2 += laplacian_consts_2nd[abs(start_q)-1][q-start_q] * temp_mesh[i][j+q][k];
		}
	}
	d2t_dy2 = d2t_dy2 / (y_step*y_step);
	
	
	// Top face BC
	if (k==0)
	{
		d2t_dz2 = tb_bc_temps[0][i][j] - 2.0*T_000 + temp_mesh[i][j][k+1];
	}
	// Bottom face BC
	else if(k==num_vert_depth-1)
	{
		d2t_dz2 = temp_mesh[i][j][k-1] - 2.0*T_000 + tb_bc_temps[1][i][j];
	}
	// Bulk material
	else
	{
		d2t_dz2 = temp_mesh[i][j][k-1] - 2.0*T_000 + temp_mesh[i][j][k+1];
	}
	d2t_dz2 = d2t_dz2 / (z_step*z_step);
	
	return d2t_dx2 + d2t_dy2 + d2t_dz2;
}

/** Calculates the 7-point stencil 3D laplacian of the fine mesh. 1st order in z direction, 2nd order in y direction, and 3rd order in x direction
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @return 7-point stencil, 3rd order, 3D laplacian at (i,j,k). 1st order in z direction, 2nd order in y direction, and 3rd order in x direction
*/
double Finite_Element_Solver::get_laplacian_fine(int i, int j, int k)
{
	int i_ind = get_ind(i);
	double T_000 = temp_mesh_fine[i_ind][j][k];
	double d2t_dx2 = 0.0;
	double d2t_dy2 = 0.0;
	double d2t_dz2 = 0.0;
	
	// Right face BC
	if (i==0)
	{
		d2t_dx2 = lr_bc_temps_fine[0][j][k] - 2.0*T_000 + temp_mesh_fine[get_ind(i+1)][j][k];
	}
	// Left face BC
	else if(i==num_vert_length_fine-1)
	{
		d2t_dx2 = temp_mesh_fine[get_ind(i-1)][j][k] - 2.0*T_000 + lr_bc_temps_fine[1][j][k];
	}
	// Bulk material
	else
	{
		int start_p = -3;
		start_p = (i==1) ? -1 : start_p;
		start_p = (i==2) ? -2 : start_p;
		start_p = (i==num_vert_length_fine-3) ? -4 : start_p;
		start_p = (i==num_vert_length_fine-2) ? -5 : start_p;
		for (int p = start_p; p < start_p + 7; p++)
		{
			d2t_dx2 += laplacian_consts_3rd[abs(start_p)-1][p-start_p] * temp_mesh_fine[get_ind(i+p)][j][k];
		}
	}
	d2t_dx2 = d2t_dx2 / (x_step_fine*x_step_fine);
	
	
	// Front face BC
	if (j==0)
	{
		d2t_dy2 = fb_bc_temps_fine[0][i_ind][k] - 2.0*T_000 + temp_mesh_fine[i_ind][j+1][k];
	}
	// Back face BC
	else if(j==num_vert_width_fine-1)
	{
		d2t_dy2 = temp_mesh_fine[i_ind][j-1][k] - 2.0*T_000 + fb_bc_temps_fine[1][i_ind][k];
	}
	// Bulk material
	else
	{
		int start_q = -2;
		start_q = (j==1) ? -1 : start_q;
		start_q = (j==num_vert_width_fine-2) ? -3 : start_q;
		for (int q = start_q; q < start_q + 5; q++)
		{
			d2t_dy2 += laplacian_consts_2nd[abs(start_q)-1][q-start_q] * temp_mesh_fine[i_ind][j+q][k];
		}
	}
	d2t_dy2 = d2t_dy2 / (y_step_fine*y_step_fine);
	
	
	// Top face BC
	if (k==0)
	{
		d2t_dz2 = tb_bc_temps_fine[0][i_ind][j] - 2.0*T_000 + temp_mesh_fine[i_ind][j][k+1];
	}
	// Bottom face BC
	else if(k==num_vert_depth_fine-1)
	{
		d2t_dz2 = temp_mesh_fine[i_ind][j][k-1] - 2.0*T_000 + tb_bc_temps_fine[1][i_ind][j];
	}
	// Bulk material
	else
	{
		d2t_dz2 = temp_mesh_fine[i_ind][j][k-1] - 2.0*T_000 + temp_mesh_fine[i_ind][j][k+1];
	}
	d2t_dz2 = d2t_dz2 / (z_step_fine*z_step_fine);
	
	return d2t_dx2 + d2t_dy2 + d2t_dz2;
}

/** Calculates the cure rate at every point in the 3D mesh and uses this data to update the cure, temperature, and front meshes
*/
void Finite_Element_Solver::step_meshes()
{	
	// Temperature mesh variables
	update_lr_bc_temps();
	update_fb_bc_temps();
	update_tb_bc_temps();

	// Reset front calculation variables
	int num_front_instances = 0;
	double curr_front_mean_x_loc = 0.0;
	front_shape_param = 0.0;
	front_temp = 0.0;
	
	// Update the mesh
	#pragma omp parallel
	{	
		int local_front_index = 0;
		double local_front_temp = 0.0;
		
		//***************************************************************************************** Left coarse *****************************************************************************************//
		// Calculate the laplacian mesh for the left side of the coarse mesh
		#pragma	omp for collapse(3)
		for (int i = 0; i < coarse_mesh_start_x_index; i++)
		for (int j = 0; j < num_vert_width; j++)
		for (int k = 0; k < num_vert_depth; k++)
		{
			laplacian_mesh[i][j][k] = get_laplacian(i, j, k);
		}
		
		
		// Update the temperature and cure mesh for the left side of the coarse mesh
		#pragma	omp for collapse(3) nowait
		for (int i = 0; i < coarse_mesh_start_x_index; i++)
		for (int j = 0; j < num_vert_width; j++)
		for (int k = 0; k < num_vert_depth; k++)
		{
			double cure_rate = 0.0;
				
			// Only calculate the cure rate if curing has started but is incomplete
			if ((temp_mesh[i][j][k] >= cure_critical_temperature) && (cure_mesh[i][j][k] < 1.0))
			{
				if (use_DCPD_GC1)
				{
					cure_rate = DCPD_GC1_pre_exponential * exp(-DCPD_GC1_activiation_energy / (gas_const * temp_mesh[i][j][k])) *
					pow((1.0 - cure_mesh[i][j][k]), DCPD_GC1_model_fit_order) * 
					(1.0 + DCPD_GC1_autocatalysis_const * cure_mesh[i][j][k]);
				}
				else if (use_DCPD_GC2)
				{
					cure_rate = DCPD_GC2_pre_exponential * exp(-DCPD_GC2_activiation_energy / (gas_const * temp_mesh[i][j][k])) *  
					pow((1.0 - cure_mesh[i][j][k]), DCPD_GC2_model_fit_order) * 
					pow(cure_mesh[i][j][k], DCPD_GC2_m_fit) * 
					(1.0 / (1.0 + exp(DCPD_GC2_diffusion_const*(cure_mesh[i][j][k] - DCPD_GC2_critical_cure))));

				}
				else if (use_COD)
				{
					cure_rate = COD_pre_exponential * exp(-COD_activiation_energy / (gas_const * temp_mesh[i][j][k])) *  
					pow((1.0 - cure_mesh[i][j][k]), COD_model_fit_order) * 
					pow(cure_mesh[i][j][k], COD_m_fit);
				}
				
				// Limit cure rate such that a single time step will not yield a degree of cure greater than 1.0
				cure_rate = cure_rate > (1.0 - cure_mesh[i][j][k])/time_step ? (1.0 - cure_mesh[i][j][k])/time_step : cure_rate;
				cure_rate = cure_rate < 0.0 ? 0.0 : cure_rate;	
			}
			
			// Step the cure_mesh
			cure_mesh[i][j][k] = cure_mesh[i][j][k] + time_step * cure_rate;
				
			// Ensure current cure is in expected range
			cure_mesh[i][j][k] = cure_mesh[i][j][k] > 1.0 ? 1.0 : cure_mesh[i][j][k];
			cure_mesh[i][j][k] = cure_mesh[i][j][k] < 0.0 ? 0.0 : cure_mesh[i][j][k];

			// Step temp mesh and ensure current temp is in expected range
			temp_mesh[i][j][k] = temp_mesh[i][j][k] + time_step * (thermal_diffusivity*laplacian_mesh[i][j][k]+(enthalpy_of_reaction*cure_rate)/specific_heat);
			temp_mesh[i][j][k] = temp_mesh[i][j][k] < 0.0 ? 0.0 : temp_mesh[i][j][k];
		}
		
		//***************************************************************************************** Right coarse *****************************************************************************************//
		// Calculate the laplacian mesh for the right side of the coarse mesh
		#pragma	omp for collapse(3)
		for (int i = coarse_mesh_start_x_index + coarse_steps_per_fine_mesh_x; i < num_vert_length; i++)
		for (int j = 0; j < num_vert_width; j++)
		for (int k = 0; k < num_vert_depth; k++)
		{
			laplacian_mesh[i][j][k] = get_laplacian(i, j, k);
		}
		
		
		// Update the temperature and cure mesh for the right side of the coarse mesh
		#pragma	omp for collapse(3) nowait
		for (int i = coarse_mesh_start_x_index + coarse_steps_per_fine_mesh_x; i < num_vert_length; i++)
		for (int j = 0; j < num_vert_width; j++)
		for (int k = 0; k < num_vert_depth; k++)
		{
			double cure_rate = 0.0;
				
			// Only calculate the cure rate if curing has started but is incomplete
			if ((temp_mesh[i][j][k] >= cure_critical_temperature) && (cure_mesh[i][j][k] < 1.0))
			{
				if (use_DCPD_GC1)
				{
					cure_rate = DCPD_GC1_pre_exponential * exp(-DCPD_GC1_activiation_energy / (gas_const * temp_mesh[i][j][k])) *
					pow((1.0 - cure_mesh[i][j][k]), DCPD_GC1_model_fit_order) * 
					(1.0 + DCPD_GC1_autocatalysis_const * cure_mesh[i][j][k]);
				}
				else if (use_DCPD_GC2)
				{
					cure_rate = DCPD_GC2_pre_exponential * exp(-DCPD_GC2_activiation_energy / (gas_const * temp_mesh[i][j][k])) *  
					pow((1.0 - cure_mesh[i][j][k]), DCPD_GC2_model_fit_order) * 
					pow(cure_mesh[i][j][k], DCPD_GC2_m_fit) * 
					(1.0 / (1.0 + exp(DCPD_GC2_diffusion_const*(cure_mesh[i][j][k] - DCPD_GC2_critical_cure))));

				}
				else if (use_COD)
				{
					cure_rate = COD_pre_exponential * exp(-COD_activiation_energy / (gas_const * temp_mesh[i][j][k])) *  
					pow((1.0 - cure_mesh[i][j][k]), COD_model_fit_order) * 
					pow(cure_mesh[i][j][k], COD_m_fit);
				}
				
				// Limit cure rate such that a single time step will not yield a degree of cure greater than 1.0
				cure_rate = cure_rate > (1.0 - cure_mesh[i][j][k])/time_step ? (1.0 - cure_mesh[i][j][k])/time_step : cure_rate;
				cure_rate = cure_rate < 0.0 ? 0.0 : cure_rate;	
			}
			
			// Step the cure_mesh
			cure_mesh[i][j][k] = cure_mesh[i][j][k] + time_step * cure_rate;
				
			// Ensure current cure is in expected range
			cure_mesh[i][j][k] = cure_mesh[i][j][k] > 1.0 ? 1.0 : cure_mesh[i][j][k];
			cure_mesh[i][j][k] = cure_mesh[i][j][k] < 0.0 ? 0.0 : cure_mesh[i][j][k];

			// Step temp mesh and ensure current temp is in expected range
			temp_mesh[i][j][k] = temp_mesh[i][j][k] + time_step * (thermal_diffusivity*laplacian_mesh[i][j][k]+(enthalpy_of_reaction*cure_rate)/specific_heat);
			temp_mesh[i][j][k] = temp_mesh[i][j][k] < 0.0 ? 0.0 : temp_mesh[i][j][k];
				
		}
		
		//***************************************************************************************** Fine mesh *****************************************************************************************//
		for(int subtime_ind = 0; subtime_ind < fine_time_steps_per_coarse; subtime_ind++)
		{		
			// Calculate the laplacian mesh for the fine section
			#pragma	omp for collapse(3)
			for (int i = 0; i < num_vert_length_fine; i++)
			for (int j = 0; j < num_vert_width_fine; j++)
			for (int k = 0; k < num_vert_depth_fine; k++)
			{
				laplacian_mesh_fine[get_ind(i)][j][k] = get_laplacian_fine(i, j, k);
			}
		
			// Update the temperature and cure mesh for the fine mesh
			#pragma	omp for collapse(3) nowait
			for (int i = 0; i < num_vert_length_fine; i++)
			for (int j = 0; j < num_vert_width_fine; j++)
			for (int k = 0; k < num_vert_depth_fine; k++)
			{
				int i_ind = get_ind(i);
				
				double exponential_term = 0.0;
				double cure_rate = 0.0;
				double first_stage_cure_rate = 0.0;
				double second_stage_cure = 0.0;
				double second_stage_cure_rate = 0.0;
				double third_stage_cure = 0.0;
				double third_stage_cure_rate = 0.0;
				double fourth_stage_cure = 0.0;
				double fourth_stage_cure_rate = 0.0;
					
				// Only calculate the cure rate if curing has started but is incomplete
				if ((temp_mesh_fine[i_ind][j][k] >= cure_critical_temperature) && (cure_mesh_fine[i_ind][j][k] < 1.0))
				{
					if (use_DCPD_GC1)
					{
						cure_rate = DCPD_GC1_pre_exponential * exp(-DCPD_GC1_activiation_energy / (gas_const * temp_mesh_fine[i_ind][j][k])) *
						pow((1.0 - cure_mesh_fine[i_ind][j][k]), DCPD_GC1_model_fit_order) * 
						(1.0 + DCPD_GC1_autocatalysis_const * cure_mesh_fine[i_ind][j][k]);
					}
					else if (use_DCPD_GC2)
					{
					
						exponential_term = DCPD_GC2_pre_exponential * exp(-DCPD_GC2_activiation_energy / (gas_const * temp_mesh_fine[i_ind][j][k]));
					
						// Stage 1
						first_stage_cure_rate = exponential_term *  
						pow((1.0 - cure_mesh_fine[i_ind][j][k]), DCPD_GC2_model_fit_order) * 
						pow(cure_mesh_fine[i_ind][j][k], DCPD_GC2_m_fit) * 
						(1.0 / (1.0 + exp(DCPD_GC2_diffusion_const*(cure_mesh_fine[i_ind][j][k] - DCPD_GC2_critical_cure))));
						
						// FE for shallow cure rates
						if( first_stage_cure_rate < 1.5 )
						{
							cure_rate = first_stage_cure_rate;
						}
						
						// RK4 for steep cure rates
						else
						{
							// Stage 2
							second_stage_cure = cure_mesh_fine[i_ind][j][k] + 0.5*time_step_fine*first_stage_cure_rate;
							if(second_stage_cure<1.0)
							{
								second_stage_cure_rate = exponential_term *  
								pow((1.0 - second_stage_cure), DCPD_GC2_model_fit_order) * 
								pow(second_stage_cure, DCPD_GC2_m_fit) * 
								(1.0 / (1.0 + exp(DCPD_GC2_diffusion_const*(second_stage_cure - DCPD_GC2_critical_cure))));
							}
							else {second_stage_cure_rate=0.0;}
							
							// Stage 3
							third_stage_cure = cure_mesh_fine[i_ind][j][k] + 0.5*time_step_fine*second_stage_cure_rate;
							if(third_stage_cure<1.0)
							{
								third_stage_cure_rate = exponential_term *  
								pow((1.0 - third_stage_cure), DCPD_GC2_model_fit_order) * 
								pow(third_stage_cure, DCPD_GC2_m_fit) * 
								(1.0 / (1.0 + exp(DCPD_GC2_diffusion_const*(third_stage_cure - DCPD_GC2_critical_cure))));
							}
							else {third_stage_cure_rate=0.0;}
							
							// Stage 4
							fourth_stage_cure = cure_mesh_fine[i_ind][j][k] + time_step_fine*third_stage_cure_rate;
							if(fourth_stage_cure<1.0)
							{
								fourth_stage_cure = exponential_term *  
								pow((1.0 - fourth_stage_cure), DCPD_GC2_model_fit_order) * 
								pow(fourth_stage_cure, DCPD_GC2_m_fit) * 
								(1.0 / (1.0 + exp(DCPD_GC2_diffusion_const*(fourth_stage_cure - DCPD_GC2_critical_cure))));
							}
							else {fourth_stage_cure=0.0;}
							
							// Apply RK4 algorithm
							cure_rate = (first_stage_cure_rate + 2.0*second_stage_cure_rate + 2.0*third_stage_cure_rate + fourth_stage_cure_rate)/6.0;
						}

					}
					else if (use_COD)
					{
						cure_rate = COD_pre_exponential * exp(-COD_activiation_energy / (gas_const * temp_mesh_fine[i_ind][j][k])) *  
						pow((1.0 - cure_mesh_fine[i_ind][j][k]), COD_model_fit_order) * 
						pow(cure_mesh_fine[i_ind][j][k], COD_m_fit);
					}
					
					// Limit cure rate such that a single time step will not yield a degree of cure greater than 1.0
					cure_rate = cure_rate > (1.0 - cure_mesh_fine[i_ind][j][k])/time_step_fine ? (1.0 - cure_mesh_fine[i_ind][j][k])/time_step_fine : cure_rate;
					cure_rate = cure_rate < 0.0 ? 0.0 : cure_rate;	
				}
				
				// Step the cure_mesh
				cure_mesh_fine[i_ind][j][k] = cure_mesh_fine[i_ind][j][k] + time_step_fine * cure_rate;
					
				// Ensure current cure is in expected range
				cure_mesh_fine[i_ind][j][k] = cure_mesh_fine[i_ind][j][k] > 1.0 ? 1.0 : cure_mesh_fine[i_ind][j][k];
				cure_mesh_fine[i_ind][j][k] = cure_mesh_fine[i_ind][j][k] < 0.0 ? 0.0 : cure_mesh_fine[i_ind][j][k];

				// Step temp mesh and ensure current temp is in expected range
				temp_mesh_fine[i_ind][j][k] = temp_mesh_fine[i_ind][j][k] + time_step_fine * (thermal_diffusivity*laplacian_mesh_fine[i_ind][j][k]+(enthalpy_of_reaction*cure_rate)/specific_heat);
				temp_mesh_fine[i_ind][j][k] = temp_mesh_fine[i_ind][j][k] < 0.0 ? 0.0 : temp_mesh_fine[i_ind][j][k];
				
				if((subtime_ind==(fine_time_steps_per_coarse-1)) && (k==0) && (cure_mesh_fine[i_ind][j][k] >= front_min_cure) && (cure_mesh_fine[i_ind][j][k] <= front_max_cure))
				{
					// Collect front location and shape information
					int thread_num = omp_get_thread_num();
					if(local_front_index < front_location_indicies_length)
					{
						threadwise_front_curve[thread_num][0][local_front_index] = (double)i * x_step_fine + (double)coarse_mesh_start_x_index * x_step;
						threadwise_front_curve[thread_num][1][local_front_index] = (double)j * y_step_fine;
						local_front_index++;
					}
					
					// Search for the highest temperature just behind the front
					if( temp_mesh_fine[i_ind][j][k] > local_front_temp )
					{
						local_front_temp = temp_mesh_fine[i_ind][j][k];
					}
					bool done = false;
					int search_i = i-1;
					while(!done)
					{
						int curr_i_ind = get_ind(search_i);
						if( (search_i>=0) && (temp_mesh_fine[curr_i_ind][j][k] > local_front_temp) )
						{
							local_front_temp = temp_mesh_fine[curr_i_ind][j][k];
							search_i--;
						}
						else
						{
							done = true;
						}
					}
				}
			}
		}
		
		// Reduce collected front location and shape information
		#pragma omp critical
		{     
			int thread_num = omp_get_thread_num();
			int i = 0; 
			
			while( (i < local_front_index-1) && (num_front_instances < front_location_indicies_length-1) )
			{
				front_curve[0][num_front_instances] = threadwise_front_curve[thread_num][0][i];
				front_curve[1][num_front_instances] = threadwise_front_curve[thread_num][1][i];
					
				front_curve[0][num_front_instances] = front_curve[0][num_front_instances] > length ? length : front_curve[0][num_front_instances];
				front_curve[1][num_front_instances] = front_curve[1][num_front_instances] > width ? width : front_curve[1][num_front_instances];
					
				curr_front_mean_x_loc += threadwise_front_curve[thread_num][0][i];
				front_shape_param += threadwise_front_curve[thread_num][0][i]*threadwise_front_curve[thread_num][0][i];
				
				front_temp = local_front_temp > front_temp ? local_front_temp : front_temp;
					
				i++;
				num_front_instances++;
			}
			
			// Mark the end of front curve data
			front_curve[0][num_front_instances] = -1.0;
			front_curve[1][num_front_instances] = -1.0;
			
		}
	}
	
	// Copy fine mesh results to coarse mesh
	copy_fine_to_coarse();
	
	//******************************************************************** Update the front location, velocity, and temperature ********************************************************************//
	if (num_front_instances != 0)
	{
		// Determine quarter fine length normalized front x stdev and mean front x location
		front_shape_param = sqrt((front_shape_param/(double)num_front_instances) - (curr_front_mean_x_loc/(double)num_front_instances)*(curr_front_mean_x_loc/(double)num_front_instances)) / (0.25 * length_fine);
		front_shape_param = front_shape_param > 1.0 ? 1.0 : front_shape_param;
		curr_front_mean_x_loc = curr_front_mean_x_loc / (double)num_front_instances;
		
		// Calculate front speed  through single pole low pass filter and update front location
		front_vel += front_filter_alpha * (abs((curr_front_mean_x_loc - front_mean_x_loc) / time_step) - front_vel);
		front_mean_x_loc = curr_front_mean_x_loc;
		
		// Determine if fine mesh is to be slid to the right
		int avg_front_coarse_ind = (int)floor(curr_front_mean_x_loc / x_step);
		int cen_fine_mesh_coarse_ind = (int)floor( (double)coarse_mesh_start_x_index + (double)coarse_steps_per_fine_mesh_x/2.0 );
		while( avg_front_coarse_ind > cen_fine_mesh_coarse_ind)
		{
			slide_fine_mesh_right();
			avg_front_coarse_ind--;
		}
	}
	
	else
	{

		// Calculate front speed, temp, and update front location
		front_mean_x_loc = 0.0;
		front_shape_param = 1.0;
		front_vel = 0.0;
		front_temp = initial_temperature;
	}
}

/**
* Steps the environments time and updates the target velocity
* Boolean that determines whether simulation is complete or not
*/
bool Finite_Element_Solver::step_time()
{
	// Update the current time and check for simulation completion
	bool done = (current_index == get_steps_per_episode() - 1);
	if (!done)
	{
		current_time = current_time + time_step;
		current_index = current_index + 1;
		current_target = target_vector[current_index];
	}

	return done;
}
