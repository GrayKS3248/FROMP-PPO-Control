#include "Finite_Difference_Solver.h"
#define M_PI   3.14159265358979323846264338327950288

using namespace std;

// ================================================================================================= CONSTRUCTOR/DESTRUCTOR ================================================================================================= //
/**
* Default constructor
*/
Finite_Difference_Solver::Finite_Difference_Solver()
{	
	// Set randomization seed
	srand(time(NULL));
	
	// Load configuration data
	if (load_config() == 1)
	{
		throw 1;
	}
	
	
	// ************************************************** MONOMER PARAMS ************************************************** //
	// Monomer physical parameters
	if (monomer_code==1)
	{
		thermal_diffusivity = DCPD_GC1_thermal_conductivity / (DCPD_GC1_specific_heat * DCPD_GC1_density);
		thermal_conductivity = DCPD_GC1_thermal_conductivity;
		enthalpy_of_reaction = DCPD_GC1_enthalpy_of_reaction;
		specific_heat = DCPD_GC1_specific_heat;
		critical_temp = -DCPD_GC1_activiation_energy / (log((pow(1.0-initial_cure,-DCPD_GC1_model_fit_order)*critical_cure_rate) / ((1.0+DCPD_GC1_autocatalysis_const*initial_cure)*DCPD_GC1_pre_exponential)) * gas_const);
	}
	else if (monomer_code==2)
	{
		thermal_diffusivity = DCPD_GC2_thermal_conductivity / (DCPD_GC2_specific_heat * DCPD_GC2_density);
		thermal_conductivity = DCPD_GC2_thermal_conductivity;
		enthalpy_of_reaction = DCPD_GC2_enthalpy_of_reaction;
		specific_heat = DCPD_GC2_specific_heat;
		critical_temp = -DCPD_GC2_activiation_energy / ((log((exp(initial_cure*DCPD_GC2_diffusion_const)/DCPD_GC2_pre_exponential + exp(DCPD_GC2_diffusion_const*DCPD_GC2_critical_cure)/DCPD_GC2_pre_exponential) * pow((1.0-initial_cure), -DCPD_GC2_model_fit_order) * pow(initial_cure, -DCPD_GC2_m_fit))- DCPD_GC2_diffusion_const*DCPD_GC2_critical_cure + log(critical_cure_rate)) * gas_const);
	}
	else if (monomer_code==3)
	{
		thermal_diffusivity = COD_thermal_conductivity / (COD_specific_heat * COD_density);
		thermal_conductivity = COD_thermal_conductivity;
		enthalpy_of_reaction = COD_enthalpy_of_reaction;
		specific_heat = COD_specific_heat;
		critical_temp = 0.0;
	}
	
	
	// ************************************************** COARSE MESH ************************************************** //
	// Generate coarse mesh
	coarse_x_mesh = new double**[num_coarse_vert_x];
	coarse_y_mesh = new double**[num_coarse_vert_x];
	coarse_z_mesh = new double**[num_coarse_vert_x];
	coarse_temp_mesh = new double**[num_coarse_vert_x];
	coarse_laplacian_mesh = new double**[num_coarse_vert_x];
	coarse_cure_mesh = new double**[num_coarse_vert_x];
	for(int i = 0; i < num_coarse_vert_x; i++)
	{
		coarse_x_mesh[i] = new double*[num_coarse_vert_y];
		coarse_y_mesh[i] = new double*[num_coarse_vert_y];
		coarse_z_mesh[i] = new double*[num_coarse_vert_y];
		coarse_temp_mesh[i] = new double*[num_coarse_vert_y];
		coarse_laplacian_mesh[i] = new double*[num_coarse_vert_y];
		coarse_cure_mesh[i] = new double*[num_coarse_vert_y];
		for(int j = 0; j < num_coarse_vert_y; j++)
		{
			coarse_x_mesh[i][j] = new double[num_coarse_vert_z];
			coarse_y_mesh[i][j] = new double[num_coarse_vert_z];
			coarse_z_mesh[i][j] = new double[num_coarse_vert_z];
			coarse_temp_mesh[i][j] = new double[num_coarse_vert_z];
			coarse_laplacian_mesh[i][j] = new double[num_coarse_vert_z];
			coarse_cure_mesh[i][j] = new double[num_coarse_vert_z];
			for(int k = 0; k < num_coarse_vert_z; k++)
			{
				coarse_x_mesh[i][j][k] = ((double)i / (double)(num_coarse_vert_x - 1)) * coarse_x_len;
				coarse_y_mesh[i][j][k] = ((double)j / (double)(num_coarse_vert_y - 1)) * coarse_y_len;
				coarse_z_mesh[i][j][k] = ((double)k / (double)(num_coarse_vert_z - 1)) * coarse_z_len;
				coarse_temp_mesh[i][j][k] = initial_temp;
				coarse_laplacian_mesh[i][j][k] = 0.0;
				coarse_cure_mesh[i][j][k] = initial_cure;
			}
		}
	}
	
	// Perturb temperature and cure meshes
	perturb_mesh(coarse_temp_mesh, max_initial_temp_deviation);
	perturb_mesh(coarse_cure_mesh, max_initial_cure_deviation);
	
	// Determine coarse step sizes
	coarse_x_step = coarse_x_mesh[1][0][0];
	coarse_y_step = coarse_y_mesh[0][1][0];
	coarse_z_step = coarse_z_mesh[0][0][1];
	
	// Calculate coarse/fine ratios
	coarse_x_steps_per_fine_x_len = (int)ceil( (double)num_coarse_vert_x * (fine_x_len/coarse_x_len) );
	coarse_y_steps_per_fine_y_len = num_coarse_vert_y;
	coarse_z_steps_per_fine_z_len = num_coarse_vert_z;
	
	
	// ************************************************** FINE MESH ************************************************** //
	// Calculate fine x coarse_x_len such that the number of coarse steps per fine mesh x coarse_x_len is a whole number
	fine_x_len = coarse_x_len * ((double)coarse_x_steps_per_fine_x_len/(double)num_coarse_vert_x);
	fine_y_len = coarse_y_len;
	fine_z_len = coarse_z_len;

	// Calculate fine mesh vertices
	num_fine_vert_x = coarse_x_steps_per_fine_x_len * fine_x_steps_per_coarse_x_step;
	num_fine_vert_y = coarse_y_steps_per_fine_y_len * fine_y_steps_per_coarse_y_step;
	num_fine_vert_z = coarse_z_steps_per_fine_z_len * fine_z_steps_per_coarse_z_step;
	
	// Generate fine mesh
	fine_temp_mesh = new double**[num_fine_vert_x];
	fine_laplacian_mesh = new double**[num_fine_vert_x];
	fine_cure_mesh = new double**[num_fine_vert_x];
	for(int i = 0; i < num_fine_vert_x; i++)
	{
		fine_temp_mesh[i] = new double*[num_fine_vert_y];
		fine_laplacian_mesh[i] = new double*[num_fine_vert_y];
		fine_cure_mesh[i] = new double*[num_fine_vert_y];
		for(int j = 0; j < num_fine_vert_y; j++)
		{
			fine_temp_mesh[i][j] = new double[num_fine_vert_z];
			fine_laplacian_mesh[i][j] = new double[num_fine_vert_z];
			fine_cure_mesh[i][j] = new double[num_fine_vert_z];
		}
	}
	
	// Copy coarse mesh initial values to the fine mesh
	copy_coarse_to_fine();
	
	// Get the step size of the fine mesh
	fine_x_step = (1.0 / (double)(num_fine_vert_x - 1)) * fine_x_len;
	fine_y_step = (1.0 / (double)(num_fine_vert_y - 1)) * fine_y_len;
	fine_z_step = (1.0 / (double)(num_fine_vert_z - 1)) * fine_z_len;
	
	
	// ************************************************** TIME ************************************************** //
	// Simulation time and target velocity index
	curr_sim_time = 0.0;
	curr_sim_step = 0;
	fine_time_step = coarse_time_step / (double)fine_time_steps_per_coarse_time_step;
	
	
	// ************************************************** TARGET ************************************************** //
	// Determine target value
	if(control_code==1)
	{
		mean_target = mean_target_speed;
		max_target_deviation = max_target_speed_deviation;
	}
	else if(control_code==2)
	{
		mean_target = mean_target_temp;
		max_target_deviation = max_target_temp_deviation;
	}
	
	// Generate target array
	target_arr = new double[get_num_sim_steps()];
	if(target_code==1)
	{
		for(int i = 0; i < get_num_sim_steps(); i++)
		{
			target_arr[i] = mean_target;
		}
	}
	else if (target_code==2)
	{
		double target_deviation = 2.0 * max_target_deviation * ((double)rand()/(double)RAND_MAX - 0.5);
		for (int i = 0; i < get_num_sim_steps(); i++)
		{
			target_arr[i] = mean_target + target_deviation;
		}
	}
	else if (target_code==3)
	{
		int switch_location = (int) floor((0.20 * (double)rand()/(double)RAND_MAX + 0.40) * (double)(get_num_sim_steps() - 1));
		double target_deviation_1 = 2.0 * max_target_deviation * ((double)rand()/(double)RAND_MAX - 0.5);
		double target_deviation_2 = 2.0 * max_target_deviation * ((double)rand()/(double)RAND_MAX - 0.5);
		for (int i = 0; i < get_num_sim_steps(); i++)
		{
			if(i < switch_location)
			{
				target_arr[i] = mean_target + target_deviation_1;
			}
			else
			{
				target_arr[i] = mean_target + target_deviation_2;
			}
		}
	}
	
	
	// ************************************************** BOUNDARY CONDITIONS ************************************************** //
	// Randomize htc and ambient temperature
	htc = mean_htc + 2.0 * max_htc_deviation * ((double)rand()/(double)RAND_MAX - 0.5);
	amb_temp = mean_amb_temp + 2.0 * max_amb_temp_deviation * ((double)rand()/(double)RAND_MAX - 0.5);
	min_possible_temp = amb_temp < (initial_temp - max_initial_temp_deviation) ? amb_temp : (initial_temp - max_initial_temp_deviation);
	
	// Allocate memory for coarse BCs
	coarse_lr_bc_temps = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		coarse_lr_bc_temps[i] = new double*[num_coarse_vert_y];
		for(int j = 0; j < num_coarse_vert_y; j++)
		{
			coarse_lr_bc_temps[i][j] = new double[num_coarse_vert_z];
		}
	}
	coarse_fb_bc_temps = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		coarse_fb_bc_temps[i] = new double*[num_coarse_vert_x];
		for(int j = 0; j < num_coarse_vert_x; j++)
		{
			coarse_fb_bc_temps[i][j] = new double[num_coarse_vert_z];
		}
	}
	coarse_tb_bc_temps = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		coarse_tb_bc_temps[i] = new double*[num_coarse_vert_x];
		for(int j = 0; j < num_coarse_vert_x; j++)
		{
			coarse_tb_bc_temps[i][j] = new double[num_coarse_vert_y];
		}
	}
	
	// Allocate memory for fine BCs
	fine_lr_bc_temps = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		fine_lr_bc_temps[i] = new double*[num_fine_vert_y];
		for(int j = 0; j < num_fine_vert_y; j++)
		{
			fine_lr_bc_temps[i][j] = new double[num_fine_vert_z];
		}
	}
	fine_fb_bc_temps = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		fine_fb_bc_temps[i] = new double*[num_fine_vert_x];
		for(int j = 0; j < num_fine_vert_x; j++)
		{
			fine_fb_bc_temps[i][j] = new double[num_fine_vert_z];
		}
	}
	fine_tb_bc_temps = new double**[2];
	for(int i = 0; i < 2; i++)
	{
		fine_tb_bc_temps[i] = new double*[num_fine_vert_x];
		for(int j = 0; j < num_fine_vert_x; j++)
		{
			fine_tb_bc_temps[i][j] = new double[num_fine_vert_y];
		}
	}
	
	
	// ************************************************** FRONT PARAMETERS ************************************************** //
	// Set front detection parameters
	max_front_instances = 10 * num_fine_vert_y;
	front_filter_alpha = 1.0 - exp(-coarse_time_step/front_filter_time_const);
	front_mean_x_loc_history_len = front_mean_x_loc_history_time_len / coarse_time_step;

	// Init gloabal front curve
	global_front_curve = new double*[2];
	global_front_curve[0] = new double[max_front_instances];
	global_front_curve[1] = new double[max_front_instances];
	for(int i = 0; i < max_front_instances; i++)
	{
		global_front_curve[0][i] = -1.0;
		global_front_curve[1][i] = -1.0;
	}
	
	// Init threadwise front curve arr
	thread_front_curve = new double**[omp_get_max_threads()];
	for(int i = 0; i < omp_get_max_threads(); i++)
	{
		thread_front_curve[i] = new double*[2];
		thread_front_curve[i][0] = new double[max_front_instances];
		thread_front_curve[i][1] = new double[max_front_instances];
	}
	
	// Set all front tracking parameters to 0
	front_vel = 0.0;
	front_temp = initial_temp;
	front_shape_param = 1.0;
	front_mean_x_loc = 0.0;
	front_mean_x_loc_history = deque<double>(front_mean_x_loc_history_len, 0.0);
	front_mean_x_loc_history_avg = 0.0;

	// ************************************************** PRECALCULATION ************************************************** //
	// Allocate space for precalculated arrays
	precalc_exp_arr_len = (int) floor((precalc_end_temp - precalc_start_temp) / precalc_temp_step) + 1;
	precalc_pow_arr_len = (int) floor((precalc_end_cure - precalc_start_cure) / precalc_cure_step) + 1;
	precalc_exp_arr = new double[precalc_exp_arr_len];
	precalc_pow_arr = new double[precalc_pow_arr_len];

	// Populate exp precalculated array
	double curr_precalc_temp = precalc_start_temp;
	for(int i = 0; i < precalc_exp_arr_len; i++)
	{
		if (monomer_code==1)
		{
			precalc_exp_arr[i] = DCPD_GC1_pre_exponential * exp(-DCPD_GC1_activiation_energy / (gas_const * curr_precalc_temp));
		}
		else if (monomer_code==2)
		{
			precalc_exp_arr[i] = DCPD_GC2_pre_exponential * exp(-DCPD_GC2_activiation_energy / (gas_const * curr_precalc_temp));
		}
		else if (monomer_code==3)
		{
			precalc_exp_arr[i] = COD_pre_exponential * exp(-COD_activiation_energy / (gas_const * curr_precalc_temp));
		}
		curr_precalc_temp += precalc_temp_step;
	}
	
	// Populate pow precalculated array
	double curr_precalc_cure = precalc_start_cure;
	for(int i = 0; i < precalc_pow_arr_len; i++)
	{
		if (monomer_code==1)
		{
			precalc_pow_arr[i] = pow((1.0 - curr_precalc_cure), DCPD_GC1_model_fit_order) * (1.0 + DCPD_GC1_autocatalysis_const * curr_precalc_cure);
		}
		else if (monomer_code==2)
		{
			precalc_pow_arr[i] = pow((1.0 - curr_precalc_cure), DCPD_GC2_model_fit_order) * pow(curr_precalc_cure, DCPD_GC2_m_fit) * (1.0 / (1.0 + exp(DCPD_GC2_diffusion_const*(curr_precalc_cure - DCPD_GC2_critical_cure))));
		}
		else if (monomer_code==3)
		{
			precalc_pow_arr[i] = pow((1.0 - curr_precalc_cure), COD_model_fit_order) * pow(curr_precalc_cure, COD_m_fit);
		}
		curr_precalc_cure += precalc_cure_step;
	}


	// ************************************************** TRIGGER ************************************************** //
	// Disengage trigger if set to off
	if (!trigger_is_on)
	{
		trigger_flux = 0.0;
		trigger_time = 0.0;
		trigger_duration = 0.0;
	}
	
	
	// ************************************************** INPUT ************************************************** //
	// Allocate memory space for input mesh
	input_mesh = new double*[num_coarse_vert_x];
	for(int i = 0; i < num_coarse_vert_x; i++)
	{
		input_mesh[i] = new double[num_coarse_vert_y];
	}
	
	// Allocate space for input array
	input_location = new double[2];
	
	if(input_is_on)
	{
		// Input parameters
		input_const = -1.0 / (0.2171472409514 * radius_of_input * radius_of_input);
		peak_input_mag = input_total_power / (M_PI * 0.2171472409514 * radius_of_input * radius_of_input);
		min_input_x_loc = 0.0;
		max_input_x_loc = coarse_x_len;
		min_input_y_loc = 0.0;
		max_input_y_loc = coarse_y_len;
		
		// Randomly select laser power percentage
		input_percent = (double)rand()/(double)RAND_MAX;

		// Assign the random input locationj
		input_location[0] = coarse_x_len * ((double)rand()/(double)RAND_MAX);
		input_location[1] = coarse_y_len * ((double)rand()/(double)RAND_MAX);
	}
	else
	{
		// Input parameters
		input_const = 0.0;
		peak_input_mag = 0.0;
		min_input_x_loc = 0.0;
		max_input_x_loc = 0.0;
		min_input_y_loc = 0.0;
		max_input_y_loc = 0.0;
		
		// Assign 0 power
		input_percent = 0.0;

		// Assign 0 input location
		input_location[0] = 0.0;
		input_location[1] = 0.0;
	}

	// Generate input mesh values
	for (int i = 0; i < num_coarse_vert_x; i++)
	for (int j = 0; j < num_coarse_vert_y; j++)
	{
		double local_input_power = input_percent * peak_input_mag * exp(pow((coarse_x_mesh[i][j][0] - input_location[0]), 2.0)*input_const + pow((coarse_y_mesh[i][j][0] - input_location[1]), 2.0)*input_const);
		if (local_input_power < 0.01 * peak_input_mag)
		{
			input_mesh[i][j] = 0.0;
		}
		else
		{
			input_mesh[i][j] = local_input_power;
		}
	}
}

/**
* Destructor
*/
Finite_Difference_Solver::~Finite_Difference_Solver()
{
	// ************************************************** COARSE MESH ************************************************** //
	// Delete coarse mesh
	for(int i = 0; i != num_coarse_vert_x; ++i)
	{
		for(int j = 0; j != num_coarse_vert_y; ++j)
		{
			delete[] coarse_x_mesh[i][j];
			delete[] coarse_y_mesh[i][j];
			delete[] coarse_z_mesh[i][j];
			delete[] coarse_temp_mesh[i][j];
			delete[] coarse_laplacian_mesh[i][j];
			delete[] coarse_cure_mesh[i][j];
		}
		delete[] coarse_x_mesh[i];
		delete[] coarse_y_mesh[i];
		delete[] coarse_z_mesh[i];
		delete[] coarse_temp_mesh[i];
		delete[] coarse_laplacian_mesh[i];
		delete[] coarse_cure_mesh[i];
	}
	delete[] coarse_x_mesh;
	delete[] coarse_y_mesh;
	delete[] coarse_z_mesh;
	delete[] coarse_temp_mesh;
	delete[] coarse_laplacian_mesh;
	delete[] coarse_cure_mesh;
	
	
	// ************************************************** FINE MESH ************************************************** //
	// Delete fine mesh
	for(int i = 0; i != num_fine_vert_x; ++i)
	{
		for(int j = 0; j != num_fine_vert_y; ++j)
		{
			delete[] fine_cure_mesh[i][j];
			delete[] fine_temp_mesh[i][j];
			delete[] fine_laplacian_mesh[i][j];
		}
		delete[] fine_cure_mesh[i];
		delete[] fine_temp_mesh[i];
		delete[] fine_laplacian_mesh[i];
	}
	delete[] fine_cure_mesh;
	delete[] fine_temp_mesh;
	delete[] fine_laplacian_mesh;
	
	
	// ************************************************** TARGET ************************************************** //
	// Delete target array
	delete[] target_arr;
	
	
	// ************************************************** BOUNDARY CONDITIONS ************************************************** //
	// Delete coarse BCs
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_coarse_vert_y; ++j)
		{
			delete[] coarse_lr_bc_temps[i][j];
		}
		delete[] coarse_lr_bc_temps[i];
	}
	delete[] coarse_lr_bc_temps;
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_coarse_vert_x; ++j)
		{
			delete[] coarse_fb_bc_temps[i][j];
		}
		delete[] coarse_fb_bc_temps[i];
	}
	delete[] coarse_fb_bc_temps;
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_coarse_vert_x; ++j)
		{
			delete[] coarse_tb_bc_temps[i][j];
		}
		delete[] coarse_tb_bc_temps[i];
	}
	delete[] coarse_tb_bc_temps;
	
	// Delete fine BCs
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_fine_vert_y; ++j)
		{
			delete[] fine_lr_bc_temps[i][j];
		}
		delete[] fine_lr_bc_temps[i];
	}
	delete[] fine_lr_bc_temps;
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_fine_vert_x; ++j)
		{
			delete[] fine_fb_bc_temps[i][j];
		}
		delete[] fine_fb_bc_temps[i];
	}
	delete[] fine_fb_bc_temps;
	for(int i = 0; i != 2; ++i)
	{
		for(int j = 0; j != num_fine_vert_x; ++j)
		{
			delete[] fine_tb_bc_temps[i][j];
		}
		delete[] fine_tb_bc_temps[i];
	}
	delete[] fine_tb_bc_temps;
	
	
	// ************************************************** FRONT PARAMETERS ************************************************** //
	// Delete gloabal front curve
	for(int i = 0; i != 2; ++i)
	{
		delete[] global_front_curve[i];
	}
	delete[] global_front_curve;
	
	// Delete threadwise front curve array
	for(int i = 0; i != omp_get_max_threads(); ++i)
	{
		for(int j = 0; j != 2; ++j)
		{
			delete[] thread_front_curve[i][j];
		}
		delete[] thread_front_curve[i];
	}
	delete[] thread_front_curve;
	
	
	// ************************************************** PRECALCULATION ************************************************** //
	// Delete precalculated arrays
	delete[] precalc_exp_arr;
	delete[] precalc_pow_arr;
	
	
	// ************************************************** INPUT ************************************************** //
	// Delete input location
	delete[] input_location;
	
	// Delete input mesh
	for(int i = 0; i != num_coarse_vert_x; ++i)
	{
		delete[] input_mesh[i];
	}
	delete[] input_mesh;
}

// ================================================================================================= GETTERS ================================================================================================= //
// ************************************************** MESH GETTERS ************************************************** //
/**
* Gets the number of vertices in lengthwise direction
* @return The number of vertices in the lengthwise direction
*/
int Finite_Difference_Solver::get_num_coarse_vert_x()
{
	return num_coarse_vert_x;
}

/**
* Gets the number of vertices in widthwise direction
* @return The number of vertices in the widthwise direction
*/
int Finite_Difference_Solver::get_num_coarse_vert_y()
{
	return num_coarse_vert_y;
}

/**
* Gets the number of vertices in depthwise direction
* @return The number of vertices in the depthwise direction
*/
int Finite_Difference_Solver::get_num_coarse_vert_z()
{
	return num_coarse_vert_z;
}

/**
* Gets the top layer of the x mesh
* @return The top layer of the x mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Difference_Solver::get_coarse_x_mesh_z0()
{
	vector<vector<double>> ret_val = vector<vector<double>>(num_coarse_vert_x, vector<double>(num_coarse_vert_y, 0.0));
	for (int i = 0; i < num_coarse_vert_x; i++)
	{
		for (int j = 0; j < num_coarse_vert_y; j++)
		{
			ret_val[i][j] = coarse_x_mesh[i][j][0];
		}
	}
	return ret_val;
}

/**
* Gets the top layer of the y mesh
* @return The top layer of the y mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Difference_Solver::get_coarse_y_mesh_z0()
{
	vector<vector<double>> ret_val = vector<vector<double>>(num_coarse_vert_x, vector<double>(num_coarse_vert_y, 0.0));
	for (int i = 0; i < num_coarse_vert_x; i++)
	{
		for (int j = 0; j < num_coarse_vert_y; j++)
		{
			ret_val[i][j] = coarse_y_mesh[i][j][0];
		}
	}
	return ret_val;
}

/**
* Gets the starting and ending x location of the fine mesh
* @return vector containin starting and ending x location of the fine mesh
*/
vector<double> Finite_Difference_Solver::get_fine_mesh_loc()
{
	vector<double> ret_val = vector<double>(2, 0.0);
	ret_val[0] = fine_mesh_start_loc;
	ret_val[1] = fine_mesh_end_loc;
	
	return ret_val; 
}


// ************************************************** TIME GETTERS ************************************************** //
/**
* Gets the duration of the simulation
* @return The duration of the simulation in seconds
*/
double Finite_Difference_Solver::get_sim_duration()
{
	return sim_duration;
}

/**
* Gets the number of discrete time steps per episode
* @return The coarse_x_len of the target array
*/
int Finite_Difference_Solver::get_num_sim_steps()
{
	return (int)round(sim_duration / coarse_time_step);
}

/**
* Gets the time step used in the simulation
* @return The simulation time step in seconds
*/
double Finite_Difference_Solver::get_coarse_time_step()
{
	return coarse_time_step;
}

/**
* Gets the current time
* @return The time in seconds
*/
double Finite_Difference_Solver::get_curr_sim_time()
{
	return curr_sim_time;
}

// ************************************************** INPUT PARAMETERS GETTERS ************************************************** //
/**
* Gets the maximum magnitude of the input in W/m^2
* @return The peak magnitude of the input in W/m^2
*/
double Finite_Difference_Solver::get_peak_input_mag()
{
	return peak_input_mag;
}

/**
* Gets the exponent constant used to calculate input mesh
* @return The exponent constant used to calculate input mesh in W/m^2
*/
double Finite_Difference_Solver::get_input_const()
{
	return input_const;
}

/**
* Gets the maximum possible input magnitude percent rate
* @return maximum possible input magnitude percent rate in decimal percent per second
*/
double Finite_Difference_Solver::get_max_input_mag_percent_rate()
{
	return max_input_mag_percent_rate;
}

/**
* Gets the maximum possible input single axis movement rate
* @return the maximum possible input single axis movement rate in m/s
*/
double Finite_Difference_Solver::get_max_input_slew_speed()
{
	return max_input_slew_speed;
}


// ************************************************** INPUT STATE GETTERS ************************************************** //
/**
* Gets the current input power percent
* @return The power level of the input in percent
*/
double Finite_Difference_Solver::get_input_percent()
{
	return input_percent;
}

/**
* Gets the input location
* @return The input location as a vector {x,y}
*/
vector<double> Finite_Difference_Solver::get_input_location()
{
	vector<double> ret_val = vector<double>(2, 0.0);
	ret_val[0] = input_location[0];
	ret_val[1] = input_location[1];
	
	return ret_val;
}


// ************************************************** TARGET GETTERS ************************************************** //
/**
* Gets the current target
* @return The current target front
*/
double Finite_Difference_Solver::get_curr_target()
{
	return target_arr[curr_sim_step];
}


// ************************************************** SIM OPTION GETTERS ************************************************** //
/**
* Gets the control mode
* @return True upon control speed, false upon control temperature
*/
bool Finite_Difference_Solver::get_control_mode()
{
	return control_code==1;
}


// ************************************************** TEMP + CURE GETTERS ************************************************** //
/**
* Gets the top layer of the coarse temperature mesh
* @return The top layer of the temperature mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Difference_Solver::get_coarse_temp_z0()
{
	vector<vector<double>> ret_val = vector<vector<double>>(num_coarse_vert_x, vector<double>(num_coarse_vert_y, 0.0));
	for (int i = 0; i < num_coarse_vert_x; i++)
	{
		for (int j = 0; j < num_coarse_vert_y; j++)
		{
			ret_val[i][j] = coarse_temp_mesh[i][j][0];
		}
	}
	return ret_val;
}

/**
* Gets the top layer of the fine temperature mesh
* @return The top layer of the temperature mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Difference_Solver::get_fine_temp_z0()
{
	vector<vector<double>> ret_val = vector<vector<double>>(num_fine_vert_x, vector<double>(num_fine_vert_y, 0.0));
	for (int i = 0; i < num_fine_vert_x; i++)
	{
		for (int j = 0; j < num_fine_vert_y; j++)
		{
			ret_val[i][j] = fine_temp_mesh[get_ind(i)][j][0];
		}
	}
	return ret_val;
}

/**
* Gets the normalized top layer of the temperature mesh around the front
* @return The top layer of the temperature mesh around the front as a 2D vector in x,y normalized against in 0.90*T0 to 1.10*Tmax
*/
vector<vector<double>> Finite_Difference_Solver::get_norm_coarse_temp_z0()
{
	vector<vector<double>> ret_val = vector<vector<double>>(num_coarse_vert_x, vector<double>(num_coarse_vert_y, 0.0));
	for (int i = 0; i < num_coarse_vert_x; i++)
	{
		for (int j = 0; j < num_coarse_vert_y; j++)
		{
			ret_val[i][j] = (coarse_temp_mesh[i][j][0] - 0.90*initial_temp) / (1.1*monomer_burn_temp - 0.90*initial_temp);
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
vector<vector<double>> Finite_Difference_Solver::get_coarse_cure_z0()
{
	vector<vector<double>> ret_val = vector<vector<double>>(num_coarse_vert_x, vector<double>(num_coarse_vert_y, 0.0));
	for (int i = 0; i < num_coarse_vert_x; i++)
	{
		for (int j = 0; j < num_coarse_vert_y; j++)
		{
			ret_val[i][j] = coarse_cure_mesh[i][j][0];
		}
	}
	return ret_val;
}

/**
* Gets the top layer of the fine cure mesh
* @return The top layer of the cure mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Difference_Solver::get_fine_cure_z0()
{
	vector<vector<double>> ret_val = vector<vector<double>>(num_fine_vert_x, vector<double>(num_fine_vert_y, 0.0));
	for (int i = 0; i < num_fine_vert_x; i++)
	{
		for (int j = 0; j < num_fine_vert_y; j++)
		{
			ret_val[i][j] = fine_cure_mesh[get_ind(i)][j][0];
		}
	}
	return ret_val;
}


// ************************************************** FRONT STATE GETTERS ************************************************** //
/**
* Gets the current front curve
* @return Front curve of standard coarse_x_len. If front does not populate entire vector, end padded with -1.0
*/
vector<vector<double>> Finite_Difference_Solver::get_front_curve()
{
	vector<vector<double>> ret_val = vector<vector<double>>(2, vector<double>(max_front_instances, 0.0));
	bool done = false;
	for(int i = 0; i < max_front_instances; i++)
	{
		if(!done)
		{
			ret_val[0][i] = global_front_curve[0][i];
			ret_val[1][i] = global_front_curve[1][i];
			done = (global_front_curve[0][i] < 0.0) || (global_front_curve[1][i] < 0.0);
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
double Finite_Difference_Solver::get_front_vel()
{
	return front_vel;
}

/**
* Gets the current front temperature
* @return The current front mean temperature
*/
double Finite_Difference_Solver::get_front_temp()
{
	return front_temp;
}

/**
* Gets the current front shape parameters
* @return The current front shape parameter (x stdev normalized against quarter fine mesh coarse_x_len)
*/
double Finite_Difference_Solver::get_front_shape_param()
{
	return front_shape_param;
}


// ================================================================================================= PUBLIC FUNCTIONS ================================================================================================= //
/**
* Prints the finite element solver parameters to std out
*/
void Finite_Difference_Solver::print_params()
{
	// Simulation device
	cout << "\nSimulation(\n";
	cout << "  (Device): CPU\n";
	cout << "  (Num Threads): " << omp_get_max_threads() << "\n";
	cout << ")\n";
	
	// Input parameters
	cout << "\nInput(\n";
	if (!input_is_on)
	{
		cout << "  No input.\n";
	}
	else
	{ 
		cout << "  (Radius): " << 1000.0 * radius_of_input << " mm\n";
		cout << "  (Power): " << 1000.0 * input_total_power << " mW\n";
		cout << "  (Power Rate): " << 1000.0 * input_total_power * max_input_mag_percent_rate << " mW/s\n";
		cout << "  (Slew Rate): " << 1000.0 * max_input_slew_speed << " mm/s\n";
	}
	cout << ")\n";
	
	cout << "\nTarget(\n";
	if (control_code==1)
	{
		cout << "  (Type): Front speed\n";
		if (target_code==1)
		{
			cout << "  (Style): Constant target\n";
			cout << "  (Target): " << 1000.0*mean_target_speed << " mm/s\n";
		}
		else if (target_code==2)
		{
			cout << "  (Style): Random target\n";
			cout << "  (Target): " << 1000.0*mean_target_speed << " mm/s +- " << 1000.0*max_target_speed_deviation << "mm/s\n";
		}
		else if (target_code==3)
		{
			cout << "  (Style): Switch target\n";
			cout << "  (Target): " << 1000.0*mean_target_speed << " mm/s +- " << 1000.0*max_target_speed_deviation << "mm/s\n";
		}
	}
	else if (control_code==2)
	{
		cout << "  (Type): Front temperature\n";
		if (target_code==1)
		{
			cout << "  (Style): Constant target\n";
			cout << "  (Target): " << mean_target_temp-273.15 << " C\n";
		}
		else if (target_code==2)
		{
			cout << "  (Style): Random target\n";
			cout << "  (Target): " << mean_target_temp-273.15 << " C +- " << max_target_temp_deviation << "C\n";
		}
		else if (target_code==3)
		{
			cout << "  (Style): Switch target\n";
			cout << "  (Target): " << mean_target_temp-273.15 << " C +- " << max_target_temp_deviation << "C\n";
		}
	}

	
	// Trigger parameters
	cout << "\nTrigger(\n";
	if (!trigger_is_on)
	{
		cout << "  No trigger.\n";
	}
	else if (trigger_is_on)
	{ 
		cout << "  (Flux): " << trigger_flux << " W/m^2\n";
		cout << "  (Time): " << trigger_time << " s\n";
		cout << "  (Duration): " << trigger_duration  << " s\n";
	}
	cout << ")\n";
	
	// Monomer
	cout << "\nMaterial(\n";
	if (monomer_code==1)
	{
		cout << "  (Monomer): DCPD\n";
		cout << "  (Catalyst): GC1\n";
	}
	else if (monomer_code==2)
	{ 
		cout << "  (Monomer): DCPD\n";
		cout << "  (Catalyst): GC2\n";
	}
	else if (monomer_code==3)
	{
		cout << "  (Monomer): COD\n";
		cout << "  (Catalyst): GC2\n";
	}
	cout << "  (Initial Temperature): " << initial_temp-273.15 << " C +- " << max_initial_temp_deviation << " C\n";
	cout << "  (Initial Cure): " << initial_cure << " +- " << max_initial_cure_deviation << "\n";
	cout << ")\n";
	
	// Coarse Mesh
	cout << "\nCoarse Mesh(\n";
	cout << "  (Dimensions): " << 1000.0*coarse_x_len << " x " << 1000.0*coarse_y_len << " x " << 1000.0*coarse_z_len << " mm\n";
	cout << "  (Grid Size): " << 1e6*coarse_x_len/(double)num_coarse_vert_x << " x " << 1e6*coarse_y_len/(double)num_coarse_vert_y << " x " << 1e6*coarse_z_len/(double)num_coarse_vert_z << " um\n";
	cout << "  (Grid Vertices): " << num_coarse_vert_x << " x " << num_coarse_vert_y << " x " << num_coarse_vert_z << "\n";
	cout << "  (Time Step): " << 1000.0*coarse_time_step << " ms\n";
	cout << ")\n\n";
	
	// Fine Mesh
	cout << "\nFine Mesh(\n";
	cout << "  (Fine Dimensions): " << 1000.0*fine_x_len << " x " << 1000.0*fine_y_len << " x " << 1000.0*fine_z_len << " mm\n";
	cout << "  (Fine Grid Size): " << 1e6*fine_x_len/(double)num_fine_vert_x << " x " << 1e6*fine_y_len/(double)num_fine_vert_y << " x " << 1e6*fine_z_len/(double)num_fine_vert_z << " um\n";
	cout << "  (Fine Grid Vertices): " << num_fine_vert_x << " x " << num_fine_vert_y << " x " << num_fine_vert_z << "\n";
	cout << "  (Fine Time Step): " << 1000.0*fine_time_step << " ms\n";
	cout << ")\n\n";
	
	// Environment
	cout << "\nEnvironment(\n";
	cout << "  (Duration): " << sim_duration << " s\n";
	cout << "  (Ambient Temperature): " << mean_amb_temp-273.15 << " C +- " << max_amb_temp_deviation << " C\n";
	cout << "  (HTC): " << mean_htc << " W/m^2-K +- " << max_htc_deviation << " W/m^2-K\n";
	cout << ")\n\n";
}

/**
* Resets the environment to initial conditions
*/
void Finite_Difference_Solver::reset()
{	
	// ************************************************** COARSE MESH ************************************************** //
	// Reset coarse mesh
	for(int i = 0; i < num_coarse_vert_x; i++)
	for(int j = 0; j < num_coarse_vert_y; j++)
	for(int k = 0; k < num_coarse_vert_z; k++)
	{
		coarse_temp_mesh[i][j][k] = initial_temp;
		coarse_laplacian_mesh[i][j][k] = 0.0;
		coarse_cure_mesh[i][j][k] = initial_cure;
	}
	
	// Perturb temperature and cure meshes
	perturb_mesh(coarse_temp_mesh, max_initial_temp_deviation);
	perturb_mesh(coarse_cure_mesh, max_initial_cure_deviation);
	
	
	// ************************************************** FINE MESH ************************************************** //	
	// Copy coarse mesh initial values to the fine mesh
	copy_coarse_to_fine();
	
	
	// ************************************************** TIME ************************************************** //
	// Simulation time and target velocity index
	curr_sim_time = 0.0;
	curr_sim_step = 0;
	
	
	// ************************************************** TARGET ************************************************** //
	// Reset target array
	if(target_code==1)
	{
		for(int i = 0; i < get_num_sim_steps(); i++)
		{
			target_arr[i] = mean_target;
		}
	}
	else if (target_code==2)
	{
		double target_deviation = 2.0 * max_target_deviation * ((double)rand()/(double)RAND_MAX - 0.5);
		for (int i = 0; i < get_num_sim_steps(); i++)
		{
			target_arr[i] = mean_target + target_deviation;
		}
	}
	else if (target_code==3)
	{
		int switch_location = (int) floor((0.20 * (double)rand()/(double)RAND_MAX + 0.40) * (double)(get_num_sim_steps() - 1));
		double target_deviation_1 = 2.0 * max_target_deviation * ((double)rand()/(double)RAND_MAX - 0.5);
		double target_deviation_2 = 2.0 * max_target_deviation * ((double)rand()/(double)RAND_MAX - 0.5);
		for (int i = 0; i < get_num_sim_steps(); i++)
		{
			if(i < switch_location)
			{
				target_arr[i] = mean_target + target_deviation_1;
			}
			else
			{
				target_arr[i] = mean_target + target_deviation_2;
			}
		}
	}
	
	
	// ************************************************** BOUNDARY CONDITIONS ************************************************** //
	// Randomize htc and ambient temperature
	htc = mean_htc + 2.0 * max_htc_deviation * ((double)rand()/(double)RAND_MAX - 0.5);
	amb_temp = mean_amb_temp + 2.0 * max_amb_temp_deviation * ((double)rand()/(double)RAND_MAX - 0.5);
	min_possible_temp = amb_temp < (initial_temp - max_initial_temp_deviation) ? amb_temp : (initial_temp - max_initial_temp_deviation);
	
	
	// ************************************************** FRONT PARAMETERS ************************************************** //
	// Reset gloabal front curve
	for(int i = 0; i < max_front_instances; i++)
	{
		global_front_curve[0][i] = -1.0;
		global_front_curve[1][i] = -1.0;
	}
	
	// Reset front tracking parameters to 0
	front_vel = 0.0;
	front_temp = initial_temp;
	front_shape_param = 1.0;
	front_mean_x_loc = 0.0;
	front_mean_x_loc_history = deque<double>(front_mean_x_loc_history_len, 0.0);
	front_mean_x_loc_history_avg = 0.0;
	
	
	// ************************************************** INPUT ************************************************** //	
	// Reset input location and percent
	if(input_is_on)
	{	
		// Randomly select laser power percentage
		input_percent = (double)rand()/(double)RAND_MAX;

		// Assign the random input locationj
		input_location[0] = coarse_x_len * ((double)rand()/(double)RAND_MAX);
		input_location[1] = coarse_y_len * ((double)rand()/(double)RAND_MAX);
	}
	else
	{	
		// Assign 0 power
		input_percent = 0.0;

		// Assign 0 input location
		input_location[0] = 0.0;
		input_location[1] = 0.0;
	}

	// Reset input mesh values
	for (int i = 0; i < num_coarse_vert_x; i++)
	for (int j = 0; j < num_coarse_vert_y; j++)
	{
		double local_input_power = input_percent * peak_input_mag * exp(pow((coarse_x_mesh[i][j][0] - input_location[0]), 2.0)*input_const + pow((coarse_y_mesh[i][j][0] - input_location[1]), 2.0)*input_const);
		if (local_input_power < 0.01 * peak_input_mag)
		{
			input_mesh[i][j] = 0.0;
		}
		else
		{
			input_mesh[i][j] = local_input_power;
		}
	}
}

/**
* Steps the environment forward one time step
* @param normalized X slew speed command (-1.0, 1.0)
* @param normalized Y slew speed command (-1.0, 1.0)
* @param normalized magnitude percent rate command (-1.0, 1.0)
* @return Whether the sim is done or not
*/
bool Finite_Difference_Solver::step(double x_slew_speed_cmd, double y_slew_speed_cmd, double mag_percent_rate_cmd)
{
	// Step the input, cure, front, and temperature
	step_input(x_slew_speed_cmd, y_slew_speed_cmd, mag_percent_rate_cmd);
	step_meshes();

	// Step time
	bool done = step_time();
	return done;
}

/**
* Solves for the current state's reward
* @return The calculated reward
*/
double Finite_Difference_Solver::get_reward()
{
	// Initialize reward variables
	double input_reward = 0.0;
	double max_temp_reward = 0.0;
	double front_shape_reward = 0.0;
	double target_reward = 0.0;

	// Determine maximum mesh temperature
	double max_temp = 0.0;
	
	// Get the input reward
	input_reward = input_reward_const * (1.0 - input_percent);

	// Get the overage reward
	max_temp_reward = max_temp > monomer_burn_temp ? 0.0 : max_temp_reward_const;

	// Get the front shape reward
	front_shape_reward = front_shape_reward_const * (1.0 - front_shape_param);

	// Get the total reward
	if (control_code==1)
	{
		target_reward = target_reward_const * exp(-0.5 * pow(((front_temp-target_arr[curr_sim_step])/(0.30*target_arr[curr_sim_step])), 2.0));
	}
	else if (control_code==2)
	{
		target_reward = target_reward_const * exp(-0.5 * pow(((front_vel-target_arr[curr_sim_step])/(0.03*target_arr[curr_sim_step])), 2.0));
	}

	return input_reward+max_temp_reward+front_shape_reward+target_reward;
}

// ================================================================================================= PRIVATE FUNCTIONS ================================================================================================= //

/**
* Loads FDS parameters from .cfg file
* @return 0 on success, 1 on failure
*/
int Finite_Difference_Solver::load_config()
{
	// Load from config file
	ifstream config_file;
	config_file.open("config_files/fds.cfg");
	string config_dump;
	string bool_dump;
	string string_dump;
	if (config_file.is_open())
	{
		// ************************************************** SIM OPTIONS ************************************************** //	
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> bool_dump;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		if (bool_dump.compare("true")==0)
		{
			input_is_on = true;
		}
		else if (bool_dump.compare("false")==0)
		{
			input_is_on = false;
		}
		else
		{
			cout << "\nInput configuration not recognized.";
		}
		config_file >> config_dump >> bool_dump;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		if (bool_dump.compare("true")==0)
		{
			trigger_is_on = true;
		}
		else if (bool_dump.compare("false")==0)
		{
			trigger_is_on = false;
		}
		else
		{
			cout << "\nTrigger configuration not recognized.";
		}
		config_file >> config_dump >> string_dump;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		if (string_dump.compare("dcpd_gc1")==0)
		{
			monomer_code = 1;
		}
		else if (string_dump.compare("dcpd_gc2")==0)
		{
			monomer_code = 2;
		}
		else if (string_dump.compare("cod")==0)
		{
			monomer_code = 3;
		}
		else
		{
			cout << "\nMaterial configuration not recognized.";
		}
		config_file >> config_dump >> string_dump;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		if (string_dump.compare("speed")==0)
		{
			control_code = 1;
		}
		else if (string_dump.compare("temp")==0)
		{
			control_code = 2;
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
			target_code = 1;
		}
		else if (string_dump.compare("rand")==0)
		{
			target_code = 2;
		}
		else if (string_dump.compare("switch")==0)
		{
			target_code = 3;
		}
		else
		{
			cout << "\nTarget configuration not recognized.";
		}
		
		
		// ************************************************** COARSE MESH PARAMS ************************************************** //	
		config_file >> config_dump >> num_coarse_vert_x;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> num_coarse_vert_y;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> num_coarse_vert_z;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> coarse_x_len;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> coarse_y_len;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> coarse_z_len;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** FINE MESH PARAMS ************************************************** //
		config_file >> config_dump >> fine_x_len;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> fine_x_steps_per_coarse_x_step;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> fine_y_steps_per_coarse_y_step;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> fine_z_steps_per_coarse_z_step;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** TEMPORAL PARAMS ************************************************** //
		config_file >> config_dump >> sim_duration;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> coarse_time_step;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> fine_time_steps_per_coarse_time_step;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** PROBLEM DEFINITION ************************************************** //
		config_file >> config_dump >> monomer_burn_temp;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> mean_target_speed;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> max_target_speed_deviation;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> mean_target_temp;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> max_target_temp_deviation;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** INITIAL CONDITIONS ************************************************** //
		config_file >> config_dump >> initial_temp;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> max_initial_temp_deviation;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> initial_cure;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> max_initial_cure_deviation;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** BOUNDARY CONDITIONS ************************************************** //
		config_file >> config_dump >> mean_htc;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> max_htc_deviation;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> mean_amb_temp;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> max_amb_temp_deviation;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** FRONT DETECTION PARAMS ************************************************** //
		config_file >> config_dump >> front_filter_time_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> front_mean_x_loc_history_time_len;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> front_min_cure;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> front_max_cure;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> front_min_cure_rate;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** CRITICAL CURE VALUES ************************************************** //
		config_file >> config_dump >> critical_cure_rate;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> transition_cure_rate;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** PRECALCULATION PARAMS ************************************************** //
		config_file >> config_dump >> precalc_start_temp;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> precalc_end_temp;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> precalc_temp_step;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> precalc_start_cure;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> precalc_end_cure;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> precalc_cure_step;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** TRIGGER PARAMS ************************************************** //
		config_file >> config_dump >> trigger_flux;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> trigger_time;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> trigger_duration;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** INPUT PARAMS ************************************************** //
		config_file >> config_dump >> radius_of_input;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> input_total_power;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> max_input_mag_percent_rate;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> max_input_slew_speed;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		
		// ************************************************** REWARD PARAMS ************************************************** //
		config_file >> config_dump >> input_reward_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> max_temp_reward_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> front_shape_reward_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> target_reward_const;
	}
	else
	{
		cout << "Unable to open config_files/fds.cfg." << endl;
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
void Finite_Difference_Solver::perturb_mesh(double*** arr, double max_deviation)
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
	
	for (int i = 0; i < num_coarse_vert_x; i++)
	for (int j = 0; j < num_coarse_vert_y; j++)
	for (int k = 0; k < num_coarse_vert_z; k++)
	{
		x = -2.0*min_mag + min_x_bias + (2.0*max_mag + max_x_bias + 2.0*min_mag - min_x_bias) * ((double)i / (num_coarse_vert_x-1));
		y = -2.0*min_mag + min_y_bias + (2.0*max_mag + max_y_bias + 2.0*min_mag - min_y_bias) * ((double)j / (num_coarse_vert_y-1));
		z = -2.0*min_mag + min_z_bias + (2.0*max_mag + max_z_bias + 2.0*min_mag - min_z_bias) * ((double)k / (num_coarse_vert_z-1));
		xyz = x * y * z;
		
		perturbation = mag_1 * sin(xyz + bias_1) + mag_2 * sin(2.0*xyz + bias_2) + mag_3 * sin(3.0*xyz + bias_3);
		arr[i][j][k] = arr[i][j][k] + (max_deviation * perturbation) / scale;
	}
}

/** Step the input through time
* @param normalized X slew speed command (-1.0, 1.0)
* @param normalized Y slew speed command (-1.0, 1.0)
* @param normalized magnitude percent rate command (-1.0, 1.0)
*/
void Finite_Difference_Solver::step_input(double x_slew_speed_cmd, double y_slew_speed_cmd, double mag_percent_rate_cmd)
{
	// Convert the raw PPO x command to usable, clipped x location rate command
	double cmd_x = x_slew_speed_cmd * max_input_slew_speed;
	cmd_x = cmd_x > max_input_slew_speed ? max_input_slew_speed : cmd_x;
	cmd_x = cmd_x < -max_input_slew_speed ? -max_input_slew_speed : cmd_x;

	// Update the input's x location from the converted location rate commands
	input_location[0] = input_location[0] + cmd_x * coarse_time_step;
	input_location[0] = input_location[0] > max_input_x_loc ? max_input_x_loc : input_location[0];
	input_location[0] = input_location[0] < min_input_x_loc ? min_input_x_loc : input_location[0];
	
	// Determine the approximate mesh x index of the input location
	int input_x_index = (int)round(input_location[0] / coarse_x_step) + 1;

	// Convert the raw PPO y command to usable, clipped y location rate command
	double cmd_y = y_slew_speed_cmd * max_input_slew_speed;
	cmd_y = cmd_y > max_input_slew_speed ? max_input_slew_speed : cmd_y;
	cmd_y = cmd_y < -max_input_slew_speed ? -max_input_slew_speed : cmd_y;

	// Update the input's y location from the converted location rate commands
	input_location[1] = input_location[1] + cmd_y * coarse_time_step;
	input_location[1] = input_location[1] > max_input_y_loc ? max_input_y_loc : input_location[1];
	input_location[1] = input_location[1] < min_input_y_loc ? min_input_y_loc : input_location[1];

	// Determine the approximate mesh y index of the input location
	int input_y_index = (int)round(input_location[1] / coarse_y_step) + 1;

	// Convert the raw PPO magnitude percent rate command to usable, clipped magnitude percent rate command
	double cmd_mag = mag_percent_rate_cmd * max_input_mag_percent_rate;
	cmd_mag = cmd_mag > max_input_mag_percent_rate ? max_input_mag_percent_rate : cmd_mag;
	cmd_mag = cmd_mag < -max_input_mag_percent_rate ? -max_input_mag_percent_rate : cmd_mag;

	// Update the input's magnitude from the converted magnitude rate commands
	input_percent = input_percent + cmd_mag * coarse_time_step;
	input_percent = input_percent > 1.0 ? 1.0 : input_percent;
	input_percent = input_percent < 0.0 ? 0.0 : input_percent;
	
	// Detemine the range of x and y indices within which the input resides
	int start_x = input_x_index - (int)round((double)radius_of_input/coarse_x_step) - 1;
	start_x = start_x < 0 ? 0 : start_x;
	int end_x = input_x_index + (int)round((double)radius_of_input/coarse_x_step) + 1;
	end_x = end_x > num_coarse_vert_x ? num_coarse_vert_x : end_x;
	int start_y = input_y_index - (int)round((double)radius_of_input/coarse_y_step) - 1;
	start_y = start_y < 0 ? 0 : start_y;
	int end_y = input_y_index + (int)round((double)radius_of_input/coarse_y_step) + 1;
	end_y = end_y > num_coarse_vert_y ? num_coarse_vert_y : end_y;
	
	// Update the input wattage mesh
	for (int i = start_x; i < end_x; i++)
	for (int j = start_y; j < end_y; j++)
	{
		double local_input_power = input_percent * peak_input_mag * exp(pow((coarse_x_mesh[i][j][0] - input_location[0]), 2.0) * input_const + pow((coarse_y_mesh[i][j][0] - input_location[1]), 2.0) * input_const);
		if (local_input_power < 0.01 * peak_input_mag)
		{
			input_mesh[i][j] = 0.0;
		}
		else
		{
			input_mesh[i][j] = local_input_power;
		}
	}
}

/** Compute the location in the fine temperature mesh at which the ith temperature value is stored
* @param x location associated index
* @return Index of fine mesh at which the x location index value is stored
*/
int Finite_Difference_Solver::get_ind(int i)
{
	int i_access = i + fine_mesh_zero_index;
	if( i_access >= num_fine_vert_x )
	{
		i_access = i_access - num_fine_vert_x;
	}
	
	return i_access;
}

/** Copies the coarse mesh data to the fine mesh with the fine mesh starting at the right most point.
*/
void Finite_Difference_Solver::copy_coarse_to_fine()
{
	// Assign the starting index of the fine and coarse meshes
	fine_mesh_zero_index = 0;
	coarse_x_index_at_fine_mesh_start = 0;
	fine_mesh_start_loc = 0.0;
	fine_mesh_end_loc = coarse_x_mesh[((int)floor((double)(num_fine_vert_x-1) / (double)fine_x_steps_per_coarse_x_step))+1][0][0];

	// Assign coarse mesh values to their respective fine mesh counterparts
	for(int i = 0; i < num_fine_vert_x; i++)
	for(int j = 0; j < num_fine_vert_y; j++)
	for(int k = 0; k < num_fine_vert_z; k++)
	{	
		// Determine location in coarse mesh
		int curr_coarse_x_index = (int)floor((double)i / (double)fine_x_steps_per_coarse_x_step);
		int curr_coarse_y_index = (int)floor((double)j / (double)fine_y_steps_per_coarse_y_step);
		int curr_coarse_z_index = (int)floor((double)k / (double)fine_z_steps_per_coarse_z_step);
		
		// Assign coarse values to fine mesh
		fine_temp_mesh[i][j][k] = coarse_temp_mesh[curr_coarse_x_index][curr_coarse_y_index][curr_coarse_z_index];
		fine_cure_mesh[i][j][k] = coarse_cure_mesh[curr_coarse_x_index][curr_coarse_y_index][curr_coarse_z_index];
	}
	
}

/** Slides the fine mesh right by one corase mesh element
*/
void Finite_Difference_Solver::slide_fine_mesh_right()
{
	// Ensure fine mesh is not slid off of simulation domain
	int coarse_mesh_x_slice_being_added = coarse_x_index_at_fine_mesh_start + coarse_x_steps_per_fine_x_len;
	if(coarse_mesh_x_slice_being_added >= num_coarse_vert_x)
	{
		return;
	}
	fine_mesh_start_loc = coarse_x_mesh[coarse_x_index_at_fine_mesh_start][0][0];
	if(coarse_mesh_x_slice_being_added+1 < num_coarse_vert_x)
	{
		fine_mesh_end_loc = coarse_x_mesh[coarse_mesh_x_slice_being_added+1][0][0];
	}
	else
	{
		fine_mesh_end_loc = coarse_x_len;
	}
	coarse_x_index_at_fine_mesh_start++;
			
	// Copy coarse values to fine mesh
	for(int i = 0; i < fine_x_steps_per_coarse_x_step; i++)
	for(int j = 0; j < num_fine_vert_y; j++)
	for(int k = 0; k < num_fine_vert_z; k++)
	{
		// Determine location in coarse mesh
		int curr_coarse_y_index = (int)floor((double)j / (double)fine_y_steps_per_coarse_y_step);
		int curr_coarse_z_index = (int)floor((double)k / (double)fine_z_steps_per_coarse_z_step);
		
		// Assign coarse values to fine mesh
		fine_temp_mesh[get_ind(i)][j][k] = coarse_temp_mesh[coarse_mesh_x_slice_being_added][curr_coarse_y_index][curr_coarse_z_index];
		fine_cure_mesh[get_ind(i)][j][k] = coarse_cure_mesh[coarse_mesh_x_slice_being_added][curr_coarse_y_index][curr_coarse_z_index];
	}

	// Update the fine mesh starting index
	fine_mesh_zero_index += fine_x_steps_per_coarse_x_step;
	if( fine_mesh_zero_index >= num_fine_vert_x )
	{
		fine_mesh_zero_index = fine_mesh_zero_index - num_fine_vert_x;
	}
}

/** Copies the fine mesh data to the coarse mesh
*/
void Finite_Difference_Solver::copy_fine_to_coarse()
{	
	#pragma omp parallel for collapse(3)
	for(int i = 0; i < coarse_x_steps_per_fine_x_len; i++)
	for(int j = 0; j < coarse_y_steps_per_fine_y_len; j++)
	for(int k = 0; k < coarse_z_steps_per_fine_z_len; k++)
	{
		double temp_avg = 0.0;
		double cure_avg = 0.0;
		
		int start_fine_x_index = i * fine_x_steps_per_coarse_x_step;
		int start_fine_y_index = j * fine_y_steps_per_coarse_y_step;
		int start_fine_z_index = k * fine_z_steps_per_coarse_z_step;
		
		for(int p = start_fine_x_index; p < start_fine_x_index + fine_x_steps_per_coarse_x_step; p++)
		for(int q = start_fine_y_index; q < start_fine_y_index + fine_y_steps_per_coarse_y_step; q++)
		for(int r = start_fine_z_index; r < start_fine_z_index + fine_z_steps_per_coarse_z_step; r++)
		{	
			temp_avg += fine_temp_mesh[get_ind(p)][q][r];
			cure_avg += fine_cure_mesh[get_ind(p)][q][r];
		}
		
		coarse_temp_mesh[coarse_x_index_at_fine_mesh_start+i][j][k] = temp_avg / ((double)(fine_x_steps_per_coarse_x_step*fine_y_steps_per_coarse_y_step*fine_z_steps_per_coarse_z_step));
		coarse_cure_mesh[coarse_x_index_at_fine_mesh_start+i][j][k] = cure_avg / ((double)(fine_x_steps_per_coarse_x_step*fine_y_steps_per_coarse_y_step*fine_z_steps_per_coarse_z_step));
	}
}

/** Updates the virtual temperatures outside of the mesh on the left and right faces based on the boundary conditions
*/
void Finite_Difference_Solver::update_lr_bc_temps()
{
	// Coarse mesh BCs
	for(int j = 0; j < num_coarse_vert_y; j++)
	for(int k = 0; k < num_coarse_vert_z; k++)
	{
		if(coarse_x_index_at_fine_mesh_start != 0)
		{
			if ((curr_sim_time >= trigger_time) && (curr_sim_time < trigger_time + trigger_duration))
			{
				coarse_lr_bc_temps[0][j][k] = -4.0*((coarse_x_step/thermal_conductivity)*(htc*(coarse_temp_mesh[0][j][k]-amb_temp)-trigger_flux) + (5.0/6.0)*coarse_temp_mesh[0][j][k] + (-3.0/2.0)*coarse_temp_mesh[1][j][k] + (1.0/2.0)*coarse_temp_mesh[2][j][k] + (-1.0/12.0)*coarse_temp_mesh[3][j][k]);
			}
			else
			{
				coarse_lr_bc_temps[0][j][k] = -4.0*((coarse_x_step*htc/thermal_conductivity)*(coarse_temp_mesh[0][j][k]-amb_temp) + (5.0/6.0)*coarse_temp_mesh[0][j][k] + (-3.0/2.0)*coarse_temp_mesh[1][j][k] + (1.0/2.0)*coarse_temp_mesh[2][j][k] + (-1.0/12.0)*coarse_temp_mesh[3][j][k]);
			}
		}
		
		if(coarse_x_index_at_fine_mesh_start + coarse_x_steps_per_fine_x_len != num_coarse_vert_x)
		{
			coarse_lr_bc_temps[1][j][k] = -4.0*((coarse_x_step*htc/thermal_conductivity)*(coarse_temp_mesh[num_coarse_vert_x-1][j][k]-amb_temp) + (5.0/6.0)*coarse_temp_mesh[num_coarse_vert_x-1][j][k] + (-3.0/2.0)*coarse_temp_mesh[num_coarse_vert_x-2][j][k] + (1.0/2.0)*coarse_temp_mesh[num_coarse_vert_x-3][j][k] + (-1.0/12.0)*coarse_temp_mesh[num_coarse_vert_x-4][j][k]);
		}
	}
	
	// Fine mesh BCs
	for(int j = 0; j < num_fine_vert_y; j++)
	for(int k = 0; k < num_fine_vert_z; k++)
	{
		// Determine location in coarse mesh
		int curr_coarse_y_index = (int)floor((double)j / (double)fine_y_steps_per_coarse_y_step);
		int curr_coarse_z_index = (int)floor((double)k / (double)fine_z_steps_per_coarse_z_step);
		
		// Left BC if fine mesh is on left edge of domain
		if(coarse_x_index_at_fine_mesh_start == 0)
		{
			if ((curr_sim_time >= trigger_time) && (curr_sim_time < trigger_time + trigger_duration))
			{
				fine_lr_bc_temps[0][j][k] = -4.0*((fine_x_step/thermal_conductivity)*(htc*(fine_temp_mesh[get_ind(0)][j][k]-amb_temp)-trigger_flux) + (5.0/6.0)*fine_temp_mesh[get_ind(0)][j][k] + (-3.0/2.0)*fine_temp_mesh[get_ind(1)][j][k] + (1.0/2.0)*fine_temp_mesh[get_ind(2)][j][k] + (-1.0/12.0)*fine_temp_mesh[get_ind(3)][j][k]);
			}
			else
			{
				fine_lr_bc_temps[0][j][k] = -4.0*((fine_x_step*htc/thermal_conductivity)*(fine_temp_mesh[get_ind(0)][j][k]-amb_temp) + (5.0/6.0)*fine_temp_mesh[get_ind(0)][j][k] + (-3.0/2.0)*fine_temp_mesh[get_ind(1)][j][k] + (1.0/2.0)*fine_temp_mesh[get_ind(2)][j][k] + (-1.0/12.0)*fine_temp_mesh[get_ind(3)][j][k]);
				
			}
		}
		// Left BC if fine mesh is in middle of domain
		else
		{
			
			fine_lr_bc_temps[0][j][k] = coarse_temp_mesh[coarse_x_index_at_fine_mesh_start-1][curr_coarse_y_index][curr_coarse_z_index];
		}
		
		// Right BC if fine mesh is on right edge of domain
		if(coarse_x_index_at_fine_mesh_start + coarse_x_steps_per_fine_x_len == num_coarse_vert_x)
		{
			fine_lr_bc_temps[1][j][k] = -4.0*((fine_x_step*htc/thermal_conductivity)*(fine_temp_mesh[get_ind(num_fine_vert_x-1)][j][k]-amb_temp) + (5.0/6.0)*fine_temp_mesh[get_ind(num_fine_vert_x-1)][j][k] + (-3.0/2.0)*fine_temp_mesh[get_ind(num_fine_vert_x-2)][j][k] + (1.0/2.0)*fine_temp_mesh[get_ind(num_fine_vert_x-3)][j][k] + (-1.0/12.0)*fine_temp_mesh[get_ind(num_fine_vert_x-4)][j][k]);
		}
		// Right BC if fine mesh is in middle of domain
		else
		{
			fine_lr_bc_temps[1][j][k] = coarse_temp_mesh[coarse_x_index_at_fine_mesh_start + coarse_x_steps_per_fine_x_len][curr_coarse_y_index][curr_coarse_z_index];
		}
		
	}
}

/** Updates the virtual temperatures outside of the mesh on the front and back faces based on the boundary conditions
*/
void Finite_Difference_Solver::update_fb_bc_temps()
{
	// Coarse mesh BCs
	for(int j = 0; j < num_coarse_vert_x; j++)
	for(int k = 0; k < num_coarse_vert_z; k++)
	{
		if( !((j > coarse_x_index_at_fine_mesh_start) && (j < coarse_x_index_at_fine_mesh_start+coarse_x_steps_per_fine_x_len-1)) )
		{
			coarse_fb_bc_temps[0][j][k] = -4.0*((coarse_y_step*htc/thermal_conductivity)*(coarse_temp_mesh[j][0][k]-amb_temp) + (5.0/6.0)*coarse_temp_mesh[j][0][k] + (-3.0/2.0)*coarse_temp_mesh[j][1][k] + (1.0/2.0)*coarse_temp_mesh[j][2][k] + (-1.0/12.0)*coarse_temp_mesh[j][3][k]);
			coarse_fb_bc_temps[1][j][k] = -4.0*((coarse_y_step*htc/thermal_conductivity)*(coarse_temp_mesh[j][num_coarse_vert_y-1][k]-amb_temp) + (5.0/6.0)*coarse_temp_mesh[j][num_coarse_vert_y-1][k] + (-3.0/2.0)*coarse_temp_mesh[j][num_coarse_vert_y-2][k] + (1.0/2.0)*coarse_temp_mesh[j][num_coarse_vert_y-3][k] + (-1.0/12.0)*coarse_temp_mesh[j][num_coarse_vert_y-4][k]);
		}
	}
	// Fine mesh BCs
	for(int j = 0; j < num_fine_vert_x; j++)
	for(int k = 0; k < num_fine_vert_z; k++)
	{
		fine_fb_bc_temps[0][get_ind(j)][k] = -4.0*((fine_y_step*htc/thermal_conductivity)*(fine_temp_mesh[get_ind(j)][0][k]-amb_temp) + (5.0/6.0)*fine_temp_mesh[get_ind(j)][0][k] + (-3.0/2.0)*fine_temp_mesh[get_ind(j)][1][k] + (1.0/2.0)*fine_temp_mesh[get_ind(j)][2][k] + (-1.0/12.0)*fine_temp_mesh[get_ind(j)][3][k]);
		fine_fb_bc_temps[1][get_ind(j)][k] = -4.0*((fine_y_step*htc/thermal_conductivity)*(fine_temp_mesh[get_ind(j)][num_fine_vert_y-1][k]-amb_temp) + (5.0/6.0)*fine_temp_mesh[get_ind(j)][num_fine_vert_y-1][k] + (-3.0/2.0)*fine_temp_mesh[get_ind(j)][num_fine_vert_y-2][k] + (1.0/2.0)*fine_temp_mesh[get_ind(j)][num_fine_vert_y-3][k] + (-1.0/12.0)*fine_temp_mesh[get_ind(j)][num_fine_vert_y-4][k]);
	}
}

/** Updates the virtual temperatures outside of the mesh on the top and bottom faces based on the boundary conditions
*/
void Finite_Difference_Solver::update_tb_bc_temps()
{
	// Coarse mesh BCs
	for(int j = 0; j < num_coarse_vert_x; j++)
	for(int k = 0; k < num_coarse_vert_y; k++)
	{
		if( !((j > coarse_x_index_at_fine_mesh_start) && (j < coarse_x_index_at_fine_mesh_start+coarse_x_steps_per_fine_x_len-1)) )
		{
			coarse_tb_bc_temps[0][j][k] = coarse_temp_mesh[j][k][0] - (coarse_z_step/thermal_conductivity)*(htc*(coarse_temp_mesh[j][k][0]-amb_temp)-input_mesh[j][k]);
			coarse_tb_bc_temps[1][j][k] = coarse_temp_mesh[j][k][num_coarse_vert_z-1] - (coarse_z_step*htc/thermal_conductivity)*(coarse_temp_mesh[j][k][num_coarse_vert_z-1]-amb_temp);
		}
	}
	
	// Fine mesh BCs
	for(int j = 0; j < num_fine_vert_x; j++)
	for(int k = 0; k < num_fine_vert_y; k++)
	{
		int curr_coarse_x_index = (int)floor((double)j / (double)fine_x_steps_per_coarse_x_step) + coarse_x_index_at_fine_mesh_start;
		int curr_coarse_y_index = (int)floor((double)k / (double)fine_y_steps_per_coarse_y_step);
		
		fine_tb_bc_temps[0][get_ind(j)][k] = fine_temp_mesh[get_ind(j)][k][0] - (fine_z_step/thermal_conductivity)*(htc*(fine_temp_mesh[get_ind(j)][k][0]-amb_temp)-input_mesh[curr_coarse_x_index][curr_coarse_y_index]);
		fine_tb_bc_temps[1][get_ind(j)][k] = fine_temp_mesh[get_ind(j)][k][num_fine_vert_z-1] - (fine_z_step*htc/thermal_conductivity)*(fine_temp_mesh[get_ind(j)][k][num_fine_vert_z-1]-amb_temp);
	}
}

/** Calculates the 7-point stencil 3D laplacian of the coarse mesh. 1st order in z direction, 2nd order in y direction, and 3rd order in x direction
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @return 7-point stencil, 3rd order, 3D laplacian at (i,j,k). 1st order in z direction, 2nd order in y direction, and 3rd order in x direction
*/
double Finite_Difference_Solver::get_coarse_laplacian(int i, int j, int k)
{
	double T_000 = coarse_temp_mesh[i][j][k];
	double d2t_dx2 = 0.0;
	double d2t_dy2 = 0.0;
	double d2t_dz2 = 0.0;
	
	// Right face BC
	if (i==0)
	{
		d2t_dx2 = (11.0/12.0)*coarse_lr_bc_temps[0][j][k] + (-5.0/3.0)*T_000 + (1.0/2.0)*coarse_temp_mesh[i+1][j][k] + (1.0/3.0)*coarse_temp_mesh[i+2][j][k] + (-1.0/12.0)*coarse_temp_mesh[i+3][j][k];
	}
	// Left face BC
	else if(i==num_coarse_vert_x-1)
	{
		d2t_dx2 = (-1.0/12.0)*coarse_temp_mesh[i-3][j][k] + (1.0/3.0)*coarse_temp_mesh[i-2][j][k] + (1.0/2.0)*coarse_temp_mesh[i-1][j][k] + (-5.0/3.0)*T_000 + (11.0/12.0)*coarse_lr_bc_temps[1][j][k];
	}
	// Bulk material
	else
	{
		int start_p = -3;
		start_p = (i==1) ? -1 : start_p;
		start_p = (i==2) ? -2 : start_p;
		start_p = (i==num_coarse_vert_x-3) ? -4 : start_p;
		start_p = (i==num_coarse_vert_x-2) ? -5 : start_p;
		for (int p = start_p; p < start_p + 7; p++)
		{
			d2t_dx2 += laplacian_consts[abs(start_p)-1][p-start_p] * coarse_temp_mesh[i+p][j][k];
		}
	}
	d2t_dx2 = d2t_dx2 / (coarse_x_step*coarse_x_step);
	
	// Front face BC
	if (j==0)
	{
		d2t_dy2 = (11.0/12.0)*coarse_fb_bc_temps[0][i][k] + (-5.0/3.0)*T_000 + (1.0/2.0)*coarse_temp_mesh[i][j+1][k] + (1.0/3.0)*coarse_temp_mesh[i][j+2][k] + (-1.0/12.0)*coarse_temp_mesh[i][j+3][k];
	}
	// Back face BC
	else if(j==num_coarse_vert_y-1)
	{
		d2t_dy2 = (-1.0/12.0)*coarse_temp_mesh[i][j-3][k] + (1.0/3.0)*coarse_temp_mesh[i][j-2][k] + (1.0/2.0)*coarse_temp_mesh[i][j-1][k] + (-5.0/3.0)*T_000 + (11.0/12.0)*coarse_fb_bc_temps[1][i][k];
	}
	// Bulk material
	else
	{
		int start_q = -3;
		start_q = (j==1) ? -1 : start_q;
		start_q = (j==2) ? -2 : start_q;
		start_q = (j==num_coarse_vert_y-3) ? -4 : start_q;
		start_q = (j==num_coarse_vert_y-2) ? -5 : start_q;
		for (int q = start_q; q < start_q + 7; q++)
		{
			d2t_dy2 += laplacian_consts[abs(start_q)-1][q-start_q] * coarse_temp_mesh[i][j+q][k];
		}
	}
	d2t_dy2 = d2t_dy2 / (coarse_y_step*coarse_y_step);
	
	// Top face BC
	if (k==0)
	{
		d2t_dz2 = coarse_tb_bc_temps[0][i][j] - 2.0*T_000 + coarse_temp_mesh[i][j][k+1];
	}
	// Bottom face BC
	else if(k==num_coarse_vert_z-1)
	{
		d2t_dz2 = coarse_temp_mesh[i][j][k-1] - 2.0*T_000 + coarse_tb_bc_temps[1][i][j];
	}
	// Bulk material
	else
	{
		d2t_dz2 = coarse_temp_mesh[i][j][k-1] - 2.0*T_000 + coarse_temp_mesh[i][j][k+1];
	}
	d2t_dz2 = d2t_dz2 / (coarse_z_step*coarse_z_step);
	
	return d2t_dx2 + d2t_dy2 + d2t_dz2;
}

/** Calculates the 7-point stencil 3D laplacian of the fine mesh. 1st order in z direction, 2nd order in y direction, and 3rd order in x direction
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @return 7-point stencil, 3rd order, 3D laplacian at (i,j,k). 1st order in z direction, 2nd order in y direction, and 3rd order in x direction
*/
double Finite_Difference_Solver::get_fine_laplacian(int i, int j, int k)
{
	int i_ind = get_ind(i);
	double T_000 = fine_temp_mesh[i_ind][j][k];
	double d2t_dx2 = 0.0;
	double d2t_dy2 = 0.0;
	double d2t_dz2 = 0.0;
	
	// Right face BC
	if (i==0)
	{
		d2t_dx2 = (11.0/12.0)*fine_lr_bc_temps[0][j][k] + (-5.0/3.0)*T_000 + (1.0/2.0)*fine_temp_mesh[get_ind(i+1)][j][k] + (1.0/3.0)*fine_temp_mesh[get_ind(i+2)][j][k] + (-1.0/12.0)*fine_temp_mesh[get_ind(i+3)][j][k];
	}
	// Left face BC
	else if(i==num_fine_vert_x-1)
	{
		d2t_dx2 = (-1.0/12.0)*fine_temp_mesh[get_ind(i-1)][j][k] + (1.0/3.0)*fine_temp_mesh[get_ind(i-1)][j][k] + (1.0/2.0)*fine_temp_mesh[get_ind(i-1)][j][k] + (-5.0/3.0)*T_000 + (11.0/12.0)*fine_lr_bc_temps[1][j][k];
	}
	// Bulk material
	else
	{
		int start_p = -3;
		start_p = (i==1) ? -1 : start_p;
		start_p = (i==2) ? -2 : start_p;
		start_p = (i==num_fine_vert_x-3) ? -4 : start_p;
		start_p = (i==num_fine_vert_x-2) ? -5 : start_p;
		for (int p = start_p; p < start_p + 7; p++)
		{
			d2t_dx2 += laplacian_consts[abs(start_p)-1][p-start_p] * fine_temp_mesh[get_ind(i+p)][j][k];
		}
	}
	d2t_dx2 = d2t_dx2 / (fine_x_step*fine_x_step);
	
	// Front face BC
	if (j==0)
	{
		d2t_dy2 = (11.0/12.0)*fine_fb_bc_temps[0][i_ind][k] + (-5.0/3.0)*T_000 + (1.0/2.0)*fine_temp_mesh[i_ind][j+1][k] + (1.0/3.0)*fine_temp_mesh[i_ind][j+2][k] + (-1.0/12.0)*fine_temp_mesh[i_ind][j+3][k];
	}
	// Back face BC
	else if(j==num_fine_vert_y-1)
	{
		d2t_dy2 = (-1.0/12.0)*fine_temp_mesh[i_ind][j-1][k] + (1.0/3.0)*fine_temp_mesh[i_ind][j-1][k] + (1.0/2.0)*fine_temp_mesh[i_ind][j-1][k] + (-5.0/3.0)*T_000 + (11.0/12.0)*fine_fb_bc_temps[1][i_ind][k];
	}
	// Bulk material
	else
	{
		int start_q = -3;
		start_q = (j==1) ? -1 : start_q;
		start_q = (j==2) ? -2 : start_q;
		start_q = (j==num_fine_vert_y-3) ? -4 : start_q;
		start_q = (j==num_fine_vert_y-2) ? -5 : start_q;
		for (int q = start_q; q < start_q + 7; q++)
		{
			d2t_dy2 += laplacian_consts[abs(start_q)-1][q-start_q] * fine_temp_mesh[i_ind][j+q][k];
		}
	}
	d2t_dy2 = d2t_dy2 / (fine_y_step*fine_y_step);
	
	// Top face BC
	if (k==0)
	{
		d2t_dz2 = fine_tb_bc_temps[0][i_ind][j] - 2.0*T_000 + fine_temp_mesh[i_ind][j][k+1];
	}
	// Bottom face BC
	else if(k==num_fine_vert_z-1)
	{
		d2t_dz2 = fine_temp_mesh[i_ind][j][k-1] - 2.0*T_000 + fine_tb_bc_temps[1][i_ind][j];
	}
	// Bulk material
	else
	{
		d2t_dz2 = fine_temp_mesh[i_ind][j][k-1] - 2.0*T_000 + fine_temp_mesh[i_ind][j][k+1];
	}
	d2t_dz2 = d2t_dz2 / (fine_z_step*fine_z_step);
	
	return d2t_dx2 + d2t_dy2 + d2t_dz2;
}

/** Calculates the cure rate at every point in the 3D mesh and uses this data to update the cure, temperature, and front meshes
*/
void Finite_Difference_Solver::step_meshes()
{	
	// Temperature mesh variables
	update_lr_bc_temps();
	update_fb_bc_temps();
	update_tb_bc_temps();

	// Reset front calculation variables
	int num_front_instances = 0;
	front_mean_x_loc = 0.0;
	front_shape_param = 0.0;
	front_temp = 0.0;
	
	// Update the mesh
	#pragma omp parallel
	{	
		int local_front_instances = 0;
		double local_front_temp = 0.0;
		
		// ************************************************** Left coarse ************************************************** //
		// Calculate the laplacian mesh for the left side of the coarse mesh
		#pragma	omp for collapse(3)
		for (int i = 0; i < coarse_x_index_at_fine_mesh_start; i++)
		for (int j = 0; j < num_coarse_vert_y; j++)
		for (int k = 0; k < num_coarse_vert_z; k++)
		{
			coarse_laplacian_mesh[i][j][k] = get_coarse_laplacian(i, j, k);
		}
		
		
		// Update the temperature and cure mesh for the left side of the coarse mesh
		#pragma	omp for collapse(3) nowait
		for (int i = 0; i < coarse_x_index_at_fine_mesh_start; i++)
		for (int j = 0; j < num_coarse_vert_y; j++)
		for (int k = 0; k < num_coarse_vert_z; k++)
		{
			double cure_rate = 0.0;
			
			// Only calculate the cure rate if curing has started but is incomplete
			if ((coarse_temp_mesh[i][j][k] >= critical_temp) && (coarse_cure_mesh[i][j][k] < 1.0))
			{
				// Search precalculated array
				int precalc_exp_index = (int)round((coarse_temp_mesh[i][j][k]-precalc_start_temp) / precalc_temp_step);
				precalc_exp_index = precalc_exp_index < 0 ? 0 : precalc_exp_index;
				precalc_exp_index = precalc_exp_index >= precalc_exp_arr_len ? precalc_exp_arr_len-1 : precalc_exp_index;
				
				int precalc_pow_index = (int)round((coarse_cure_mesh[i][j][k]-precalc_start_cure) / precalc_cure_step);
				precalc_pow_index = precalc_pow_index < 0 ? 0 : precalc_pow_index;
				precalc_pow_index = precalc_pow_index >= precalc_pow_arr_len ? precalc_pow_arr_len-1 : precalc_pow_index;
				
				cure_rate = precalc_exp_arr[precalc_exp_index] * precalc_pow_arr[precalc_pow_index];
					
				// Limit cure rate such that a single time step will not yield a degree of cure greater than 1.0
				cure_rate = cure_rate > (1.0 - coarse_cure_mesh[i][j][k])/coarse_time_step ? (1.0 - coarse_cure_mesh[i][j][k])/coarse_time_step : cure_rate;
				cure_rate = cure_rate < 0.0 ? 0.0 : cure_rate;	
			}
			
			// Step the cure mesh
			coarse_cure_mesh[i][j][k] += coarse_time_step * cure_rate;
				
			// Ensure current cure is in expected range
			coarse_cure_mesh[i][j][k] = coarse_cure_mesh[i][j][k] > 1.0 ? 1.0 : coarse_cure_mesh[i][j][k];
			coarse_cure_mesh[i][j][k] = coarse_cure_mesh[i][j][k] < 0.0 ? 0.0 : coarse_cure_mesh[i][j][k];

			// Step temp mesh and ensure current temp is in expected range
			coarse_temp_mesh[i][j][k] += coarse_time_step * (thermal_diffusivity*coarse_laplacian_mesh[i][j][k]+(enthalpy_of_reaction*cure_rate)/specific_heat);
			coarse_temp_mesh[i][j][k] = coarse_temp_mesh[i][j][k] < min_possible_temp ? min_possible_temp : coarse_temp_mesh[i][j][k];
		}
		
		// ************************************************** Right coarse ************************************************** //
		// Calculate the laplacian mesh for the right side of the coarse mesh
		#pragma	omp for collapse(3)
		for (int i = coarse_x_index_at_fine_mesh_start + coarse_x_steps_per_fine_x_len; i < num_coarse_vert_x; i++)
		for (int j = 0; j < num_coarse_vert_y; j++)
		for (int k = 0; k < num_coarse_vert_z; k++)
		{
			coarse_laplacian_mesh[i][j][k] = get_coarse_laplacian(i, j, k);
		}
		
		
		// Update the temperature and cure mesh for the right side of the coarse mesh
		#pragma	omp for collapse(3) nowait
		for (int i = coarse_x_index_at_fine_mesh_start + coarse_x_steps_per_fine_x_len; i < num_coarse_vert_x; i++)
		for (int j = 0; j < num_coarse_vert_y; j++)
		for (int k = 0; k < num_coarse_vert_z; k++)
		{
			double cure_rate = 0.0;
				
			// Only calculate the cure rate if curing has started but is incomplete
			if ((coarse_temp_mesh[i][j][k] >= critical_temp) && (coarse_cure_mesh[i][j][k] < 1.0))
			{
				// Search precalculated array
				int precalc_exp_index = (int)round((coarse_temp_mesh[i][j][k]-precalc_start_temp) / precalc_temp_step);
				precalc_exp_index = precalc_exp_index < 0 ? 0 : precalc_exp_index;
				precalc_exp_index = precalc_exp_index >= precalc_exp_arr_len ? precalc_exp_arr_len-1 : precalc_exp_index;
				
				int precalc_pow_index = (int)round((coarse_cure_mesh[i][j][k]-precalc_start_cure) / precalc_cure_step);
				precalc_pow_index = precalc_pow_index < 0 ? 0 : precalc_pow_index;
				precalc_pow_index = precalc_pow_index >= precalc_pow_arr_len ? precalc_pow_arr_len-1 : precalc_pow_index;
				
				cure_rate = precalc_exp_arr[precalc_exp_index] * precalc_pow_arr[precalc_pow_index];
				
				// Limit cure rate such that a single time step will not yield a degree of cure greater than 1.0
				cure_rate = cure_rate > (1.0 - coarse_cure_mesh[i][j][k])/coarse_time_step ? (1.0 - coarse_cure_mesh[i][j][k])/coarse_time_step : cure_rate;
				cure_rate = cure_rate < 0.0 ? 0.0 : cure_rate;	
			}
			
			// Step the cure mesh
			coarse_cure_mesh[i][j][k] += coarse_time_step * cure_rate;
				
			// Ensure current cure is in expected range
			coarse_cure_mesh[i][j][k] = coarse_cure_mesh[i][j][k] > 1.0 ? 1.0 : coarse_cure_mesh[i][j][k];
			coarse_cure_mesh[i][j][k] = coarse_cure_mesh[i][j][k] < 0.0 ? 0.0 : coarse_cure_mesh[i][j][k];

			// Step temp mesh and ensure current temp is in expected range
			coarse_temp_mesh[i][j][k] += coarse_time_step * (thermal_diffusivity*coarse_laplacian_mesh[i][j][k]+(enthalpy_of_reaction*cure_rate)/specific_heat);
			coarse_temp_mesh[i][j][k] = coarse_temp_mesh[i][j][k] < min_possible_temp ? min_possible_temp : coarse_temp_mesh[i][j][k];
				
		}
		
		// ************************************************** Fine mesh ************************************************** //
		for(int subtime_ind = 0; subtime_ind < fine_time_steps_per_coarse_time_step; subtime_ind++)
		{		
			// Calculate the laplacian mesh for the fine section
			#pragma	omp for collapse(3)
			for (int i = 0; i < num_fine_vert_x; i++)
			for (int j = 0; j < num_fine_vert_y; j++)
			for (int k = 0; k < num_fine_vert_z; k++)
			{
				fine_laplacian_mesh[get_ind(i)][j][k] = get_fine_laplacian(i, j, k);
			}
		
			// Update the temperature and cure mesh for the fine mesh
			#pragma	omp for collapse(3) nowait
			for (int i = 0; i < num_fine_vert_x; i++)
			for (int j = 0; j < num_fine_vert_y; j++)
			for (int k = 0; k < num_fine_vert_z; k++)
			{
				int i_ind = get_ind(i);
				
				//double exponential_term = 0.0;
				double cure_rate = 0.0;
				double first_stage_cure_rate = 0.0;
				double second_stage_cure = 0.0;
				double second_stage_cure_rate = 0.0;
				double third_stage_cure = 0.0;
				double third_stage_cure_rate = 0.0;
				double fourth_stage_cure = 0.0;
				double fourth_stage_cure_rate = 0.0;
					
				// Only calculate the cure rate if curing has started but is incomplete
				if ((fine_temp_mesh[i_ind][j][k] >= critical_temp) && (fine_cure_mesh[i_ind][j][k] < 1.0))
				{
					// Search precalculated array
					int precalc_exp_index = (int)round((fine_temp_mesh[i_ind][j][k]-precalc_start_temp) / precalc_temp_step);
					precalc_exp_index = precalc_exp_index < 0 ? 0 : precalc_exp_index;
					precalc_exp_index = precalc_exp_index >= precalc_exp_arr_len ? precalc_exp_arr_len-1 : precalc_exp_index;
					
					int precalc_pow_index = (int)round((fine_cure_mesh[i_ind][j][k]-precalc_start_cure) / precalc_cure_step);
					precalc_pow_index = precalc_pow_index < 0 ? 0 : precalc_pow_index;
					precalc_pow_index = precalc_pow_index >= precalc_pow_arr_len ? precalc_pow_arr_len-1 : precalc_pow_index;
					
					first_stage_cure_rate = precalc_exp_arr[precalc_exp_index] * precalc_pow_arr[precalc_pow_index];
					
					if( first_stage_cure_rate < transition_cure_rate )
					{
						cure_rate = first_stage_cure_rate;
					}
					else
					{
						// Stage 2
						second_stage_cure = fine_cure_mesh[i_ind][j][k] + 0.5*fine_time_step*first_stage_cure_rate;
						if(second_stage_cure<1.0)
						{
							precalc_pow_index = (int)round((second_stage_cure-precalc_start_cure) / precalc_cure_step);
							precalc_pow_index = precalc_pow_index < 0 ? 0 : precalc_pow_index;
							precalc_pow_index = precalc_pow_index >= precalc_pow_arr_len ? precalc_pow_arr_len-1 : precalc_pow_index;
							second_stage_cure_rate = precalc_exp_arr[precalc_exp_index] * precalc_pow_arr[precalc_pow_index];
						}
						else {second_stage_cure_rate=0.0;}
						
						// Stage 3
						third_stage_cure = fine_cure_mesh[i_ind][j][k] + 0.5*fine_time_step*second_stage_cure_rate;
						if(third_stage_cure<1.0)
						{
							precalc_pow_index = (int)round((second_stage_cure-precalc_start_cure) / precalc_cure_step);
							precalc_pow_index = precalc_pow_index < 0 ? 0 : precalc_pow_index;
							precalc_pow_index = precalc_pow_index >= precalc_pow_arr_len ? precalc_pow_arr_len-1 : precalc_pow_index;
							third_stage_cure_rate = precalc_exp_arr[precalc_exp_index] * precalc_pow_arr[precalc_pow_index];
						}
						else {third_stage_cure_rate=0.0;}
						
						// Stage 4
						fourth_stage_cure = fine_cure_mesh[i_ind][j][k] + fine_time_step*third_stage_cure_rate;
						if(fourth_stage_cure<1.0)
						{
							precalc_pow_index = (int)round((second_stage_cure-precalc_start_cure) / precalc_cure_step);
							precalc_pow_index = precalc_pow_index < 0 ? 0 : precalc_pow_index;
							precalc_pow_index = precalc_pow_index >= precalc_pow_arr_len ? precalc_pow_arr_len-1 : precalc_pow_index;
							fourth_stage_cure = precalc_exp_arr[precalc_exp_index] * precalc_pow_arr[precalc_pow_index];
						}
						else {fourth_stage_cure=0.0;}
						
						// Apply RK4 algorithm
						cure_rate = (first_stage_cure_rate + 2.0*second_stage_cure_rate + 2.0*third_stage_cure_rate + fourth_stage_cure_rate)/6.0;
					}
						
					
					// Limit cure rate such that a single time step will not yield a degree of cure greater than 1.0
					cure_rate = cure_rate > (1.0 - fine_cure_mesh[i_ind][j][k])/fine_time_step ? (1.0 - fine_cure_mesh[i_ind][j][k])/fine_time_step : cure_rate;
					cure_rate = cure_rate < 0.0 ? 0.0 : cure_rate;	
				}
				
				// Step the cure mesh
				fine_cure_mesh[i_ind][j][k] += fine_time_step * cure_rate;
					
				// Ensure current cure is in expected range
				fine_cure_mesh[i_ind][j][k] = fine_cure_mesh[i_ind][j][k] > 1.0 ? 1.0 : fine_cure_mesh[i_ind][j][k];
				fine_cure_mesh[i_ind][j][k] = fine_cure_mesh[i_ind][j][k] < 0.0 ? 0.0 : fine_cure_mesh[i_ind][j][k];

				// Step temp mesh and ensure current temp is in expected range
				fine_temp_mesh[i_ind][j][k] += fine_time_step * (thermal_diffusivity*fine_laplacian_mesh[i_ind][j][k]+(enthalpy_of_reaction*cure_rate)/specific_heat);
				fine_temp_mesh[i_ind][j][k] = fine_temp_mesh[i_ind][j][k] < min_possible_temp ? min_possible_temp : fine_temp_mesh[i_ind][j][k];
				
				if((subtime_ind==(fine_time_steps_per_coarse_time_step-1)) && (k==0) && (((fine_cure_mesh[i_ind][j][k] >= front_min_cure) && (fine_cure_mesh[i_ind][j][k] <= front_max_cure)) || (cure_rate >= front_min_cure_rate)) )
				{
					// Collect front location and shape information
					int thread_num = omp_get_thread_num();
					if(local_front_instances < max_front_instances)
					{
						thread_front_curve[thread_num][0][local_front_instances] = (double)i * fine_x_step + (double)coarse_x_index_at_fine_mesh_start * coarse_x_step;
						thread_front_curve[thread_num][1][local_front_instances] = (double)j * fine_y_step;
						local_front_instances++;
					}
					
					// Search for the highest temperature just behind the front
					if( fine_temp_mesh[i_ind][j][k] > local_front_temp )
					{
						local_front_temp = fine_temp_mesh[i_ind][j][k];
					}
					bool done = false;
					int search_i = i-1;
					while(!done)
					{
						if( (search_i>=0) && (fine_temp_mesh[get_ind(search_i)][j][k] > local_front_temp) )
						{
							local_front_temp = fine_temp_mesh[get_ind(search_i)][j][k];
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
			
			while( (i < local_front_instances-1) && (num_front_instances < max_front_instances-1) )
			{
				global_front_curve[0][num_front_instances] = thread_front_curve[thread_num][0][i];
				global_front_curve[1][num_front_instances] = thread_front_curve[thread_num][1][i];
					
				global_front_curve[0][num_front_instances] = global_front_curve[0][num_front_instances] > coarse_x_len ? coarse_x_len : global_front_curve[0][num_front_instances];
				global_front_curve[1][num_front_instances] = global_front_curve[1][num_front_instances] > coarse_y_len ? coarse_y_len : global_front_curve[1][num_front_instances];
					
				front_mean_x_loc += thread_front_curve[thread_num][0][i];
				front_shape_param += thread_front_curve[thread_num][0][i]*thread_front_curve[thread_num][0][i];
				
				front_temp = local_front_temp > front_temp ? local_front_temp : front_temp;
					
				i++;
				num_front_instances++;
			}
			
			// Mark the end of front curve data
			global_front_curve[0][num_front_instances] = -1.0;
			global_front_curve[1][num_front_instances] = -1.0;
		}
	}
	
	// Copy fine mesh results to coarse mesh
	copy_fine_to_coarse();
	
	// ************************************************** FRONT UPDATE ************************************************** //
	if (num_front_instances != 0)
	{
		// Determine quarter fine coarse_x_len normalized front x stdev and mean front x location
		front_shape_param = sqrt((front_shape_param/(double)num_front_instances) - (front_mean_x_loc/(double)num_front_instances)*(front_mean_x_loc/(double)num_front_instances)) / (0.25 * fine_x_len);
		front_shape_param = front_shape_param > 1.0 ? 1.0 : front_shape_param;
		front_mean_x_loc = front_mean_x_loc / (double)num_front_instances;

		// Calculate the average mean x location and sim time from front location history
		front_mean_x_loc_history_avg = front_mean_x_loc_history_avg + (front_mean_x_loc - front_mean_x_loc_history[0])/(double)front_mean_x_loc_history_len;
		double front_time_history_avg = curr_sim_time - (((double)front_mean_x_loc_history_len-1.0)/2.0)*coarse_time_step;
		
		// Update the front mean x location history
		front_mean_x_loc_history.push_back(front_mean_x_loc);
		front_mean_x_loc_history.pop_front();
		
		// Apply a simple linear regression to the front mean x location history to determine the front velocity
		double sample_covariance = 0.0;
		double sample_variance = 0.0;
		for(int i = 0; i < front_mean_x_loc_history_len; i++)
		{
			double delta_x = (curr_sim_time-coarse_time_step*((double)front_mean_x_loc_history_len-1.0-(double)i)) - front_time_history_avg;
			double delta_y = front_mean_x_loc_history[i] - front_mean_x_loc_history_avg;
			
			sample_covariance += delta_x*delta_y;
			sample_variance += delta_x*delta_x;
		}
		
		// Pass the front velocity signal through a SPLP filter
		double curr_front_vel = sample_variance==0.0 ? 0.0 : sample_covariance / sample_variance;
		front_vel += front_filter_alpha * ( curr_front_vel - front_vel );

		// Determine if fine mesh is to be slid to the right
		int avg_front_coarse_ind = (int)floor(front_mean_x_loc / coarse_x_step);
		int cen_fine_mesh_coarse_ind = (int)floor( (double)coarse_x_index_at_fine_mesh_start + (double)coarse_x_steps_per_fine_x_len/2.0 );
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
		front_temp = initial_temp;
	}
}

/**
* Steps the environments time and updates the target velocity
* Boolean that determines whether simulation is complete or not
*/
bool Finite_Difference_Solver::step_time()
{
	// Update the current time and check for simulation completion
	bool done = (curr_sim_step == get_num_sim_steps() - 1);
	if (!done)
	{
		curr_sim_time += coarse_time_step;
		curr_sim_step++;
	}

	return done;
}
