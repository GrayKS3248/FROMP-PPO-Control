#include "Finite_Difference_Solver.hpp"

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
		critical_temp = -COD_activiation_energy / (log((pow(1.0-initial_cure, -1.0*COD_model_fit_order)*pow(initial_cure, -1.0*COD_m_fit)*critical_cure_rate)/(initial_cure))*gas_const);
	}
	adiabatic_temp_of_rxn = initial_temp + ( (1.0 - initial_cure) * enthalpy_of_reaction) / specific_heat;
	
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
	for(int i = 0; i < num_coarse_vert_x; i++)
	for(int j = 0; j < num_coarse_vert_y; j++)
	for(int k = 0; k < num_coarse_vert_z; k++)
	{
		coarse_temp_mesh[i][j][k] = coarse_temp_mesh[i][j][k] > monomer_burn_temp ? monomer_burn_temp : coarse_temp_mesh[i][j][k];
		coarse_temp_mesh[i][j][k] = coarse_temp_mesh[i][j][k] < 0.0 ? 0.0 : coarse_temp_mesh[i][j][k];
		coarse_cure_mesh[i][j][k] = coarse_cure_mesh[i][j][k] > 1.0 ? 1.0 : coarse_cure_mesh[i][j][k];
		coarse_cure_mesh[i][j][k] = coarse_cure_mesh[i][j][k] < 0.0 ? 0.0 : coarse_cure_mesh[i][j][k];
	}
	
	// Determine coarse step sizes
	coarse_x_step = coarse_x_mesh[1][0][0];
	coarse_y_step = coarse_y_mesh[0][1][0];
	coarse_z_step = coarse_z_mesh[0][0][1];
	
	// Calculate coarse/fine ratios
	coarse_x_verts_per_fine_x_len = (int)ceil( (double)num_coarse_vert_x * (fine_x_len/coarse_x_len) );
	coarse_y_verts_per_fine_y_len = num_coarse_vert_y;
	coarse_z_verts_per_fine_z_len = num_coarse_vert_z;
	
	
	// ************************************************** FINE MESH ************************************************** //
	// Calculate fine x coarse_x_len such that the number of coarse steps per fine mesh x coarse_x_len is a whole number
	fine_x_len = coarse_x_len * (  ((double)coarse_x_verts_per_fine_x_len - 1.0) / ((double)num_coarse_vert_x - 1.0)  );
	fine_y_len = coarse_y_len;
	fine_z_len = coarse_z_len;

	// Calculate fine mesh vertices
	num_fine_vert_x = (coarse_x_verts_per_fine_x_len-1) * fine_x_resolution_multiplier + 1;
	num_fine_vert_y = (coarse_y_verts_per_fine_y_len-1) * fine_y_resolution_multiplier + 1;
	num_fine_vert_z = (coarse_z_verts_per_fine_z_len-1) * fine_z_resolution_multiplier + 1;
	
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
		double target_deviation_1 = max_target_deviation * 0.5*((double)rand()/(double)RAND_MAX - 0.5);
		double rand_dir = ((double)rand()/(double)RAND_MAX - 0.5);
		rand_dir = rand_dir / abs(rand_dir);
		double target_deviation_2 = max_target_deviation * 0.25*((double)rand()/(double)RAND_MAX + 3.0) * rand_dir;
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
	max_front_instances = 20 * num_fine_vert_y;
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
	num_front_instances = 0;
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
		if (!reaction)
		{
			precalc_exp_arr[i] = 0.0;
		}
		else
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
		
		// Find the index of the max value of the power array
		if (i!=0)
		{
			if ( precalc_pow_arr[i] > precalc_pow_arr[i-1] )
			{
				arg_max_precalc_pow_arr = i;
				max_precalc_pow_arr = precalc_pow_arr[i];
			}
		}
		
		// Step the precalculation alpha value
		curr_precalc_cure += precalc_cure_step;
	}


	// ************************************************** TRIGGER ************************************************** //
	// Disengage trigger if set to off
	if (!using_a_trigger)
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
		peak_input_mag = input_total_power / (3.14159265358979 * 0.2171472409514 * radius_of_input * radius_of_input);
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
* Gets the size of the domain in lengthwise direction
* @return Lengthwise size in meters of domain
*/
double Finite_Difference_Solver::get_coarse_x_len()
{
	return coarse_x_len;
}

/**
* Gets the size of the domain in widthwise direction
* @return Widthwise size in meters of domain
*/
double Finite_Difference_Solver::get_coarse_y_len()
{
	return coarse_y_len;
}

/**
* Gets the size of the domain in depthwise direction
* @return Depthwise size in meters of domain
*/
double Finite_Difference_Solver::get_coarse_z_len()
{
	return coarse_z_len;
}

/**
* Gets the volume of the simulation domain
* @return Mesh volume in [m^3]
*/
double Finite_Difference_Solver::get_volume()
{
	return coarse_x_len*coarse_y_len*coarse_z_len;
}

/**
* Gets the wetted surface area of the simulation domain
* @return Mesh surface volume in [m^2]
*/
double Finite_Difference_Solver::get_surface_area()
{
	return 2.0*coarse_x_len*coarse_y_len + 2.0*coarse_x_len*coarse_z_len + 2.0*coarse_y_len*coarse_z_len;
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

// ************************************************** INPUT GETTERS ************************************************** //
/**
* Gets the input magnitude and location
* @param Boolean flag that indicates whether the x and y location components are normalized against the simulation domain's dimensions
* @return The input's magnitude and location as a vector {m,x,y}
*/
vector<double> Finite_Difference_Solver::get_input_state(bool normalize)
{
	vector<double> ret_val = vector<double>(3, 0.0);
	ret_val[0] = input_percent;
	ret_val[1] = input_location[0];
	ret_val[2] = input_location[1];
	
	if (normalize)
	{
		ret_val[1] = ret_val[1] / coarse_x_len;
		ret_val[2] = ret_val[2] / coarse_y_len;
	}
	
	return ret_val;
}

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

/**
* Gets the total heat power being used by the thermal trigger
* @return Total heat power of trigger (watts)
*/
double Finite_Difference_Solver::get_trigger_power()
{
	double power = 0.0;
	
	// Trigger power
	if (trigger_is_on)
	{
		// Flux times area
		power += trigger_flux * coarse_y_len * coarse_z_len;
	}
	
	return power;
}

/**
* Gets the total heat power being used by the heat source
* @return Total heat power of source (watts)
*/
double Finite_Difference_Solver::get_source_power()
{
	double power = 0.0;
	
	// Input power
	if (input_is_on)
	{
		// Flux times area
		power += input_percent * input_total_power;
	}
	
	return power;
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


// ************************************************** MONOMER GETTERS ************************************************** //
/**
* Gets the temperature in kelvin at which the monomer will burn
* @return The monomer burn temp
*/
double Finite_Difference_Solver::get_monomer_burn_temp()
{
	return monomer_burn_temp;
}

/**
* Gets the adiabatic temperature of reaction based on the formula T_ad = T0 + Hr / Cp
* @return Adiabatic temperature of reaction in kelvin
*/
double Finite_Difference_Solver::get_adiabatic_temp_of_rxn()
{
	return adiabatic_temp_of_rxn;
}

/**
* Gets the specific heat of the monomer being used in [J/Kg-K]
* @return specific heat
*/
double Finite_Difference_Solver::get_specific_heat()
{
	return specific_heat;
}

/**
* Gets the density of the monomer being used in [Kg/m^3]
* @return density of the monomer
*/
double Finite_Difference_Solver::get_density()
{
	double density = 0.0;
	if (monomer_code==1)
	{
		density = DCPD_GC1_density;
	}
	else if (monomer_code==2)
	{
		density = DCPD_GC2_density;
	}
	else if (monomer_code==3)
	{
		density = COD_density;
	}
	return density;
}

/**
* Gets the thermal conductivity of the monomer being used in [W/m-K]
* @return thermal conductivity of the monomer
*/
double Finite_Difference_Solver::get_thermal_conductivity()
{
	double thermal_conductivity = 0.0;
	if (monomer_code==1)
	{
		thermal_conductivity = DCPD_GC1_thermal_conductivity;
	}
	else if (monomer_code==2)
	{
		thermal_conductivity = DCPD_GC2_thermal_conductivity;
	}
	else if (monomer_code==3)
	{
		thermal_conductivity = COD_thermal_conductivity;
	}
	return thermal_conductivity;
}


// ************************************************** BOUNDARY CONDITION GETTERS ************************************************** //
/**
* Gets the current mean initial temperature before perturbation
* @return Initial temperature in [K]
*/
double Finite_Difference_Solver::get_initial_temp()
{
	return initial_temp;
}

/**
* Gets the current heat transfer coefficient of the simulation (not the mean)
* @return Heat transfer coefficient in [W/m^2-K]
*/
double Finite_Difference_Solver::get_heat_transfer_coefficient()
{
	return 	htc;
}

/**
* Gets the current ambient temperature of the simulation (not the mean)
* @return Ambient temperature in [K]
*/
double Finite_Difference_Solver::get_ambient_temperature()
{
	return 	amb_temp;
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
* @param Boolean flag that indicates whether the returned temperature values should be normalized on the range [initial temperature, adiabatic temperature of reaction]
* @return The top layer of the temperature mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Difference_Solver::get_coarse_temp_z0(bool normalize)
{
	vector<vector<double>> ret_val = vector<vector<double>>(num_coarse_vert_x, vector<double>(num_coarse_vert_y, 0.0));
	for (int i = 0; i < num_coarse_vert_x; i++)
	{
		for (int j = 0; j < num_coarse_vert_y; j++)
		{
			if (normalize)
			{
				ret_val[i][j] = (coarse_temp_mesh[i][j][0] - initial_temp) / (adiabatic_temp_of_rxn  - initial_temp);
			}
			else
			{
				ret_val[i][j] = coarse_temp_mesh[i][j][0];				
			}
		}
	}
	return ret_val;
}

/**
* Gets the top layer of the fine temperature mesh
* @param Boolean flag that indicates whether the returned temperature values should be normalized on the range [initial temperature, adiabatic temperature of reaction]
* @return The top layer of the temperature mesh as a 2D vector in x,y
*/
vector<vector<double>> Finite_Difference_Solver::get_fine_temp_z0(bool normalize)
{
	vector<vector<double>> ret_val = vector<vector<double>>(num_fine_vert_x, vector<double>(num_fine_vert_y, 0.0));
	for (int i = 0; i < num_fine_vert_x; i++)
	{
		for (int j = 0; j < num_fine_vert_y; j++)
		{
			if (normalize)
			{
				ret_val[i][j] = (fine_temp_mesh[get_ind(i)][j][0] - initial_temp) / (adiabatic_temp_of_rxn  - initial_temp);
			}
			else
			{
				ret_val[i][j] = fine_temp_mesh[get_ind(i)][j][0];
			}
		}
	}
	return ret_val;
}


/**
* Gets the top layer of the coarse temperture grid directly over the fine grid
* @param Boolean flag that indicates whether the returned temperature values should be normalized on the range [initial temperature, adiabatic temperature of reaction]
* @return The top layer of the coarse temperture grid directly over the fine grid
*/
vector<vector<double>> Finite_Difference_Solver::get_coarse_temp_around_front_z0(bool normalize)
{
	vector<vector<double>> ret_val = vector<vector<double>>(coarse_x_verts_per_fine_x_len, vector<double>(coarse_y_verts_per_fine_y_len, 0.0));
	for(int i = 0; i < coarse_x_verts_per_fine_x_len; i++)
	for(int j = 0; j < coarse_y_verts_per_fine_y_len; j++)
	{
			if (normalize)
			{
				ret_val[i][j] = (coarse_temp_mesh[coarse_x_index_at_fine_mesh_start+i][j][0] - initial_temp) / (adiabatic_temp_of_rxn  - initial_temp);
			}
			else
			{
				ret_val[i][j] = coarse_temp_mesh[coarse_x_index_at_fine_mesh_start+i][j][0];
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
* Gets a n order polynomial fit of the x position of the detected front as a function of the y coordinate
* @param Order of polynomial fit
* @return Vector containing fit data
**/
vector<double> Finite_Difference_Solver::get_front_fit(unsigned int order)
{
	// Create a vector to hold the fit data
	vector<double> ret_val = vector<double>(order+1, 0.0);
	
	// Convert global front curve to front x and y coords
	vector<double> front_x_coords;
	vector<double> front_y_coords;
	bool done = false;
	for(int i = 0; i < max_front_instances; i++)
	{
		done = (global_front_curve[0][i] < 0.0) || (global_front_curve[1][i] < 0.0);
		if(!done)
		{
			front_x_coords.push_back(global_front_curve[0][i]);
			front_y_coords.push_back(global_front_curve[1][i]);
		}
		else
		{
			break;
		}
	}
	
	// Ensure there are sufficient front instances to apply fit
	if(front_x_coords.size() <= order)
	{
		return ret_val;
	}
	
	// Calculate the entires of the A matrix
	vector<double> A_entries = vector<double>(2*order+1, 0.0);
	A_entries[0] = (double)front_y_coords.size();
	for ( unsigned int i=1; i<2*order+1; i++ )
	{
		double sum = 0.0;
		for ( unsigned int j=0; j<front_y_coords.size(); j++ )
		{
			sum += pow(front_y_coords[j], (double)i);
		}
		A_entries[i] = sum;
	}
	
	// Arrange the entries of the A matrix
	vector<vector<double>> A_matrix = vector<vector<double>>(order+1, vector<double>(order+1, 0.0));
	for( unsigned int i=0; i<order+1; i++ )
	for( unsigned int j=0; j<order+1; j++ )
	{
		 A_matrix[i][j] = A_entries[i+j];
	}		    
		
	// Invert the A matrix
	vector<vector<double>> inv_A = get_inv(A_matrix);

	// Create and Populate the B vector
	vector<double> B_vector = vector<double>(order+1, 0.0);	
	for( unsigned int i=0; i < order+1; i++ )
	{
		double sum = 0.0;
		for ( unsigned int j=0; j<front_x_coords.size(); j++ )
		{
			sum += pow(front_y_coords[j], (double)i) * front_x_coords[j];
		}
		B_vector[i] = sum;
	}
	
	// Calculate A^-1 * B
	ret_val = mat_vec_mul(inv_A, B_vector);
	    
	return ret_val;
}

/**
* Gets the current front mean x location.
* @param Boolean flag that indicates whether data should be normalized against domain x length
* @return Front's mean x location at the current time point
*/
double Finite_Difference_Solver::get_front_mean_x_loc(bool normalize)
{
	double ret_val = front_mean_x_loc_history.back();
	if(normalize)
	{
		ret_val = ret_val / coarse_x_len;
	}
	
	return ret_val;
}


/**
* Gets the current front velocity
* @param Boolean flag that indicates whether data should be sudo normalized
* @return The current mean front velocity
*/
double Finite_Difference_Solver::get_front_vel(bool normalize)
{
	if (normalize)
	{
		return front_vel * 1000.0;
	}
	else
	{
		return front_vel;
	}
}

/**
* Gets the current front temperature
* @return The current front mean temperature
*/
double Finite_Difference_Solver::get_front_temp(bool normalize)
{
	if(normalize)
	{
		return (front_temp - initial_temp) / (adiabatic_temp_of_rxn  - initial_temp);
	}
	
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
* Prints the finite difference solver parameters to std out
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
	if (!using_a_trigger)
	{
		cout << "  No trigger.\n";
	}
	else if (using_a_trigger)
	{ 
		cout << "  (Flux): " << trigger_flux << " W/m^2\n";
		cout << "  (Time): " << trigger_time << " s\n";
		if(trigger_duration > 0.0)
		{
			cout << "  (Duration): " << trigger_duration  << " s\n";
		}
		else
		{
			cout << "  (Duration): Minimum\n";
		}
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
	if (reaction)
	{
		cout << "  (Reaction): True\n";	
	}
	else
	{
		cout << "  (Reaction): False\n";	
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
	cout << ")\n";
	
	// Fine Mesh
	cout << "\nFine Mesh(\n";
	cout << "  (Fine Dimensions): " << 1000.0*fine_x_len << " x " << 1000.0*fine_y_len << " x " << 1000.0*fine_z_len << " mm\n";
	cout << "  (Fine Grid Size): " << 1e6*fine_x_len/(double)num_fine_vert_x << " x " << 1e6*fine_y_len/(double)num_fine_vert_y << " x " << 1e6*fine_z_len/(double)num_fine_vert_z << " um\n";
	cout << "  (Fine Grid Vertices): " << num_fine_vert_x << " x " << num_fine_vert_y << " x " << num_fine_vert_z << "\n";
	cout << "  (Fine Time Step): " << 1000.0*fine_time_step << " ms\n";
	cout << ")\n";
	
	// Environment
	cout << "\nEnvironment(\n";
	cout << "  (Duration): " << sim_duration << " s\n";
	cout << "  (Ambient Temperature): " << mean_amb_temp-273.15 << " C +- " << max_amb_temp_deviation << " C\n";
	cout << "  (HTC): " << mean_htc << " W/m^2-K +- " << max_htc_deviation << " W/m^2-K\n";
	cout << ")\n\n";
}

/**
* Prints the current completion percent of the current simulation
*/
void Finite_Difference_Solver::print_progress(bool return_carriage)
{
	// Percent complete sub messege
	int percent_complete = (int)round(100.0*(double)curr_sim_step / (double)get_num_sim_steps());
	stringstream stream;
	stream << percent_complete;
	string msg1 = stream.str();
	msg1.append("% Complete.");
	if (msg1.length() < 17)
	{
		msg1.append(17 - msg1.length(), ' ');
	}
	
	// Print all sub messeges
	if (return_carriage)
	{
		cout << msg1 << "\r";
	}
	else
	{
		cout << msg1 << "\n";
	}
}

/**
* Gets the current completion percent of the current simulation
* @return double of current progress
*/
double Finite_Difference_Solver::get_progress()
{
	// Percent complete sub messege
	return 100.0*(double)curr_sim_step / (double)get_num_sim_steps();
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
	for(int i = 0; i < num_coarse_vert_x; i++)
	for(int j = 0; j < num_coarse_vert_y; j++)
	for(int k = 0; k < num_coarse_vert_z; k++)
	{
		coarse_temp_mesh[i][j][k] = coarse_temp_mesh[i][j][k] > monomer_burn_temp ? monomer_burn_temp : coarse_temp_mesh[i][j][k];
		coarse_temp_mesh[i][j][k] = coarse_temp_mesh[i][j][k] < 0.0 ? 0.0 : coarse_temp_mesh[i][j][k];
		coarse_cure_mesh[i][j][k] = coarse_cure_mesh[i][j][k] > 1.0 ? 1.0 : coarse_cure_mesh[i][j][k];
		coarse_cure_mesh[i][j][k] = coarse_cure_mesh[i][j][k] < 0.0 ? 0.0 : coarse_cure_mesh[i][j][k];
	}
	
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
	num_front_instances = 0;
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
* @param normalized X command (-1.0, 1.0)
* @param normalized Y command (-1.0, 1.0)
* @param magnitude percent command (-1.0, 1.0)
* @return Whether the sim is done or not
*/
bool Finite_Difference_Solver::step(double x_cmd, double y_cmd, double mag_cmd)
{
	// Determine state of trigger
	step_trigger();
	
	// Step the input, cure, front, and temperature
	step_input(x_cmd, y_cmd, mag_cmd);
	step_meshes();

	// Step time
	bool done = step_time();
	
	return done;
}

/**
* Solves for the current state's reward
* @return The calculated reward
*/
vector<double> Finite_Difference_Solver::get_reward()
{
	// Initialize reward variables
	double input_loc_reward = 0.0;
	double input_mag_reward = 0.0;
	double max_temp_reward = 0.0;
	double front_shape_reward = 0.0;
	double target_reward = 0.0;

	// Determine maximum mesh temperature
	double num_coarse_vert_over_max_temp = 0.0;
	double num_fine_vert_over_max_temp = 0.0;
	int i_max = num_coarse_vert_x > num_fine_vert_x ? num_coarse_vert_x : num_fine_vert_x;
	int j_max = num_coarse_vert_y > num_fine_vert_y ? num_coarse_vert_y : num_fine_vert_y;
	int k_max = num_coarse_vert_z > num_fine_vert_z ? num_coarse_vert_z : num_fine_vert_z;
	#pragma omp parallel for collapse(3)
	for (int i = 0; i < i_max; i++)
	for (int j = 0; j < j_max; j++)
	for (int k = 0; k < k_max; k++)
	{
		if ( i < num_coarse_vert_x && j < num_coarse_vert_y && k < num_coarse_vert_z )
		{
			num_coarse_vert_over_max_temp = coarse_temp_mesh[i][j][k] >= monomer_burn_temp ? num_coarse_vert_over_max_temp + 1.0 : num_coarse_vert_over_max_temp;
		}
		if ( i < num_fine_vert_x && j < num_fine_vert_y && k < num_fine_vert_z )
		{
			num_fine_vert_over_max_temp = fine_temp_mesh[get_ind(i)][j][k] >= monomer_burn_temp ? num_fine_vert_over_max_temp + 1.0 : num_fine_vert_over_max_temp;
		}
	}
	
	// Get the input reward
	input_loc_reward = input_loc_reward_const * exp(-0.5 * pow(((input_location[0] - get_front_mean_x_loc(false))/(0.1647525572455*fine_x_len)), 2.0));
	input_mag_reward = input_mag_reward_const * (1.0 - input_percent);

	// Get the overage reward
	double norm_over = 0.50 * ((num_coarse_vert_over_max_temp / (double)num_coarse_vert_y) + (num_fine_vert_over_max_temp / (double)num_fine_vert_y));
	max_temp_reward = max_temp_reward_const * (1.0 - norm_over);

	// Get the front shape reward
	double clipped_front_shape_param = front_shape_param > 1.0 ? 1.0 : front_shape_param;
	front_shape_reward = front_shape_reward_const * pow((1.0 - clipped_front_shape_param), 6.64385618978);

	// Get the total reward
	if (control_code==1)
	{
		target_reward = target_reward_const * exp(-0.5 * pow(((front_vel-target_arr[curr_sim_step])/(0.078288147512*target_arr[curr_sim_step])), 2.0));
	}
	else if (control_code==2)
	{
		target_reward = target_reward_const * exp(-0.5 * pow(((front_temp-target_arr[curr_sim_step])/(0.03*target_arr[curr_sim_step])), 2.0));
	}

	// Generate return value
	vector<double> ret_val = vector<double>(6, 0.0);
	ret_val[0] = input_loc_reward + input_mag_reward + max_temp_reward + front_shape_reward + target_reward;
	ret_val[1] = input_loc_reward;
	ret_val[2] = input_mag_reward;
	ret_val[3] = max_temp_reward;
	ret_val[4] = front_shape_reward;
	ret_val[5] = target_reward;

	return ret_val;
}

// ================================================================================================= PRIVATE FUNCTIONS ================================================================================================= //

/**
* Loads FDS parameters from .cfg file
* @return 0 on success, 1 on failure
*/
int Finite_Difference_Solver::load_config()
{
	Config_Handler fds_cfg = Config_Handler("../config_files", "fds.cfg");
	

	// ************************************************** SIM OPTIONS ************************************************** //	
	string string_dump;
	fds_cfg.get_var("use_input",input_is_on);
	fds_cfg.get_var("use_trigger",using_a_trigger);
	fds_cfg.get_var("material",string_dump);
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
		return 1;
	}
	fds_cfg.get_var("reaction",reaction);
	fds_cfg.get_var("control",string_dump);
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
		return 1;
	}
	fds_cfg.get_var("target_type",string_dump);
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
		return 1;
	}
		
	// ************************************************** PROBLEM DEFINITION ************************************************** //
	fds_cfg.get_var("sim_duration",sim_duration);
	fds_cfg.get_var("burn_temp",monomer_burn_temp);
	fds_cfg.get_var("target_speed",mean_target_speed);
	fds_cfg.get_var("tar_speed_dev",max_target_speed_deviation);
	fds_cfg.get_var("target_temp",mean_target_temp);
	fds_cfg.get_var("tar_temp_dev",max_target_temp_deviation);
	
	// ************************************************** INITIAL CONDITIONS ************************************************** //
	fds_cfg.get_var("initial_temp",initial_temp);
	fds_cfg.get_var("temp_deviation",max_initial_temp_deviation);
	fds_cfg.get_var("initial_cure",initial_cure);
	fds_cfg.get_var("cure_deviation",max_initial_cure_deviation);
	
	// ************************************************** BOUNDARY CONDITIONS ************************************************** //
	fds_cfg.get_var("mean_htc",mean_htc);
	fds_cfg.get_var("htc_dev",max_htc_deviation);
	fds_cfg.get_var("mean_amb_temp",mean_amb_temp);
	fds_cfg.get_var("amb_temp_dev",max_amb_temp_deviation);
	
	// ************************************************** TRIGGER PARAMS ************************************************** //
	fds_cfg.get_var("trigger_flux",trigger_flux);
	fds_cfg.get_var("trigger_time",trigger_time);
	fds_cfg.get_var("trigger_len",string_dump);
	if (string_dump.compare("min")==0)
	{
		trigger_duration = -1.0;
	}
	else
	{
		trigger_duration = stof(string_dump, NULL);
	}
	
		
	// ************************************************** INPUT PARAMS ************************************************** //
	fds_cfg.get_var("radius",radius_of_input);
	fds_cfg.get_var("total_power",input_total_power);
	fds_cfg.get_var("max_mag_rate",max_input_mag_percent_rate);
	fds_cfg.get_var("max_slew_speed",max_input_slew_speed);
	
	// ************************************************** REWARD PARAMS ************************************************** //
	fds_cfg.get_var("input_loc",input_loc_reward_const);
	fds_cfg.get_var("input_mag",input_mag_reward_const);
	fds_cfg.get_var("max_temp",max_temp_reward_const);
	fds_cfg.get_var("front_shape",front_shape_reward_const);
	fds_cfg.get_var("target",target_reward_const);
			
	// ************************************************** TEMPORAL PARAMS ************************************************** //
	fds_cfg.get_var("time_step",coarse_time_step);
	fds_cfg.get_var("time_mult",fine_time_steps_per_coarse_time_step);
	
	// ************************************************** COARSE MESH PARAMS ************************************************** //	
	fds_cfg.get_var("num_vert_x",num_coarse_vert_x);
	fds_cfg.get_var("num_vert_y",num_coarse_vert_y);
	fds_cfg.get_var("num_vert_z",num_coarse_vert_z);
	fds_cfg.get_var("coarse_x_len",coarse_x_len);
	fds_cfg.get_var("coarse_y_len",coarse_y_len);
	fds_cfg.get_var("coarse_z_len",coarse_z_len);
	
	// ************************************************** FINE MESH PARAMS ************************************************** //
	fds_cfg.get_var("fine_x_len",fine_x_len);
	fds_cfg.get_var("x_step_mult",fine_x_resolution_multiplier);
	fds_cfg.get_var("y_step_mult",fine_y_resolution_multiplier);
	fds_cfg.get_var("z_step_mult",fine_z_resolution_multiplier);	
	
	// ************************************************** FRONT DETECTION PARAMS ************************************************** //
	fds_cfg.get_var("time_const",front_filter_time_const);
	fds_cfg.get_var("time_scale",front_mean_x_loc_history_time_len);
	fds_cfg.get_var("min_cure",front_min_cure);
	fds_cfg.get_var("max_cure",front_max_cure);
	fds_cfg.get_var("min_cure_rate",front_min_cure_rate);
	
	// ************************************************** CRITICAL CURE VALUES ************************************************** //
	fds_cfg.get_var("crit_cure_rate",critical_cure_rate);
	fds_cfg.get_var("trans_cure_rate",transition_cure_rate);
	
	// ************************************************** PRECALCULATION PARAMS ************************************************** //
	fds_cfg.get_var("start_temp",precalc_start_temp);
	fds_cfg.get_var("end_temp",precalc_end_temp);
	fds_cfg.get_var("temp_step",precalc_temp_step);
	fds_cfg.get_var("start_cure",precalc_start_cure);
	fds_cfg.get_var("end_cure",precalc_end_cure);
	fds_cfg.get_var("cure_step",precalc_cure_step);
	
	return 0;
}

/** Perturb mesh with smooth, 3D noise (improved Perlin)
* @ param array being perturbed
* @ param maximum magnitude of perturbation
* @ return sum of size_array and smooth continuous perturbation of magnitude delta
*/
void Finite_Difference_Solver::perturb_mesh(double*** arr, double max_deviation)
{	
	// Input parameters
	double gradient_grid_size_multiplier = 13.0;
	int noise_grid_multiplier = 1;

	// Base length scale on max length
	double length_scale = coarse_x_len;
	length_scale = coarse_y_len > length_scale ? coarse_y_len : length_scale;
	length_scale = coarse_z_len > length_scale ? coarse_z_len : length_scale;
	length_scale = length_scale / gradient_grid_size_multiplier;

	// Define a grid of random unit length gradient vectors
	int num_grad_vert_x = (int)round(coarse_x_len/length_scale);
	int num_grad_vert_y = (int)round(coarse_y_len/length_scale);
	int num_grad_vert_z = (int)round(coarse_z_len/length_scale);
	num_grad_vert_x = num_grad_vert_x < 3 ? 3 : num_grad_vert_x;
	num_grad_vert_y = num_grad_vert_y < 3 ? 3 : num_grad_vert_y;
	num_grad_vert_z = num_grad_vert_z < 3 ? 3 : num_grad_vert_z;
	
	// Populate gradient grid
	double grads[num_grad_vert_x][num_grad_vert_y][num_grad_vert_z][3];
	double grad_coords[num_grad_vert_x][num_grad_vert_y][num_grad_vert_z][3];
	for(int i = 0; i < num_grad_vert_x; i++)
	for(int j = 0; j < num_grad_vert_y; j++)
	for(int k = 0; k < num_grad_vert_z; k++)
	{
		// Generate random rotation angles
		double dirn_1 = (double)rand()/(double)RAND_MAX >= 0.50 ? 0.707107 : -0.707107;
		double dirn_2 = (double)rand()/(double)RAND_MAX >= 0.50 ? 0.707107 : -0.707107;
		double dirn_3 = (double)rand()/(double)RAND_MAX >= 0.50 ? 0.707107 : -0.707107;
		int zero_index = (int)round(2.99998*(double)rand()/(double)RAND_MAX - 0.49999); // 0 to 2
		
		// Convert euler angles to unit gradient vectors
		grads[i][j][k][0] = dirn_1;
		grads[i][j][k][1] = dirn_2;
		grads[i][j][k][2] = dirn_3;
		grads[i][j][k][zero_index] = 0.0;
		
		// Store the x,y,z location of the gradient grid
		grad_coords[i][j][k][0] = ((double)i / (double)(num_grad_vert_x - 1)) * coarse_x_len;
		grad_coords[i][j][k][1] = ((double)j / (double)(num_grad_vert_y - 1)) * coarse_y_len;
		grad_coords[i][j][k][2] = ((double)k / (double)(num_grad_vert_z - 1)) * coarse_z_len;
	}
	
	// Compute the noise
	double min_noise = 1e16;
	double max_noise = -1e16;
	double noise[num_grad_vert_x*noise_grid_multiplier-1][num_grad_vert_y*noise_grid_multiplier-1][num_grad_vert_z*noise_grid_multiplier-1];
	double noise_coords[num_grad_vert_x*noise_grid_multiplier-1][num_grad_vert_y*noise_grid_multiplier-1][num_grad_vert_z*noise_grid_multiplier-1][3];
	for (int i = 0; i < num_grad_vert_x*noise_grid_multiplier-1; i++)
	for (int j = 0; j < num_grad_vert_y*noise_grid_multiplier-1; j++)
	for (int k = 0; k < num_grad_vert_z*noise_grid_multiplier-1; k++)
	{
		//save the coordinates of the current part of the grid
		noise_coords[i][j][k][0] = ((double)i / (double)((num_grad_vert_x*noise_grid_multiplier-1) - 1)) * coarse_x_len;
		noise_coords[i][j][k][1] = ((double)j / (double)((num_grad_vert_y*noise_grid_multiplier-1) - 1)) * coarse_y_len;
		noise_coords[i][j][k][2] = ((double)k / (double)((num_grad_vert_z*noise_grid_multiplier-1) - 1)) * coarse_z_len;

		// Get the indicies of the 8 vertices of the gradient grid that form a box that contains the current point
		int grad_i = (int) floor((((double)num_grad_vert_x - 1.0) * noise_coords[i][j][k][0])/coarse_x_len);
		int grad_j = (int) floor((((double)num_grad_vert_y - 1.0) * noise_coords[i][j][k][1])/coarse_y_len);
		int grad_k = (int) floor((((double)num_grad_vert_z - 1.0) * noise_coords[i][j][k][2])/coarse_z_len);
		int grad_min_i = grad_i != (num_grad_vert_x-1) ? grad_i : grad_i - 1;
		int grad_max_i = grad_i != (num_grad_vert_x-1) ? grad_i+1 : grad_i;
		int grad_min_j = grad_j != (num_grad_vert_y-1) ? grad_j : grad_j - 1;
		int grad_max_j = grad_j != (num_grad_vert_y-1) ? grad_j+1 : grad_j;
		int grad_min_k = grad_k != (num_grad_vert_z-1) ? grad_k : grad_k - 1;
		int grad_max_k = grad_k != (num_grad_vert_z-1) ? grad_k+1 : grad_k;
		
		// Get the 8 offsets
		double offsets[8][3];
		for (int dim = 0; dim < 3; dim++)
		{
			offsets[0][dim] = noise_coords[i][j][k][dim] - grad_coords[grad_min_i][grad_min_j][grad_min_k][dim];
			offsets[1][dim] = noise_coords[i][j][k][dim] - grad_coords[grad_min_i][grad_min_j][grad_max_k][dim];
			offsets[2][dim] = noise_coords[i][j][k][dim] - grad_coords[grad_min_i][grad_max_j][grad_min_k][dim];
			offsets[3][dim] = noise_coords[i][j][k][dim] - grad_coords[grad_min_i][grad_max_j][grad_max_k][dim];
			offsets[4][dim] = noise_coords[i][j][k][dim] - grad_coords[grad_max_i][grad_min_j][grad_min_k][dim];
			offsets[5][dim] = noise_coords[i][j][k][dim] - grad_coords[grad_max_i][grad_min_j][grad_max_k][dim];
			offsets[6][dim] = noise_coords[i][j][k][dim] - grad_coords[grad_max_i][grad_max_j][grad_min_k][dim];
			offsets[7][dim] = noise_coords[i][j][k][dim] - grad_coords[grad_max_i][grad_max_j][grad_max_k][dim];
		}
		
		// Calculate and store the noise value
		double noise_val = 0.0;
		noise_val += offsets[0][0]*grads[grad_min_i][grad_min_j][grad_min_k][0] + offsets[0][1]*grads[grad_min_i][grad_min_j][grad_min_k][1] + offsets[0][2]*grads[grad_min_i][grad_min_j][grad_min_k][2];
		noise_val += offsets[1][0]*grads[grad_min_i][grad_min_j][grad_max_k][0] + offsets[1][1]*grads[grad_min_i][grad_min_j][grad_max_k][1] + offsets[1][2]*grads[grad_min_i][grad_min_j][grad_max_k][2];
		noise_val += offsets[2][0]*grads[grad_min_i][grad_max_j][grad_min_k][0] + offsets[2][1]*grads[grad_min_i][grad_max_j][grad_min_k][1] + offsets[2][2]*grads[grad_min_i][grad_max_j][grad_min_k][2];
		noise_val += offsets[3][0]*grads[grad_min_i][grad_max_j][grad_max_k][0] + offsets[3][1]*grads[grad_min_i][grad_max_j][grad_max_k][1] + offsets[3][2]*grads[grad_min_i][grad_max_j][grad_max_k][2];
		noise_val += offsets[4][0]*grads[grad_max_i][grad_min_j][grad_min_k][0] + offsets[4][1]*grads[grad_max_i][grad_min_j][grad_min_k][1] + offsets[4][2]*grads[grad_max_i][grad_min_j][grad_min_k][2];
		noise_val += offsets[5][0]*grads[grad_max_i][grad_min_j][grad_max_k][0] + offsets[5][1]*grads[grad_max_i][grad_min_j][grad_max_k][1] + offsets[5][2]*grads[grad_max_i][grad_min_j][grad_max_k][2];
		noise_val += offsets[6][0]*grads[grad_max_i][grad_max_j][grad_min_k][0] + offsets[6][1]*grads[grad_max_i][grad_max_j][grad_min_k][1] + offsets[6][2]*grads[grad_max_i][grad_max_j][grad_min_k][2];
		noise_val += offsets[7][0]*grads[grad_max_i][grad_max_j][grad_max_k][0] + offsets[7][1]*grads[grad_max_i][grad_max_j][grad_max_k][1] + offsets[7][2]*grads[grad_max_i][grad_max_j][grad_max_k][2];
		noise[i][j][k] = noise_val;
		
		// Save min and max values
		min_noise = noise_val < min_noise ? noise_val : min_noise;
		max_noise = noise_val > max_noise ? noise_val : max_noise;
	}
	
	// Assign the computed noise normalized to up to +- max_deviation
	double deviation = (double)rand()/(double)RAND_MAX*max_deviation;
	double bias = ((2.0*(double)rand()/(double)RAND_MAX)-1.0)*(max_deviation - deviation);
	for (int i = 0; i < num_coarse_vert_x; i++)
	for (int j = 0; j < num_coarse_vert_y; j++)
	for (int k = 0; k < num_coarse_vert_z; k++)
	{
		//save the coordinates of the current part of the grid
		double x_coord = coarse_x_mesh[i][j][k];
		double y_coord = coarse_y_mesh[i][j][k];
		double z_coord = coarse_z_mesh[i][j][k];

		// Get the indicies of the 8 vertices of the noise grid that form a box that contains the current point
		int noise_i = (int) floor((((double)(num_grad_vert_x*noise_grid_multiplier-1) - 1.0) * x_coord)/coarse_x_len);
		int noise_j = (int) floor((((double)(num_grad_vert_y*noise_grid_multiplier-1) - 1.0) * y_coord)/coarse_y_len);
		int noise_k = (int) floor((((double)(num_grad_vert_z*noise_grid_multiplier-1) - 1.0) * z_coord)/coarse_z_len);
		int noise_min_i = noise_i != ((num_grad_vert_x*noise_grid_multiplier-1)-1) ? noise_i : noise_i - 1;
		int noise_max_i = noise_i != ((num_grad_vert_x*noise_grid_multiplier-1)-1) ? noise_i+1 : noise_i;
		int noise_min_j = noise_j != ((num_grad_vert_y*noise_grid_multiplier-1)-1) ? noise_j : noise_j - 1;
		int noise_max_j = noise_j != ((num_grad_vert_y*noise_grid_multiplier-1)-1) ? noise_j+1 : noise_j;
		int noise_min_k = noise_k != ((num_grad_vert_z*noise_grid_multiplier-1)-1) ? noise_k : noise_k - 1;
		int noise_max_k = noise_k != ((num_grad_vert_z*noise_grid_multiplier-1)-1) ? noise_k+1 : noise_k;
		
		// Interpolate based on 3D shape function
		double x_d = (x_coord - noise_coords[noise_min_i][noise_min_j][noise_min_k][0]) / (noise_coords[noise_max_i][noise_min_j][noise_min_k][0] - noise_coords[noise_min_i][noise_min_j][noise_min_k][0]);
		x_d = 6.0*(x_d*x_d*x_d*x_d*x_d) - 15.0*(x_d*x_d*x_d*x_d) + 10.0*(x_d*x_d*x_d);
		double y_d = (y_coord - noise_coords[noise_min_i][noise_min_j][noise_min_k][1]) / (noise_coords[noise_min_i][noise_max_j][noise_min_k][1] - noise_coords[noise_min_i][noise_min_j][noise_min_k][1]);
		y_d = 6.0*(y_d*y_d*y_d*y_d*y_d) - 15.0*(y_d*y_d*y_d*y_d) + 10.0*(y_d*y_d*y_d);
		double z_d = (z_coord - noise_coords[noise_min_i][noise_min_j][noise_min_k][2]) / (noise_coords[noise_min_i][noise_min_j][noise_max_k][2] - noise_coords[noise_min_i][noise_min_j][noise_min_k][2]);
		z_d = 6.0*(z_d*z_d*z_d*z_d*z_d) - 15.0*(z_d*z_d*z_d*z_d) + 10.0*(z_d*z_d*z_d);
		double c00 = noise[noise_min_i][noise_min_j][noise_min_k]*(1.0 - x_d) + noise[noise_max_i][noise_min_j][noise_min_k]*(x_d);
		double c01 = noise[noise_min_i][noise_min_j][noise_max_k]*(1.0 - x_d) + noise[noise_max_i][noise_min_j][noise_max_k]*(x_d);
		double c10 = noise[noise_min_i][noise_max_j][noise_min_k]*(1.0 - x_d) + noise[noise_max_i][noise_max_j][noise_min_k]*(x_d);
		double c11 = noise[noise_min_i][noise_max_j][noise_max_k]*(1.0 - x_d) + noise[noise_max_i][noise_max_j][noise_max_k]*(x_d);
		double c0 = c00*(1.0 - y_d) + c10*(y_d);
		double c1 = c01*(1.0 - y_d) + c11*(y_d);
		double c = c0*(1.0 - z_d) + c1*(z_d);
		arr[i][j][k] = arr[i][j][k] + deviation*(2.0*c-max_noise-min_noise)/(max_noise-min_noise) + bias;
	}
	
}

/** Determines whether the trigger is on or not
*/
void Finite_Difference_Solver::step_trigger()
{
	if (trigger_duration > 0.0)
	{
		trigger_is_on = ((curr_sim_time >= trigger_time) && (curr_sim_time < trigger_time + trigger_duration));
	}
	else
	{
		// The front is less than the width of the channel with a mean location of less than 2 fine steps
		trigger_is_on = ((double)num_front_instances <= (double)num_fine_vert_y) && (front_mean_x_loc_history[front_mean_x_loc_history_len-1] <= 2.0*fine_x_step);
	}
}

/** Step the input through time
* @param normalized X command (-1.0, 1.0) (speed in x dirn)
* @param normalized Y command (-1.0, 1.0) (speed in y dirn)
* @param magnitude percent command (-1.0, 1.0) (magnitude rate)
*/
void Finite_Difference_Solver::step_input(double x_cmd, double y_cmd, double mag_cmd)
{
	
	//Clip input commands
	x_cmd = x_cmd > 1.0 ? 1.0 : x_cmd;
	x_cmd = x_cmd < -1.0 ? -1.0 : x_cmd;
	y_cmd = y_cmd > 1.0 ? 1.0 : y_cmd;
	y_cmd = y_cmd < -1.0 ? -1.0 : y_cmd;
	mag_cmd = mag_cmd > 1.0 ? 1.0 : mag_cmd;
	mag_cmd = mag_cmd < -1.0 ? -1.0 : mag_cmd;
	
	// Convert x,y input commands to location rates
	double x_loc_rate = x_cmd * max_input_slew_speed;
	double y_loc_rate = y_cmd * max_input_slew_speed;
	
	// Update the input's x location from the location rate
	input_location[0] = input_location[0] + x_loc_rate * coarse_time_step;
	input_location[0] = input_location[0] > max_input_x_loc ? max_input_x_loc : input_location[0];
	input_location[0] = input_location[0] < min_input_x_loc ? min_input_x_loc : input_location[0];
	int input_x_index = (int)round(input_location[0] / coarse_x_step) + 1;
	
	// Update the input's y location from the location rate
	input_location[1] = input_location[1] + y_loc_rate * coarse_time_step;
	input_location[1] = input_location[1] > max_input_y_loc ? max_input_y_loc : input_location[1];
	input_location[1] = input_location[1] < min_input_y_loc ? min_input_y_loc : input_location[1];
	int input_y_index = (int)round(input_location[1] / coarse_y_step) + 1;

	// Set magnitude rate command to maximum value corresponding to rate direction
	double mag_delta = 0.5*(mag_cmd+1.0) - input_percent;
	double mag_percent_rate = 0.0;
	if ( mag_delta > 0.0 )
	{
		mag_percent_rate = max_input_mag_percent_rate;
		mag_percent_rate = mag_percent_rate > mag_delta/coarse_time_step ? mag_delta/coarse_time_step : mag_percent_rate;
		
	}
	else if ( mag_delta < 0.0 )
	{
		mag_percent_rate = -max_input_mag_percent_rate;
		mag_percent_rate = mag_percent_rate < mag_delta/coarse_time_step ? mag_delta/coarse_time_step : mag_percent_rate;
	}

	// Update the input's magnitude from the converted magnitude rate commands
	input_percent = input_percent + mag_percent_rate * coarse_time_step;
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
	fine_mesh_start_loc = coarse_x_mesh[coarse_x_index_at_fine_mesh_start][0][0];
	fine_mesh_end_loc = coarse_x_mesh[coarse_x_index_at_fine_mesh_start+coarse_x_verts_per_fine_x_len-1][0][0];

	// Assign coarse mesh values to their respective fine mesh counterparts
	for(int i = 0; i < num_fine_vert_x; i++)
	for(int j = 0; j < num_fine_vert_y; j++)
	for(int k = 0; k < num_fine_vert_z; k++)
	{	
		// Determine location in coarse mesh
		int curr_coarse_x_index = (int)ceil((double)i / (double)fine_x_resolution_multiplier);
		int curr_coarse_y_index = (int)ceil((double)j / (double)fine_y_resolution_multiplier);
		int curr_coarse_z_index = (int)ceil((double)k / (double)fine_z_resolution_multiplier);
		
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
	int coarse_grid_x_slice_being_added = coarse_x_index_at_fine_mesh_start + coarse_x_verts_per_fine_x_len;
	if(coarse_grid_x_slice_being_added >= num_coarse_vert_x)
	{
		return;
	}
	
	// Slide the starting x location for the fine grid over 1 coarse grid index
	coarse_x_index_at_fine_mesh_start++;
	fine_mesh_start_loc = coarse_x_mesh[coarse_x_index_at_fine_mesh_start][0][0];
	
	// Slide the ending x location for the fine grid over 1 coarse grid index
	fine_mesh_end_loc = coarse_x_mesh[coarse_x_index_at_fine_mesh_start+coarse_x_verts_per_fine_x_len-1][0][0];
			
	// Copy coarse values to fine mesh
	for(int i = 0; i < fine_x_resolution_multiplier; i++)
	for(int j = 0; j < num_fine_vert_y; j++)
	for(int k = 0; k < num_fine_vert_z; k++)
	{
		// Determine location in coarse mesh
		int curr_coarse_y_index = (int)ceil((double)j / (double)fine_y_resolution_multiplier);
		int curr_coarse_z_index = (int)ceil((double)k / (double)fine_z_resolution_multiplier);
		
		// Assign coarse values to fine mesh
		fine_temp_mesh[get_ind(i)][j][k] = coarse_temp_mesh[coarse_grid_x_slice_being_added][curr_coarse_y_index][curr_coarse_z_index];
		fine_cure_mesh[get_ind(i)][j][k] = coarse_cure_mesh[coarse_grid_x_slice_being_added][curr_coarse_y_index][curr_coarse_z_index];
	}

	// Update the fine mesh starting index
	fine_mesh_zero_index += fine_x_resolution_multiplier;
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
	for(int i = 0; i < coarse_x_verts_per_fine_x_len; i++)
	for(int j = 0; j < coarse_y_verts_per_fine_y_len; j++)
	for(int k = 0; k < coarse_z_verts_per_fine_z_len; k++)
	{
		// Calculate the range of fine mesh indices that correspond to the nodes closest to coarse node with index i,j,k
		int start_fine_x_index = i*fine_x_resolution_multiplier - (int)floor((double)fine_x_resolution_multiplier/2.0);
		int start_fine_y_index = j*fine_y_resolution_multiplier - (int)floor((double)fine_y_resolution_multiplier/2.0);
		int start_fine_z_index = k*fine_z_resolution_multiplier - (int)floor((double)fine_z_resolution_multiplier/2.0);
		
		start_fine_x_index = start_fine_x_index < 0 ? 0 : start_fine_x_index;
		start_fine_y_index = start_fine_y_index < 0 ? 0 : start_fine_y_index;
		start_fine_z_index = start_fine_z_index < 0 ? 0 : start_fine_z_index;
		
		int end_fine_x_index = i*fine_x_resolution_multiplier + (int)floor((double)(fine_x_resolution_multiplier-1)/2.0);
		int end_fine_y_index = j*fine_y_resolution_multiplier + (int)floor((double)(fine_y_resolution_multiplier-1)/2.0);
		int end_fine_z_index = k*fine_z_resolution_multiplier + (int)floor((double)(fine_z_resolution_multiplier-1)/2.0);
		
		end_fine_x_index = end_fine_x_index >= num_fine_vert_x ? (num_fine_vert_x-1) : end_fine_x_index;
		end_fine_y_index = end_fine_y_index >= num_fine_vert_y ? (num_fine_vert_y-1) : end_fine_y_index;
		end_fine_z_index = end_fine_z_index >= num_fine_vert_z ? (num_fine_vert_z-1) : end_fine_z_index;
		
		// Average the temperature and cure value over the closest fine nodes
		double temp_avg = 0.0;
		double cure_avg = 0.0;
		double counter = 0.0;
		for(int p = start_fine_x_index; p <= end_fine_x_index; p++)
		for(int q = start_fine_y_index; q <= end_fine_y_index; q++)
		for(int r = start_fine_z_index; r <= end_fine_z_index; r++)
		{	
			temp_avg += fine_temp_mesh[get_ind(p)][q][r];
			cure_avg += fine_cure_mesh[get_ind(p)][q][r];
			counter = counter + 1.0;
		}
		
		coarse_temp_mesh[coarse_x_index_at_fine_mesh_start+i][j][k] = temp_avg / counter;
		coarse_cure_mesh[coarse_x_index_at_fine_mesh_start+i][j][k] = cure_avg / counter;
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
		
		if(coarse_x_index_at_fine_mesh_start + coarse_x_verts_per_fine_x_len != num_coarse_vert_x)
		{
			coarse_lr_bc_temps[1][j][k] = -4.0*((coarse_x_step*htc/thermal_conductivity)*(coarse_temp_mesh[num_coarse_vert_x-1][j][k]-amb_temp) + (5.0/6.0)*coarse_temp_mesh[num_coarse_vert_x-1][j][k] + (-3.0/2.0)*coarse_temp_mesh[num_coarse_vert_x-2][j][k] + (1.0/2.0)*coarse_temp_mesh[num_coarse_vert_x-3][j][k] + (-1.0/12.0)*coarse_temp_mesh[num_coarse_vert_x-4][j][k]);
		}
	}
	
	// Fine mesh BCs
	for(int j = 0; j < num_fine_vert_y; j++)
	for(int k = 0; k < num_fine_vert_z; k++)
	{
		// Determine location in coarse mesh
		int curr_coarse_y_index = (int)ceil((double)j / (double)fine_y_resolution_multiplier);
		int curr_coarse_z_index = (int)ceil((double)k / (double)fine_z_resolution_multiplier);
		
		// Left BC if fine mesh is on left edge of domain
		if(coarse_x_index_at_fine_mesh_start == 0)
		{
			if (trigger_is_on)
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
		if(coarse_x_index_at_fine_mesh_start + coarse_x_verts_per_fine_x_len == num_coarse_vert_x)
		{
			fine_lr_bc_temps[1][j][k] = -4.0*((fine_x_step*htc/thermal_conductivity)*(fine_temp_mesh[get_ind(num_fine_vert_x-1)][j][k]-amb_temp) + (5.0/6.0)*fine_temp_mesh[get_ind(num_fine_vert_x-1)][j][k] + (-3.0/2.0)*fine_temp_mesh[get_ind(num_fine_vert_x-2)][j][k] + (1.0/2.0)*fine_temp_mesh[get_ind(num_fine_vert_x-3)][j][k] + (-1.0/12.0)*fine_temp_mesh[get_ind(num_fine_vert_x-4)][j][k]);
		}
		// Right BC if fine mesh is in middle of domain
		else
		{
			fine_lr_bc_temps[1][j][k] = coarse_temp_mesh[coarse_x_index_at_fine_mesh_start + coarse_x_verts_per_fine_x_len][curr_coarse_y_index][curr_coarse_z_index];
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
		if( !((j > coarse_x_index_at_fine_mesh_start) && (j < coarse_x_index_at_fine_mesh_start+coarse_x_verts_per_fine_x_len-1)) )
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
		if( !((j > coarse_x_index_at_fine_mesh_start) && (j < coarse_x_index_at_fine_mesh_start+coarse_x_verts_per_fine_x_len-1)) )
		{
			coarse_tb_bc_temps[0][j][k] = coarse_temp_mesh[j][k][0] - (coarse_z_step/thermal_conductivity)*(htc*(coarse_temp_mesh[j][k][0]-amb_temp)-input_mesh[j][k]);
			coarse_tb_bc_temps[1][j][k] = coarse_temp_mesh[j][k][num_coarse_vert_z-1] - (coarse_z_step*htc/thermal_conductivity)*(coarse_temp_mesh[j][k][num_coarse_vert_z-1]-amb_temp);
		}
	}
	
	// Fine mesh BCs
	for(int j = 0; j < num_fine_vert_x; j++)
	for(int k = 0; k < num_fine_vert_y; k++)
	{
		int curr_coarse_x_index = (int)floor((double)j / (double)fine_x_resolution_multiplier) + coarse_x_index_at_fine_mesh_start;
		int curr_coarse_y_index = (int)floor((double)k / (double)fine_y_resolution_multiplier);
		
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
	num_front_instances = 0;
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
		for (int i = coarse_x_index_at_fine_mesh_start + coarse_x_verts_per_fine_x_len; i < num_coarse_vert_x; i++)
		for (int j = 0; j < num_coarse_vert_y; j++)
		for (int k = 0; k < num_coarse_vert_z; k++)
		{
			coarse_laplacian_mesh[i][j][k] = get_coarse_laplacian(i, j, k);
		}
		
		
		// Update the temperature and cure mesh for the right side of the coarse mesh
		#pragma	omp for collapse(3) nowait
		for (int i = coarse_x_index_at_fine_mesh_start + coarse_x_verts_per_fine_x_len; i < num_coarse_vert_x; i++)
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
				double cure_rate = 0.0;
				double explicit_cure_rate = 0.0;
				double implicit_cure_rate = 0.0;
					
				// Only calculate the cure rate if curing has started but is incomplete
				if ((fine_temp_mesh[i_ind][j][k] >= critical_temp) && (fine_cure_mesh[i_ind][j][k] < 1.0))
				{
					// Save current degree cure at i,j,k to variables
					double curr_alpha = fine_cure_mesh[i_ind][j][k];
				
					// Search precalculated array
					int precalc_exp_index = (int)round((fine_temp_mesh[i_ind][j][k]-precalc_start_temp) / precalc_temp_step);
					precalc_exp_index = precalc_exp_index < 0 ? 0 : precalc_exp_index;
					precalc_exp_index = precalc_exp_index >= precalc_exp_arr_len ? precalc_exp_arr_len-1 : precalc_exp_index;
					double precalc_exp = precalc_exp_arr[precalc_exp_index];
					
					int precalc_pow_index = (int)round((curr_alpha-precalc_start_cure) / precalc_cure_step);
					precalc_pow_index = precalc_pow_index < 0 ? 0 : precalc_pow_index;
					precalc_pow_index = precalc_pow_index >= precalc_pow_arr_len ? precalc_pow_arr_len-1 : precalc_pow_index;
					
					explicit_cure_rate = precalc_exp * precalc_pow_arr[precalc_pow_index];
					
					if( explicit_cure_rate < transition_cure_rate )
					{
						cure_rate = explicit_cure_rate;
					}
					else
					{
						// Determine the maximum possible next degree of cure
						double max_alpha;
						if( precalc_pow_index > arg_max_precalc_pow_arr )
						{
							max_alpha = curr_alpha + fine_time_step * precalc_exp * precalc_pow_arr[precalc_pow_index];
						}
						else
						{
							max_alpha = curr_alpha + fine_time_step * precalc_exp * max_precalc_pow_arr;
						}
						max_alpha = max_alpha > 1.0 ? 1.0 : max_alpha;
						
						// Implicit Euler iterates
						double min_alpha = curr_alpha;
						double prev_error = 1.0;
						int count = 0;
						
						// Implicit Euler iterative solver (binary search)
						while (  abs(prev_error) > 1.0e-4 )
						{
							// Make new index guess in middle of known admissable range
							double guess_alpha = min_alpha + (max_alpha - min_alpha) / 2.0;
							precalc_pow_index = (int)round( (guess_alpha - precalc_start_cure) / precalc_cure_step );
							precalc_pow_index = precalc_pow_index < 0 ? 0 : precalc_pow_index;
							precalc_pow_index = precalc_pow_index >= precalc_pow_arr_len ? precalc_pow_arr_len-1 : precalc_pow_index;
							
							// Calcute error given new guess
							double curr_error = curr_alpha + fine_time_step * precalc_exp * precalc_pow_arr[precalc_pow_index] - guess_alpha;
							
							// Split range and continue
							if( curr_error > 0.0 ) // Underestimate a
							{
								min_alpha = guess_alpha;
							}
							else if ( curr_error < 0.0 ) // Overestimate a
							{
								max_alpha = guess_alpha;
							}
							else
							{
								break;
							}
							
							// Check termination condition
							if(count==14)
							{
								break;
							}
							
							// Step error
							prev_error = curr_error;
							count++;
							
						}
						implicit_cure_rate = precalc_exp * precalc_pow_arr[precalc_pow_index];
						
						// Apply Implicit Euler algorithm
						cure_rate = implicit_cure_rate;
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
		if( front_shape_param <= 1.0e-4 )
		{
			front_shape_param = 0.0;
		}
		else
		{
			front_shape_param = sqrt((front_shape_param/(double)num_front_instances) - (front_mean_x_loc/(double)num_front_instances)*(front_mean_x_loc/(double)num_front_instances)) / (0.10 * coarse_y_len);
		}
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
		int cen_fine_mesh_coarse_ind = (int)floor( (double)coarse_x_index_at_fine_mesh_start + (double)coarse_x_verts_per_fine_x_len/2.0 );
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