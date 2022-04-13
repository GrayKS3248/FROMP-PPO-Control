#define PY_SSIZE_T_CLEAN
#include "Config_Handler.hpp"
#include "python_interface.hpp"
#include "Speed_Estimator.hpp"
#include "Finite_Difference_Solver.hpp"

/**
* Runs a set of trajectories using the PPO policy, updates the PPO agent, and collects relevant training data
* @param The finite element solver object used to propogate time
* @param Configuration handler object that contains all loaded configuration data
* @param The class used to estimate front speed based on front location estimations
* @param The ppo agent be used to generate actions
* @param The save render and plotting class of the ppo agent being trained
* @param The time at which simulation was started
 *@param String containing cfg data
* @return 0 on success, 1 on failure
*/
int run(Finite_Difference_Solver* FDS, Config_Handler* solver_cfg, Speed_Estimator* estimator, PyObject* agent, PyObject* save_render_plot, auto &start_time, string configs_string)
{	
	// Get values from config file
	int num_input_actions;
	solver_cfg->get_var("num_input_actions",num_input_actions);
	double frame_rate;
	solver_cfg->get_var("frame_rate",frame_rate);
	
	// Frame rate calculations
	int steps_per_agent_cycle = (int) round((FDS->get_sim_duration() / (double)num_input_actions) / FDS->get_coarse_time_step());
	steps_per_agent_cycle = steps_per_agent_cycle <= 0 ? 1 : steps_per_agent_cycle;	
	int steps_per_speed_estimator_frame = (int) round(estimator->get_observation_delta_t() / FDS->get_coarse_time_step());
	steps_per_speed_estimator_frame = steps_per_speed_estimator_frame <= 0 ? 1 : steps_per_speed_estimator_frame;	
	int steps_per_render_frame = (int) round(1.0 / (FDS->get_coarse_time_step() * frame_rate));
	steps_per_render_frame = steps_per_render_frame <= 0 ? 1 : steps_per_render_frame;
	int steps_per_progress_update = (int) round(1.0 / (FDS->get_coarse_time_step() * (100.0 / FDS->get_sim_duration())));
	steps_per_progress_update = steps_per_progress_update <= 0 ? 1 : steps_per_progress_update;
	
	// Trajectory memory
	vector<double> input_location_x;
	vector<double> input_location_y;
	vector<double> input_percent;
	vector<double> trigger_power;
	vector<double> source_power;
	vector<double> time;
	vector<double> target;
	vector<vector<double>> reward;
	vector<double> front_velocity;
	vector<double> front_temperature;
	vector<double> front_shape_param;
	vector<vector<double>> fine_mesh_loc;
	vector<vector<vector<double>>> front_curve;
	vector<vector<double>> front_fit;
	vector<vector<vector<double>>> temperature_field;
	vector<vector<vector<double>>> cure_field;
	vector<vector<vector<double>>> fine_temperature_field;
	vector<vector<vector<double>>> fine_cure_field;
	
	// Run the simulation loop again to generate trajectory data
	bool done = false;
	double action_1=0.0, action_2=0.0, action_3=0.0;
	int step_in_trajectory = 0;
	
	// Reset
	FDS->reset();
	estimator->reset();
		
	// Run trajectory and save the results
	while (!done)
	{
		// Determine what to run this simulation step
		bool run_agent = (step_in_trajectory % steps_per_agent_cycle == 0) || (step_in_trajectory==0);
		bool observe_speed = (step_in_trajectory % steps_per_speed_estimator_frame == 0) || (step_in_trajectory==0);
		bool save_frame = (step_in_trajectory % steps_per_render_frame == 0) || (step_in_trajectory==0);
		bool print_progress = (step_in_trajectory % steps_per_progress_update == 0) || (step_in_trajectory==0);
		step_in_trajectory++;
		
		// Print progress
		if(print_progress)
		{
			FDS->print_progress(true);
		}
		
		// Add observation to speed estimator
		if (observe_speed && agent != NULL)
		{	
			// Get image and convert to canonical form
			PyObject* py_state_image = get_2D_list<vector<vector<double>>>(FDS->get_coarse_temp_z0(true));
			PyObject* py_canonical_state_image = PyObject_CallMethod(agent, "forward", "O", py_state_image);
			if (py_canonical_state_image == NULL)
			{
				fprintf(stderr, "\nFailed to call PPO forward function.\n");
				PyErr_Print();
				Py_DECREF(py_state_image);

				return 1;
			}
			vector<vector<double>> canonical_state_image = get_2D_vector(py_canonical_state_image);
			
			// Add canonical image and observation time to estimator buffer
			estimator->observe(canonical_state_image, FDS->get_curr_sim_time());
			
			// Cleanup
			Py_DECREF(py_state_image);
			Py_DECREF(py_canonical_state_image);
		}
		
		// Update the logs
		if (save_frame)
		{
			// Get environment data
			vector<double> input_state = FDS->get_input_state(false);
			
			// Store simulation input data
			input_percent.push_back(input_state[0]);
			input_location_x.push_back(input_state[1]);
			input_location_y.push_back(input_state[2]);
			trigger_power.push_back(FDS->get_trigger_power());
			source_power.push_back(FDS->get_source_power());
			
			// Store simualtion target and time data
			time.push_back(FDS->get_curr_sim_time());
			target.push_back(FDS->get_curr_target());
			reward.push_back(FDS->get_reward());
			
			// Store simulation front data
			front_velocity.push_back(FDS->get_front_vel(false));
			front_temperature.push_back(FDS->get_front_temp(false));
			front_shape_param.push_back(FDS->get_front_shape_param());
			front_curve.push_back(FDS->get_front_curve());
			front_fit.push_back(FDS->get_front_fit(3));
			
			// Store fine mesh data
			fine_mesh_loc.push_back(FDS->get_fine_mesh_loc());
			
			// Store simulation field data
			temperature_field.push_back(FDS->get_coarse_temp_z0(true));
			cure_field.push_back(FDS->get_coarse_cure_z0());
			fine_temperature_field.push_back(FDS->get_fine_temp_z0(true));
			fine_cure_field.push_back(FDS->get_fine_cure_z0());
			
		}
		
		// Run the agent
		if (run_agent)
		{
			// Get loaded agent actions
			if(agent != NULL)
			{
				// Gather temperature state data
				PyObject* py_state_image = get_2D_list<vector<vector<double>>>(FDS->get_coarse_temp_z0(true));
				
				// Get speed from speed estimator and calculate error
				double front_speed = estimator->estimate();
				double front_speed_error = front_speed / FDS->get_curr_target();
				//double front_speed_error = 10.0*(front_speed - FDS->get_curr_target()) / FDS->get_curr_target();
				
				// Combine all additional inputs to PPO agent
				vector<double> additional_ppo_inputs(1, 0.0);
				additional_ppo_inputs[0] = front_speed_error;
				PyObject* py_additional_ppo_inputs = get_1D_list<vector<double>>(additional_ppo_inputs);
				
				// Gather input data
				PyObject* py_inputs = get_1D_list<vector<double>>(FDS->get_input_state(true));
				
				// Get agent action based on temperature state data
				PyObject* py_action_and_stdev = PyObject_CallMethod(agent, "get_action", "(O,O,O)", py_state_image, py_additional_ppo_inputs, py_inputs);
				if (py_action_and_stdev == NULL)
				{
					fprintf(stderr, "\nFailed to call get action function.\n");
					PyErr_Print();
					Py_DECREF(py_state_image);
					Py_DECREF(py_additional_ppo_inputs);
					Py_DECREF(py_inputs);
					return 1;
				}
				
				// Get the agent commanded action
				action_1 = PyFloat_AsDouble(PyTuple_GetItem(py_action_and_stdev, 0));
				action_2 = PyFloat_AsDouble(PyTuple_GetItem(py_action_and_stdev, 1));
				action_3 = PyFloat_AsDouble(PyTuple_GetItem(py_action_and_stdev, 2));

				// Step the environment
				done = FDS->step(action_1, action_2, action_3);
				
				// Release the python memory
				Py_DECREF(py_state_image);
				Py_DECREF(py_additional_ppo_inputs);
				Py_DECREF(py_inputs);
				Py_DECREF(py_action_and_stdev);
			}
			
			// Do a random action
			else
			{
				// Generate random actions from -1.0 to 1.0
				action_1 = 2.0*((double)rand()/(double)RAND_MAX - 0.5);
				action_2 = 2.0*((double)rand()/(double)RAND_MAX - 0.5);
				action_3 = 2.0*((double)rand()/(double)RAND_MAX - 0.5);
			}
			
			// Step the environment
			done = FDS->step(action_1, action_2, action_3);
			
		}
		
		// Step the environment 
		if (!run_agent)
		{
			// Step the environment based on the previously commanded action
			done = FDS->step(action_1, action_2, action_3);
		}
		
	}
	
	// Final progress report
	FDS->print_progress(false);
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("Simulation took: %.1f seconds.\n\nConverting simulation results...", duration);
	
	// Send all relevant data to save render and plot module
	start_time = chrono::high_resolution_clock::now();
	if(store_input_history(save_render_plot, input_location_x, input_location_y, input_percent, trigger_power, source_power) == 1) {return 1;}
	if(store_field_history(save_render_plot, temperature_field, cure_field, fine_temperature_field, fine_cure_field, fine_mesh_loc) == 1) {return 1;}
	if(store_front_history(save_render_plot, front_curve, front_fit, front_velocity, front_temperature, front_shape_param) == 1) {return 1;}
	if(store_target_and_time(save_render_plot, target, time, reward) == 1) {return 1;}
	if(store_top_mesh(save_render_plot, FDS->get_coarse_x_mesh_z0(), FDS->get_coarse_y_mesh_z0()) == 1) {return 1;}
	if(store_input_params(save_render_plot, FDS->get_peak_input_mag(), FDS->get_input_const()) == 1) {return 1;}
	if(store_options(save_render_plot, FDS->get_control_mode(), configs_string) == 1) {return 1;}
	if(store_monomer_properties(save_render_plot, FDS->get_specific_heat(), FDS->get_density(), FDS->get_adiabatic_temp_of_rxn()) == 1) {return 1;}
	if(store_domain_properties(save_render_plot, FDS->get_volume(), FDS->get_surface_area()) == 1) {return 1;}
	if(store_boundary_conditions(save_render_plot, FDS->get_heat_transfer_coefficient(), FDS->get_ambient_temperature(), FDS->get_initial_temp()) == 1) {return 1;}

	// Stop clock and print duration
	duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("\nData conversion took: %.1f seconds.", duration);

	// Save, plot, and render
	start_time = chrono::high_resolution_clock::now();
	bool render;
	solver_cfg->get_var("render", render);
	return save_results(save_render_plot, render);
	
}


int main()
{	
	// Load run solver and fds cfg files
	Config_Handler fds_cfg = Config_Handler("../config_files", "fds.cfg");
	Config_Handler* solver_cfg = new Config_Handler("../config_files", "solver.cfg");
	string configs_string = "";
	configs_string = configs_string + fds_cfg.get_orig_cfg();
	configs_string = configs_string + "\n\n======================================================================================\n\n";
	configs_string = configs_string + solver_cfg->get_orig_cfg();
	
	// Init py environment
	Py_Initialize();
	PyRun_SimpleString("import  sys");
	PyRun_SimpleString("sys.path.append('../py_src/')");
	
	// Initialize FDS
	Finite_Difference_Solver* FDS;
	try
	{
		FDS = new Finite_Difference_Solver();
	}
	catch (int e)
	{
		cout << "An exception occurred. Exception num " << e << '\n';
		cin.get();
		return 1;
	}
    
    	// Initialize front speed estimator
	Speed_Estimator* estimator = new Speed_Estimator(FDS->get_coarse_x_mesh_z0());
    
    	// Initialize agent if it is used
	string input_load_path;
	solver_cfg->get_var("input_load_path", input_load_path);
	PyObject* agent = NULL;
	if( input_load_path.compare("none")!=0 )
	{
		Config_Handler speed_estimator_cfg = Config_Handler("../config_files", "speed_estimator.cfg");
		configs_string = configs_string + "\n\n======================================================================================\n\n";
		configs_string = configs_string + speed_estimator_cfg.get_orig_cfg();
		agent = init_agent(1, 3, input_load_path);
		if (agent == NULL)
		{ 
			Py_FinalizeEx();
			cin.get();
			return 1; 
		}
	}
    
    	// Init save_render_plot
	PyObject* save_render_plot = init_save_render_plot();
	if (save_render_plot == NULL)
	{ 
		Py_FinalizeEx();
		cin.get();
		return 1; 
	}
	
	// Print parameters to stdout
	string dummy_str;
	cout << "\nAgent Hyperparameters(\n";
	cout << "  (Input Load Path): " << solver_cfg->get_var("input_load_path", dummy_str) << "\n";
	cout << "  (Steps per Trajectory): " << solver_cfg->get_var("num_input_actions", dummy_str) << "\n";
	cout << ")\n";
	FDS->print_params();

	// Train agent
	cout << "\nSimulating...\n";
	auto start_time = chrono::high_resolution_clock::now();
	if (run(FDS, solver_cfg, estimator, agent, save_render_plot, start_time, configs_string) == 1)
	{ 
		Py_FinalizeEx();
		cin.get();
		return 1; 
	}
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("Saving and Rendering Took: %.1f seconds.\n\nDone!", duration);
	
	// Finish
	Py_FinalizeEx();
	cin.get();
	return 0;
}