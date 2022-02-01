#define PY_SSIZE_T_CLEAN
#include "Config_Handler.hpp"
#include "python_interface.hpp"
#include "Speed_Estimator.hpp"
#include "Finite_Difference_Solver.hpp"

/**
* Prints to stdout a readout of the current RL agent training process
* @param the current trajecotry being simulated
* @param the total number of trajectories to be simulated
* @param the total reward given for the previous trajectory
* @param number of RL agent actions taken per trajectory 
* @param vector containing episodic reward memory
* @param the best episodic reward to date
*/
void print_training_info(int curr_trajectory, int total_trajectories, double prev_episode_reward, int steps_per_trajectory, vector<double> r_per_episode, double best_episode)
{
	// Percent complete sub messege
	int percent_complete = total_trajectories>1 ? 100.0 * curr_trajectory / (total_trajectories-1) : 0.0;
	percent_complete = curr_trajectory == total_trajectories ? 100.0 : percent_complete;
	stringstream stream;
	stream << fixed << setprecision(1) << percent_complete;
	string msg1 = stream.str();
	msg1.append("% Complete");
	if (msg1.length() < 16)
	{
		msg1.append(16 - msg1.length(), ' ');
	}
	
	// Trajectory count sub messege
	string msg2 = "";
	msg2.append("| Traj: " + to_string(curr_trajectory+1) + "/" + to_string(total_trajectories));
	if (msg2.length() < 22)
	{
		msg2.append(22 - msg2.length(), ' ');
	}
	
	// Previous reward per step sub messege
	string msg3 = "| Curr R: ";
	stream.str(string());
	stream << fixed << setprecision(2) << prev_episode_reward/(double)steps_per_trajectory;
	msg3.append(stream.str());
	if (msg3.length() < 16)
	{
		msg3.append(16 - msg3.length(), ' ');
	}
	
	// Rolling average reward per step sub messege
	string msg4 = "| Avg R: ";
	if (r_per_episode.empty())
	{
		stream.str(string());
		stream << fixed << setprecision(2) << 0.0;
		msg4.append(stream.str());
	}
	else
	{
		int start_index = (int) r_per_episode.size() - 100;
		start_index = start_index < 0 ? 0 : start_index;
		double avg_r_per_episode = 0.0;
		for (int i = start_index; i < (int)r_per_episode.size(); i++)
		{
			avg_r_per_episode += r_per_episode[i];
		}
		avg_r_per_episode = avg_r_per_episode / (r_per_episode.size() - start_index);
		stream.str(string());
		stream << fixed << setprecision(2) << avg_r_per_episode;
		msg4.append(stream.str());
	}
	if (msg4.length() < 15)
	{
		msg4.append(15 - msg4.length(), ' ');
	}

	// Best recorded reward per simulation step sub messege
	string msg5 = "| Best R: ";
	stream.str(string());
	stream << fixed << setprecision(2) << best_episode/(double)steps_per_trajectory;
	msg5.append(stream.str());
	if (msg5.length() < 16)
	{
		msg5.append(16 - msg5.length(), ' ');
	}
	
	// Print all sub messeges
	cout << msg1+msg2+msg3+msg4+msg5 << "|\r";
}


/**
* Runs a set of trajectories using the PPO policy, updates the PPO agent, and collects relevant training data
* @param The finite element solver object used to propogate time
* @param Configuration handler object that contains all loaded configuration data
* @param The class used to estimate front speed based on front location estimations
* @param The ppo agent being trained
* @param The save render and plotting class of the ppo agent being trained
* @param The time at which simulation was started
 *@param String containing cfg data
* @return 0 on success, 1 on failure
*/
int run(Finite_Difference_Solver* FDS, Config_Handler* train_agent_cfg, Speed_Estimator* estimator, PyObject* agent, PyObject* save_render_plot, auto &start_time, string configs_string)
{
	// Get values from config file
	int total_trajectories;
	train_agent_cfg->get_var("total_trajectories",total_trajectories);
	int steps_per_trajectory;
	train_agent_cfg->get_var("steps_per_trajectory",steps_per_trajectory);
	int trajectories_per_batch;
	train_agent_cfg->get_var("trajectories_per_batch",trajectories_per_batch);
	double frame_rate;
	train_agent_cfg->get_var("frame_rate",frame_rate);
	
	// Frame rate calculations
	int steps_per_agent_cycle = (int) round((FDS->get_sim_duration() / (double)steps_per_trajectory) / FDS->get_coarse_time_step());
	steps_per_agent_cycle = steps_per_agent_cycle <= 0 ? 1 : steps_per_agent_cycle;	
	int steps_per_speed_estimator_frame = (int) round(estimator->get_observation_delta_t() / FDS->get_coarse_time_step());
	steps_per_speed_estimator_frame = steps_per_speed_estimator_frame <= 0 ? 1 : steps_per_speed_estimator_frame;	
	int steps_per_render_frame = (int) round(1.0 / (FDS->get_coarse_time_step() * frame_rate));
	steps_per_render_frame = steps_per_render_frame <= 0 ? 1 : steps_per_render_frame;
	
	// Agent training data storage
	vector<double> r_per_episode;
	vector<double> critic_loss;
	vector<double> actor_lr;
	vector<double> critic_lr;
	vector<double> x_stdev;
	vector<double> y_stdev;
	vector<double> mag_stdev;

	// Simulation set variables
	double total_reward = 0.0;
	double best_episode_reward = 0.0;
	double prev_episode_reward = 0.0;
	
	// Run a set of episodes
	for (int i = 0; i < total_trajectories; i++)
	{
		// Initialize simulation variables
		bool done = false;
		double action_1=0.0, stdev_1=0.0, action_2=0.0, stdev_2=0.0, action_3=0.0, stdev_3=0.0, reward;
		int step_in_trajectory = 0;
		
		// User readout
		print_training_info(i, total_trajectories, prev_episode_reward, steps_per_trajectory, r_per_episode, best_episode_reward);
		prev_episode_reward = total_reward;

		// Reset
		FDS->reset();
		estimator->reset();
		
		// Simulation loop
		while (!done)
		{
			// Determine what to run this simulation step
			bool run_agent = (step_in_trajectory % steps_per_agent_cycle == 0) || (step_in_trajectory==0);
 			bool observe_speed = (step_in_trajectory % steps_per_speed_estimator_frame == 0) || (step_in_trajectory==0);
			step_in_trajectory++;
			
			// Add observation to speed estimator
			if (observe_speed)
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
			
			// Run the agent
			if (run_agent)
			{
				// Gather temperature state data
				PyObject* py_state_image = get_2D_list<vector<vector<double>>>(FDS->get_coarse_temp_z0(true));
				
				// Get speed from speed estimator and calculate error
				double front_speed = estimator->estimate();
				double front_speed_error = front_speed / FDS->get_curr_target();
				
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
				
				// Get the agent's stdev
				stdev_1 = PyFloat_AsDouble(PyTuple_GetItem(py_action_and_stdev, 3));
				stdev_2 = PyFloat_AsDouble(PyTuple_GetItem(py_action_and_stdev, 4));
				stdev_3 = PyFloat_AsDouble(PyTuple_GetItem(py_action_and_stdev, 5));

				// Step the environment
				done = FDS->step(action_1, action_2, action_3);
				
				// Combine the action data
				vector<double> actions = vector<double>(3,0.0);
				actions[0] = action_1;
				actions[1] = action_2;
				actions[2] = action_3;
				PyObject* py_actions = get_1D_list<vector<double>>(actions);

				// Update the agent
				vector<double> reward_arr = FDS->get_reward();
				reward = reward_arr[0];
				PyObject* py_critic_loss_and_lr = PyObject_CallMethod(agent, "update_agent", "(O,O,O,O,f)", py_state_image, py_additional_ppo_inputs, py_inputs, py_actions, reward);
				if (py_critic_loss_and_lr == NULL)
				{
					fprintf(stderr, "\nFailed to update agent\n");
					PyErr_Print();
					Py_DECREF(py_state_image);
					Py_DECREF(py_additional_ppo_inputs);
					Py_DECREF(py_inputs);
					Py_DECREF(py_action_and_stdev);
					Py_DECREF(py_actions);
					return 1;
				}
				
				// Collect critic loss data
				vector<double> curr_critic_loss_and_lr = get_1D_vector(py_critic_loss_and_lr);
				if (curr_critic_loss_and_lr.size() > 0)
				{
					actor_lr.push_back(curr_critic_loss_and_lr[0]);
					critic_lr.push_back(curr_critic_loss_and_lr[1]);
					for (unsigned int i = 2; i < curr_critic_loss_and_lr.size(); i++)
					{
						critic_loss.push_back(curr_critic_loss_and_lr[i]);
					}
				}
				
				// Update reward
				total_reward = total_reward + reward;
				
				// Release the python memory
				Py_DECREF(py_state_image);
				Py_DECREF(py_additional_ppo_inputs);
				Py_DECREF(py_inputs);
				Py_DECREF(py_action_and_stdev);
				Py_DECREF(py_actions);
				Py_DECREF(py_critic_loss_and_lr);
			}
			
			// Step the environment 
			if (!run_agent)
			{
				// Step the environment based on the previously commanded action
				done = FDS->step(action_1, action_2, action_3);
			}
			
		}

		// Update the best trajectory memory
		prev_episode_reward = total_reward - prev_episode_reward;
		if (prev_episode_reward > best_episode_reward)
		{
			best_episode_reward = prev_episode_reward;
		}

		// Store actor training data
		r_per_episode.push_back(prev_episode_reward/(double)steps_per_trajectory);
		x_stdev.push_back(0.5*FDS->get_max_input_slew_speed()*stdev_1);
		y_stdev.push_back(0.5*FDS->get_max_input_slew_speed()*stdev_2);		
		mag_stdev.push_back(FDS->get_peak_input_mag()*stdev_3);

		// Final user readout
		if (i == total_trajectories - 1) { print_training_info(i, total_trajectories, prev_episode_reward, steps_per_trajectory, r_per_episode, best_episode_reward); }
	}
	
	// Trajectory memory
	vector<double> input_location_x;
	vector<double> input_location_y;
	vector<double> input_percent;
	vector<double> power;
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
		
	// Run one last trajectory and save the results
	while (!done)
	{
		// Determine what to run this simulation step
		bool run_agent = (step_in_trajectory % steps_per_agent_cycle == 0) || (step_in_trajectory==0);
		bool observe_speed = (step_in_trajectory % steps_per_speed_estimator_frame == 0) || (step_in_trajectory==0);
		bool save_frame = (step_in_trajectory % steps_per_render_frame == 0) || (step_in_trajectory==0);
		step_in_trajectory++;
		
		// Add observation to speed estimator
		if (observe_speed)
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
			power.push_back(FDS->get_power());
			
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
			// Gather temperature state data
			PyObject* py_state_image = get_2D_list<vector<vector<double>>>(FDS->get_coarse_temp_z0(true));
			
			// Get speed from speed estimator and calculate error
			double front_speed = estimator->estimate();
			double front_speed_error = front_speed / FDS->get_curr_target();
			
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
		
		// Step the environment 
		if (!run_agent)
		{
			// Step the environment based on the previously commanded action
			done = FDS->step(action_1, action_2, action_3);
		}
		
	}
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("Simulation took: %.1f seconds.\n\nConverting simulation results...", duration);
	
	// Send all relevant data to save render and plot module
	start_time = chrono::high_resolution_clock::now();
	if(store_training_curves(save_render_plot, r_per_episode, critic_loss) == 1) {return 1;}
	if(store_lr_curves(save_render_plot, actor_lr, critic_lr) == 1) {return 1;}
	if(store_stdev_history(save_render_plot, x_stdev, y_stdev, mag_stdev) == 1) {return 1;}
	if(store_input_history(save_render_plot, input_location_x, input_location_y, input_percent, power) == 1) {return 1;}
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
	return save_agent_results(save_render_plot, agent);
}


int main()
{	
	// Load run solver and fds cfg files
	Config_Handler fds_cfg = Config_Handler("../config_files", "fds.cfg");
	Config_Handler* train_agent_cfg = new Config_Handler("../config_files", "trainer.cfg");
	Config_Handler speed_estimator_cfg = Config_Handler("../config_files", "speed_estimator.cfg");
	string configs_string = "";
	configs_string = configs_string + fds_cfg.get_orig_cfg();
	configs_string = configs_string + "\n\n======================================================================================\n\n";
	configs_string = configs_string + train_agent_cfg->get_orig_cfg();
	configs_string = configs_string + "\n\n======================================================================================\n\n";
	configs_string = configs_string + speed_estimator_cfg.get_orig_cfg();
	
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
    
	// Init agent
	PyObject* agent = init_agent(1, 3);
	if (agent == NULL)
	{ 
		Py_FinalizeEx();
		cin.get();
		return 1; 
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
	cout << "  (Network Load Path): " << train_agent_cfg->get_var("network_load_path", dummy_str) << "\n";
	cout << "  (Steps per Trajectory): " << train_agent_cfg->get_var("steps_per_trajectory", dummy_str) << "\n";
	cout << "  (Trajectories per Batch): " << train_agent_cfg->get_var("trajectories_per_batch", dummy_str) << "\n";
	cout << "  (Epochs per Batch): " << train_agent_cfg->get_var("epochs", dummy_str) << "\n";
	cout << "  (Discount Ratio): " << train_agent_cfg->get_var("gamma", dummy_str) << " \n";
	cout << "  (GAE Parameter): " << train_agent_cfg->get_var("lambda", dummy_str) << " \n";
	cout << "  (Clipping Parameter): " << train_agent_cfg->get_var("epsilon", dummy_str) << " \n";
	cout << "  (Start LR): " << train_agent_cfg->get_var("start_alpha", dummy_str) << "\n";
	cout << "  (End LR): " << train_agent_cfg->get_var("end_alpha", dummy_str) << " \n";
	cout << ")\n";
	FDS->print_params();

	// Train agent
	cout << "\nSimulating...\n";
	auto start_time = chrono::high_resolution_clock::now();
	if (run(FDS, train_agent_cfg, estimator, agent, save_render_plot, start_time, configs_string) == 1)
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