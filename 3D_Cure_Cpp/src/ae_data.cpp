#define PY_SSIZE_T_CLEAN
#include "Config_Handler.hpp"
#include "python_interface.hpp"
#include "Speed_Estimator.hpp"
#include "Finite_Difference_Solver.hpp"

/**
* Prints to stdout a readout of the current RL agent training process
* @param the current trajecotry being simulated
* @param the total number of trajectories to be simulated
*/
void print_training_info(int curr_trajectory, int total_trajectories)
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
	cout << msg1;
	
	// Trajectory count sub messege
	string msg2 = "";
	msg2.append("| Traj: " + to_string(curr_trajectory+1) + "/" + to_string(total_trajectories));
	if (msg2.length() < 22)
	{
		msg2.append(22 - msg2.length(), ' ');
	}
	cout << msg2;
	
	// Print return carriage
	cout << "|\r";
}

/**
* Generates random set of frames to be used to update autoencoder
* @param total number of frames generated by 1 trajectory
* @param Number of frames taken each trajectory
* @return vector of update frame indices
*/
vector<int> get_frame_indices(int tot_num_sim_steps, int samples_per_trajectory)
{
	// Create list of all frames
	vector<int> all_frames;
	for (int i = 0; i < tot_num_sim_steps; i++)
	{
		all_frames.push_back(i);
	}
	
	// Randomly shuffle list of all frames
	random_shuffle (all_frames.begin(), all_frames.end());
	
	// Populate set of update frames from shuffled list of all frames
	vector<int> update_frames;
	for (int i = 0; i < samples_per_trajectory; i++)
	{
		update_frames.push_back(all_frames[i]);
	}
	
	// Sort list of update frames
	sort (update_frames.begin(), update_frames.end());
	return update_frames;
}

/**
* Runs a set of trajectories using a random policy and simultaneously trains and autoencoder to create a reduced state representation
* @param The finite element solver object used to propogate time
* @param Configuration handler object that contains all loaded configuration data
* @param speed estimator used by agent if any
* @param agent if any 
* @return 0 on success, 1 on failure
*/
int run(Finite_Difference_Solver* FDS, Config_Handler* ae_data_cfg, Speed_Estimator* estimator, PyObject* agent)
{
	int total_trajectories;
	ae_data_cfg->get_var("total_trajectories",total_trajectories);
	int samples_per_trajectory;
	ae_data_cfg->get_var("samples_per_trajectory",samples_per_trajectory);
	int samples_per_batch;
	ae_data_cfg->get_var("samples_per_batch",samples_per_batch);
	int num_input_actions;
	ae_data_cfg->get_var("num_input_actions",num_input_actions);
	string path;
	ae_data_cfg->get_var("save_path",path);
	
	// Frame rate calcualtions
	int steps_per_agent_cycle = (int) round((FDS->get_sim_duration() / (double)num_input_actions) / FDS->get_coarse_time_step());
	steps_per_agent_cycle = steps_per_agent_cycle <= 0 ? 1 : steps_per_agent_cycle;	
	int steps_per_speed_estimator_frame = (int) round(estimator->get_observation_delta_t() / FDS->get_coarse_time_step());
	steps_per_speed_estimator_frame = steps_per_speed_estimator_frame <= 0 ? 1 : steps_per_speed_estimator_frame;	
	
	for (int curr_epoch = 0; curr_epoch < (int)floor(((double)total_trajectories*(double)samples_per_trajectory)/(double)samples_per_batch); curr_epoch++)
	{
		// Open string buffer to name data files
		stringstream stream;
	
		// Open file to save temp to
		stream << path << "/temp_data_" << curr_epoch << ".csv";
		string temp_load_name = stream.str();
		ofstream temp_file;
		temp_file.open(temp_load_name, ofstream::trunc);
		if(!temp_file.is_open())
		{
			cout << "\nFailed to open temp file\n";
			return 1;
		}
		
		// Open file to save cure to
		stream.str(string());
		stream << path << "/cure_data_" << curr_epoch << ".csv";
		string cure_load_name = stream.str();
		ofstream cure_file;
		cure_file.open (cure_load_name, ofstream::trunc);
		if(!cure_file.is_open())
		{
			cout << "\nFailed to open cure file\n";
			return 1;
		}
		
		// Open file to save front temp to
		stream.str(string());
		stream << path << "/ftemp_data_" << curr_epoch << ".csv";
		string ftemp_load_name = stream.str();
		ofstream ftemp_file;
		ftemp_file.open (ftemp_load_name, ofstream::trunc);
		if(!ftemp_file.is_open())
		{
			cout << "\nFailed to open ftemp file\n";
			return 1;
		}
		
		// Open file to save front location to
		stream.str(string());
		stream << path << "/floc_data_" << curr_epoch << ".csv";
		string floc_load_name = stream.str();
		ofstream floc_file;
		floc_file.open (floc_load_name, ofstream::trunc);
		if(!floc_file.is_open())
		{
			cout << "\nFailed to open floc file\n";
			return 1;
		}
		
		// Open file to save front speed to
		stream.str(string());
		stream << path << "/fspeed_data_" << curr_epoch << ".csv";
		string fspeed_load_name = stream.str();
		ofstream fspeed_file;
		fspeed_file.open (fspeed_load_name, ofstream::trunc);
		if(!fspeed_file.is_open())
		{
			cout << "\nFailed to open fspeed file\n";
			return 1;
		}
		
		// Open file to save front shape to
		stream.str(string());
		stream << path << "/fshape_data_" << curr_epoch << ".csv";
		string fshape_load_name = stream.str();
		ofstream fshape_file;
		fshape_file.open (fshape_load_name, ofstream::trunc);
		if(!fshape_file.is_open())
		{
			cout << "\nFailed to open fshape file\n";
			return 1;
		}
		
		// Run a set of episodes
		for (int curr_traj = 0; curr_traj < (int)floor((double)samples_per_batch/(double)samples_per_trajectory); curr_traj++)
		{
			// Declare simulation variables
			bool done = false;
			double action_1=0.0, action_2=0.0, action_3=0.0;
			int step_in_trajectory = 0;
			
			// Select random set of frames to be used to update autoencoder
			vector<int> frame_indices = get_frame_indices(FDS->get_num_sim_steps(), samples_per_trajectory);
			int frame_count = 0;
			int frame_index = frame_indices[frame_count];

			// User readout
			print_training_info(curr_traj+curr_epoch*(int)floor((double)samples_per_batch/(double)samples_per_trajectory), total_trajectories);
			
			// Save data structure
			vector<vector<vector<double>>> temp_frames;
			vector<vector<vector<double>>> cure_frames;
			vector<double> ftemp_frames;
			vector<double> floc_frames;
			vector<double> fspeed_frames;
			vector<double> fshape_frames;

			// Reset environment
			FDS->reset();
			estimator->reset();
			
			// Simulation for loop
			while (!done)
			{
				// Determine what to run this simulation step
				bool run_agent = (step_in_trajectory % steps_per_agent_cycle == 0) || (step_in_trajectory==0);
				bool observe_speed = (step_in_trajectory % steps_per_speed_estimator_frame == 0) || (step_in_trajectory==0);
				bool save_frame = step_in_trajectory == frame_index;
				step_in_trajectory++;
				
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
				
				// Save data to files
				if (save_frame)
				{
					temp_frames.push_back(FDS->get_coarse_temp_z0(true));
					cure_frames.push_back(FDS->get_coarse_cure_z0());
					ftemp_frames.push_back(FDS->get_front_temp(true));
					floc_frames.push_back(FDS->get_front_mean_x_loc(true));
					fspeed_frames.push_back(FDS->get_front_vel(true));
					fshape_frames.push_back(FDS->get_front_shape_param());
					
					
					// Update which frame is to be saved next
					frame_count++;
					if (frame_count < samples_per_trajectory)
					{
						frame_index = frame_indices[frame_count];
					}
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
					
				}
				
				// Step the environment 
				done = FDS->step(action_1, action_2, action_3);

			}
			
			// Save current trajectory's frames
			for (int i = 0; i < samples_per_trajectory; i++)
			{
				ftemp_file << ftemp_frames[i] << "\n";
				floc_file << floc_frames[i] << "\n";
				fspeed_file << fspeed_frames[i] << "\n";
				fshape_file << fshape_frames[i] << "\n";
					
				for (int j = 0; j < FDS->get_num_coarse_vert_x(); j++)
				{
					for(int k = 0; k < FDS->get_num_coarse_vert_y(); k++)
					{
						if (k == FDS->get_num_coarse_vert_y()-1)
						{
							temp_file << temp_frames[i][j][k] << "\n";
							cure_file << cure_frames[i][j][k] << "\n";
						}
						else
						{
							temp_file << temp_frames[i][j][k] << ",";
							cure_file << cure_frames[i][j][k] << ",";
						}
					}
				}
			}

			// Final user readout
			if (curr_traj+curr_epoch*(int)floor((double)samples_per_batch/(double)samples_per_trajectory) == total_trajectories - 1) { print_training_info(curr_traj+curr_epoch*(int)floor((double)samples_per_batch/(double)samples_per_trajectory), total_trajectories); }
		}

		if (temp_file.is_open()) { temp_file.close(); }
		if (cure_file.is_open()) { cure_file.close(); }
		if (ftemp_file.is_open()) { ftemp_file.close(); }
		if (floc_file.is_open()) { floc_file.close(); }
		if (fspeed_file.is_open()) { fspeed_file.close(); }
		if (fshape_file.is_open()) { fshape_file.close(); }
		
	}
	return 0;
}


int main()
{		
	// Load run solver and fds cfg files
	Config_Handler fds_cfg = Config_Handler("../config_files", "fds.cfg");
	Config_Handler* ae_data_cfg = new Config_Handler("../config_files", "ae_data.cfg");
	string configs_string = "";
	configs_string = configs_string + fds_cfg.get_orig_cfg();
	configs_string = configs_string + "\n\n======================================================================================\n\n";
	configs_string = configs_string + ae_data_cfg->get_orig_cfg();
		
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
	string load_path;
	ae_data_cfg->get_var("load_path", load_path);
	PyObject* agent = NULL;
	if( load_path.compare("none")!=0 )
	{
		Config_Handler speed_estimator_cfg = Config_Handler("../config_files", "speed_estimator.cfg");
		configs_string = configs_string + "\n\n======================================================================================\n\n";
		configs_string = configs_string + speed_estimator_cfg.get_orig_cfg();
		agent = init_agent(1, 3, load_path);
		if (agent == NULL)
		{ 
			Py_FinalizeEx();
			cin.get();
			return 1; 
		}
	}
	
	// Initialize cfg stuff
	string dummy_str;
	cout << "Get Data Parameters(\n";
	cout << "  (Total Trajectories): " << ae_data_cfg->get_var("total_trajectories", dummy_str) << "\n";
	cout << "  (Samples per Trajectory): " << ae_data_cfg->get_var("samples_per_trajectory", dummy_str) << "\n";
	cout << "  (Samples per Batch): " << ae_data_cfg->get_var("samples_per_batch", dummy_str) << "\n";
	cout << "  (Actions per Trajectory): " << ae_data_cfg->get_var("num_input_actions", dummy_str) << "\n";
	cout << "  (Load path): " << ae_data_cfg->get_var("load_path", dummy_str) << "\n";
	cout << "  (Save path): " << ae_data_cfg->get_var("save_path", dummy_str) << "\n";
	cout << ")\n";
	FDS->print_params();

	// Generate data
	cout << "\nSimulating...\n";
	auto start_time = chrono::high_resolution_clock::now();
	if (run(FDS, ae_data_cfg, estimator, agent) == 1)
	{ 
		cin.get();
		return 1; 
	}
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("\nSimulation took: %.1f seconds.\n\nDone!", duration);
	
	// Save the settings
	string save_path;
	ae_data_cfg->get_var("save_path",save_path);
	ofstream file;
	file.open(save_path+"/settings.dat", ofstream::trunc);
	if(!file.is_open())
	{
		cout << "\nFailed to open settings.dat file\n";
		return 1;
	}
	file << configs_string;
	file.close();
	
	// Finish
	cin.get();
	return 0;
}