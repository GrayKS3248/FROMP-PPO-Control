#include "Finite_Difference_Solver6.h"

using namespace std;


//******************************************************************** CONFIGURATION FUNCTIONS ********************************************************************//
/**
* Loads parameters from .cfg file
* @return 0 on success, 1 on failure
*/
int load_config(int& total_trajectories, int& samples_per_trajectory, int& samples_per_batch, int& actions_per_trajectory, string& path)
{
	// Load from config file
	ifstream config_file;
	config_file.open("../config_files/get_ae_data6.cfg");
	string config_dump;
	if (config_file.is_open())
	{
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> total_trajectories;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> samples_per_trajectory;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> samples_per_batch;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> actions_per_trajectory;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> path;
	}
	else
	{
		cout << "Unable to open ../config_files/get_ae_data6.cfg." << endl;
		return 1;
	}
	config_file.close();
	return 0;
}

/**
* Saves copy of original fds and get_ae_data config to save path
* @param Path to which settings file will be saved
* @return 0 on success, 1 on failure
*/
int save_config(string path)
{
	// Reset original string
	string configs_string = "";
	
	// Read lines from fds config file
	ifstream file_in;
	file_in.open("../config_files/fds6.cfg", ofstream::in);
	string read_line;
	if (file_in.is_open())
	{
		// Copy lines to original string
		while( getline(file_in, read_line) )
		{
			configs_string = configs_string + read_line + "\n";
		}
		configs_string.pop_back();
	}
	else
	{
		cout << "Unable to open ../config_files/fds6.cfg." << endl;
		return 1;
	}
	
	// Close the file
	file_in.close();
	
	// Pad between config files
	configs_string = configs_string + "\n\n======================================================================================\n\n";
	
	// Read lines from train_agent config file
	file_in.open("../config_files/get_ae_data6.cfg", ofstream::in);
	if (file_in.is_open())
	{
		// Copy lines to original string
		while( getline(file_in, read_line) )
		{
			configs_string = configs_string + read_line + "\n";
		}
		configs_string.pop_back();
	}
	else
	{
		cout << "Unable to open ../config_files/get_ae_data6.cfg." << endl;
		return 1;
	}
	
	// Close the file
	file_in.close();
	
	// Write to settings file
	ofstream write_file;
	string write_file_name = path+"/__settings__.dat";
	write_file.open(write_file_name, ofstream::trunc);
	if(!write_file.is_open())
	{
		cout << "\nFailed to open settings.dat\n";
		return 1;
	}
	write_file << configs_string;
	write_file.close();
		
	return 0;
}


//******************************************************************** USER INTERFACE FUNCTIONS ********************************************************************//
/**
* Prints the finite element solver and simulation parameters to std out
* @param total number of trajectories taken
* @param number of autoencoder frames saved per trajecotry
* @param number of actions taken by random controller during one trajectory
*/
void print_params(int total_trajectories, int samples_per_trajectory, int actions_per_trajectory)
{
	// Hyperparameters
	cout << "\nSimulation Parameters(\n";
	cout << "  (Number of Trajectories): " << total_trajectories << "\n";
	cout << "  (Samples per Trajectory): " << samples_per_trajectory << "\n";
	cout << "  (Actions per Trajectory): " << actions_per_trajectory << "\n";
	cout << ")\n";
}

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
* @param The total number of trajectories to be executed
* @param The number of simulation steps taken per single cycle of control application
* @param Number of trajectories to be simulated
* @param Number of training data sampled during on trajecotry
* @param Number of frames per training epoch
* @param Path to save training data to
* @return 0 on success, 1 on failure
*/
int run(Finite_Difference_Solver* FDS, int total_trajectories, int steps_per_control_cycle, int tot_num_sim_steps, int samples_per_trajectory, int samples_per_batch, string path)
{
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
		
		// Open file to save front mean x location to
		stream.str(string());
		stream << path << "/loc_data_" << curr_epoch << ".csv";
		string loc_load_name = stream.str();
		ofstream loc_file;
		loc_file.open (loc_load_name, ofstream::trunc);
		if(!loc_file.is_open())
		{
			cout << "\nFailed to open loc file\n";
			return 1;
		}
		
		// Open file to save front temperature tp
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
		
		// Run a set of episodes
		for (int curr_traj = 0; curr_traj < (int)floor((double)samples_per_batch/(double)samples_per_trajectory); curr_traj++)
		{
			// Declare simulation variables
			bool done = false;
			double action_1=0.0, action_2=0.0, action_3=0.0;
			bool apply_control, save_frame;
			int trajectory_index = 0;
			
			// Select random set of frames to be used to update autoencoder
			vector<int> frame_indices = get_frame_indices(tot_num_sim_steps, samples_per_trajectory);
			int frame_count = 0;
			int frame_index = frame_indices[frame_count];

			// User readout
			print_training_info(curr_traj+curr_epoch*(int)floor((double)samples_per_batch/(double)samples_per_trajectory), total_trajectories);

			// Save data structure
			vector<vector<vector<double>>> temp_frames;
			vector<vector<vector<double>>> cure_frames;
			vector<double> loc_frames;
			vector<double> ftemp_frames;

			// Reset environment
			FDS->reset();
			
			// Simulation for loop
			while (!done)
			{
				// Determine what to run this simulation step
				apply_control = (trajectory_index % steps_per_control_cycle == 0) || (trajectory_index==0);
				save_frame = trajectory_index == frame_index;

				// Run the random controller
				if (apply_control)
				{

					// Get a random action
					action_1 = (2.0 * ((double)rand()/(double)RAND_MAX - 0.5));
					action_2 = (2.0 * ((double)rand()/(double)RAND_MAX - 0.5));
					action_3 = (2.0 * ((double)rand()/(double)RAND_MAX - 0.5));

					// Step the environment
					done = FDS->step(action_1, action_2, true, action_3);
				}
				else
				{
					// Step the environment
					done = FDS->step(action_1, action_2, true, action_3);
				}
				
				// Save data to files
				if (save_frame)
				{
					temp_frames.push_back(FDS->get_coarse_temp_z0(true));
					cure_frames.push_back(FDS->get_coarse_cure_z0());
					loc_frames.push_back(FDS->get_front_mean_x_loc(true));
					ftemp_frames.push_back(FDS->get_front_temp(true));
			
					// Update which frame is to be saved next
					frame_count++;
					if (frame_count < samples_per_trajectory)
					{
						frame_index = frame_indices[frame_count];
					}
				}

				// Update the current state and the step in episode
				trajectory_index++;

			}
			
			// Save current trajectory's frames
			for (int i = 0; i < samples_per_trajectory; i++)
			{
				loc_file << loc_frames[i] << "\n";
				ftemp_file << ftemp_frames[i] << "\n";
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
		
	}
	return 0;
}

int main()
{	
	// Training data parameters
	int total_trajectories;
	int samples_per_trajectory;
	int samples_per_batch;
	int actions_per_trajectory;
	string path;
	if (load_config(total_trajectories, samples_per_trajectory, samples_per_batch, actions_per_trajectory, path) == 1) 
	{ 
		cin.get();
		return 1; 
	}

	// Save settings
	if (save_config(path) == 1) 
	{ 
		cin.get();
		return 1; 
	}

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

	// Calculated parameters
	double control_execution_period = (FDS->get_sim_duration() / ((double)actions_per_trajectory));
	int steps_per_control_cycle = (int) round(control_execution_period / FDS->get_coarse_time_step());
	int tot_num_sim_steps = FDS->get_num_sim_steps();
	
	// Print simulation parameters
	print_params(total_trajectories, samples_per_trajectory, actions_per_trajectory);
	FDS->print_params();

	// Train autoencoder
	cout << "\nCollecting training data...\n";
	auto start_time = chrono::high_resolution_clock::now();
	if (run(FDS, total_trajectories, steps_per_control_cycle, tot_num_sim_steps, samples_per_trajectory, samples_per_batch, path) == 1) 
	{ 
		cin.get();
		return 1; 
	}

	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("\nCollection took: %.1f seconds.\n", duration);

	// Finish
	cout << "Done!";
	return 0;
}
