#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "Finite_Difference_Solver.h"

using namespace std;


//******************************************************************** CONFIGURATION HANDLER CLASS ********************************************************************//
/**
* Config Handler class loads and stores data from cfg file
**/
class Config_Handler
{
	public:
		// public constructor and destructor
		Config_Handler();
		
		// public functions
		int save_config();
		int calculate_parameters();
		
		// configuration string
		string configs_string;
		
		// config loaded variables
		string load_path;
		int total_trajectories;
		int steps_per_trajectory;
		int trajectories_per_batch;
		int epochs_per_batch;
		bool reset_std;
		bool input_rates;
		double gamma;
		double lambda;
		double epsilon;
		double start_alpha;
		double end_alpha;
		int sequence_length;
		double observation_delta_t;
		double filter_time_const;
		double frame_rate;
			
		// FDS config loaded variables
		double sim_duration;
		double sim_time_step;
			
		// Calculated variables
		double decay_rate;
		int steps_per_agent_cycle;
		int steps_per_render_frame;
		int steps_per_speed_estimator_frame;
};


/**
* Default constructor loads and stores data from ../config_files/train_agent.cfg
*/
Config_Handler::Config_Handler()
{
	// Load from train_agent config file
	ifstream config_file;
	config_file.open("../config_files/train_agent.cfg");
	string config_dump;
	string string_dump;
	if (config_file.is_open())
	{
		// {Load path}
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> load_path;	
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		// {Agent training parameters}
		config_file >> config_dump >> total_trajectories;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> steps_per_trajectory;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> trajectories_per_batch;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> epochs_per_batch;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		// {Agent hyperparameters}
		config_file >> config_dump >> string_dump;
		if (string_dump.compare("true")==0)
		{
			reset_std = true;
		}
		else if (string_dump.compare("false")==0)
		{
			reset_std = false;
		}
		else
		{
			cout << "\nReset stdev configuration not recognized.";
			throw 7;
		}
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> string_dump;
		if (string_dump.compare("pos")==0)
		{
			input_rates = false;
		}
		else if (string_dump.compare("rate")==0)
		{
			input_rates = true;
		}
		else
		{
			cout << "\nInput type configuration not recognized.";
			throw 6;
		}
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> gamma;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> lambda;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> epsilon;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> start_alpha;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> end_alpha;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		// {Speed estimator hyperparameters}
		config_file >> config_dump >> sequence_length;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> observation_delta_t;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> filter_time_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		
		// {Rendering parameters}
		config_file >> config_dump >> frame_rate;
	}
	else
	{
		cout << "Unable to open ../config_files/train_agent.cfg." << endl;
		throw 5;
	}
	config_file.close();
	
	// Load from FDS config file
	config_file.open("../config_files/fds.cfg");
	if (config_file.is_open())
	{
		// Sim duration
		for (int i = 0; i < 8; i++ )
		{
			config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		}
		config_file >> config_dump >> sim_duration;
		
		// Sim time step
		for (int i = 0; i < 38; i++ )
		{
			config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		}
		config_file >> config_dump >> sim_time_step;
	}
	else
	{
		cout << "Unable to open ../config_files/fds.cfg." << endl;
		throw 8;
	}
	config_file.close();
	
	// Save configuration string
	if(save_config()==1)
	{
		throw 9;
	}
	
	// Calculate parameters
	if(calculate_parameters()==1)
	{
		throw 10;
	}
}

/**
* Saves copy of original fds and train_agent config to string
* @param String to which config is saved
* @return 0 on success, 1 on failure
*/
int Config_Handler::save_config()
{
	// Reset original string
	configs_string = "";
	
	// Read lines from fds config file
	ifstream file_in;
	file_in.open("../config_files/fds.cfg", ofstream::in);
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
		cout << "Unable to open ../config_files/fds.cfg." << endl;
		return 1;
	}
	
	// Close the file
	file_in.close();
	
	// Pad between config files
	configs_string = configs_string + "\n\n======================================================================================\n\n";
	
	// Read lines from train_agent config file
	file_in.open("../config_files/train_agent.cfg", ofstream::in);
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
		cout << "Unable to open ../config_files/train_agent.cfg." << endl;
		return 1;
	}
	
	// Close the file
	file_in.close();
	
	return 0;
}

int Config_Handler::calculate_parameters()
{
	// Calculated agent parameters
	decay_rate = pow(end_alpha/start_alpha, (double)trajectories_per_batch/(double)total_trajectories);
	steps_per_agent_cycle = (int) round((sim_duration / (double)steps_per_trajectory) / sim_time_step);
	steps_per_agent_cycle = steps_per_agent_cycle <= 0 ? 1 : steps_per_agent_cycle;	

	// Calculated speed estimator parameters
	steps_per_speed_estimator_frame = (int) round(observation_delta_t / sim_time_step);
	steps_per_speed_estimator_frame = steps_per_speed_estimator_frame <= 0 ? 1 : steps_per_speed_estimator_frame;	

	// Calculated rendering parameters
	steps_per_render_frame = (int) round(1.0 / (sim_time_step * frame_rate));
	steps_per_render_frame = steps_per_render_frame <= 0 ? 1 : steps_per_render_frame;
	
	return 0;
}


//******************************************************************** SPEED ESTIMATOR CLASS ********************************************************************//
/**
* Speed estimator class takes observations of front locations and times and converts these to front speed estimates
**/
class Speed_Estimator
{
	public:
		// public constructor and destructor
		Speed_Estimator(Config_Handler* config_handler, Finite_Difference_Solver* FDS);
		
		// public functions
		int observe(vector<vector<double>> temperature_image, double time);
		double estimate();
		int reset();
		
	private:
		// private variables
		int sequence_length;
		double filter_time_const;
		vector<vector<double>> coarse_x_mesh_z0;
		double coarse_x_step;
		deque<double> front_location_history;
		deque<double> observation_time_history;
		double x_loc_estimate;
		double speed_estimate;
		
		// private functions
		double get_avg(deque<double> input);
		double estimate_front_location(vector<vector<double>> temperature_image);
};


/**
* Constructor for speed estimator
* @param Configuration handler object that contains all loaded and calculated configuration data
*/
Speed_Estimator::Speed_Estimator(Config_Handler* config_handler, Finite_Difference_Solver* FDS)
{	
	// Set sequence length and filter time constant
	sequence_length = config_handler->sequence_length;
	filter_time_const = config_handler->filter_time_const;
	coarse_x_mesh_z0 = FDS->get_coarse_x_mesh_z0();
	
	// Populate observation histories
	for( int i = 0; i < sequence_length; i++ )
	{
		front_location_history.push_back(0.0);
		observation_time_history.push_back( double(sequence_length - i)*(-0.02) );
	}
	
	// Set location and speed estimate to 0
	x_loc_estimate = 0.0;
	speed_estimate = 0.0;
}

/**
* Gets the average value of a deque object
* @param The deque object for which the average is being calculated
* @return The average of the deque object
**/
double Speed_Estimator::get_avg(deque<double> input)
{
	double sum = 0.0;
	for( unsigned int i = 0; i < input.size(); i++ )
	{
		sum += input[i];
	}
	double avg = sum / (double)(input.size());
	return avg;
}

/**
* Adds front location point to location history used to estimate speed
* @param Mean front x location at time of observation
* @param Simulated time at observation
* @return 0 on success, 1 on failure
*/
int Speed_Estimator::observe(vector<vector<double>> temperature_image, double time)
{
	// Estimate the mean x location of the leading edge of the front
	x_loc_estimate = estimate_front_location(temperature_image);
	
	//  Add newest observation
	front_location_history.push_back(x_loc_estimate);
	observation_time_history.push_back(time);
	
	// Remove oldest observation
	front_location_history.pop_front();
	observation_time_history.pop_front();
	
	return 0;
}

/** 
* Estimates the mean x location of the front's leading edge given temperature field observation
* @param Normalized image of temperature field
* @return X location of the mean x location of the front's leading edge based on dT/dx (NON NORMALIZED)
*/
double Speed_Estimator::estimate_front_location(vector<vector<double>> temperature_image)
{
	// Find the x location that corresponds to the max amplitude temperature derivative in the x direction for each column j
	vector<double> dt_dx_max_x_loc = vector<double>(temperature_image[0].size(), -1.0);
	for(unsigned int j = 0; j < temperature_image[0].size(); j++)
	{
		// Tracks the maximum observed temperature derivative in the x direction in column j
		double max_abs_dt_dx_j = 0.0;
		
		for(unsigned int i = 0; i < temperature_image.size(); i++)
		{
			// Store the magnitude of the temperature derivative in the x direction at point (i,j)
			double abs_dt_dx_ij = 0.0;
			
			// Left boundary condition
			if( i==0 )
			{
				abs_dt_dx_ij = abs(-1.5*temperature_image[i][j] + 2.0*temperature_image[i+1][j] + -0.5*temperature_image[i+2][j]);
			}
			
			// Right boundary condition
			else if( i==temperature_image.size()-1 )
			{
				abs_dt_dx_ij = abs(0.5*temperature_image[i-2][j] + -2.0*temperature_image[i-1][j] + 1.5*temperature_image[i][j]);
			}
			
			// Bulk condition
			else
			{
				abs_dt_dx_ij = abs(-0.5*temperature_image[i-1][j] + 0.5*temperature_image[i+1][j]);
			}
			
			// Save max derivative x location so long as the derivate is greater than some threshold
			if ( abs_dt_dx_ij >= max_abs_dt_dx_j || abs_dt_dx_ij > 0.15 )
			{
				dt_dx_max_x_loc[j] = coarse_x_mesh_z0[i][j];
				max_abs_dt_dx_j = abs_dt_dx_ij;
			}
		}
	}
	
	// Sum the admissable front x locations
	double x_loc_sum = 0.0;
	double count = 0.0;
	for(unsigned int j = 0; j < temperature_image[0].size(); j++)
	{
		if( dt_dx_max_x_loc[j] > 0.0 )
		{
			x_loc_sum += dt_dx_max_x_loc[j];
			count = count + 1.0;
		}
	}
	
	// If there are more than 0 instances of fronts, take the average and update the front location estimate, other do not update the front x location
	double curr_estimate = x_loc_estimate;
	if(count > 0.0)
	{
		// Average
		curr_estimate = x_loc_sum / count;
	}
	else
	{
		curr_estimate = x_loc_estimate;
	}
	
	return curr_estimate;
	
}

/**
* Estimates the front velocity based on previous observations of the temperature field
* @return Estiamte of the front velocity (NON NORMALIZED)
*/
double Speed_Estimator::estimate()
{
	// Calculate the average mean x location and sim time from front location history
	double front_location_history_avg = get_avg(front_location_history);
	double observation_time_history_avg = get_avg(observation_time_history);
	
	
	// Apply a simple linear regression to the front mean x location history to determine the front velocity
	double sample_covariance = 0.0;
	double sample_variance = 0.0;
	double observation_time_step_avg = 0.0;
	for(int i = 0; i < sequence_length; i++)
	{
		double delta_x = observation_time_history[i] - observation_time_history_avg;
		double delta_y = front_location_history[i] - front_location_history_avg;
		
		sample_covariance += delta_x*delta_y;
		sample_variance += delta_x*delta_x;
		if (i != sequence_length-1)
		{
			observation_time_step_avg += observation_time_history[i+1] - observation_time_history[i];
		}
	}
	
	// Calculate filter alpha based on time constant and average time step
	observation_time_step_avg = observation_time_step_avg / (double)(sequence_length-1);
	double front_filter_alpha = 1.0 - exp(-observation_time_step_avg/filter_time_const);
	
	// Pass the front velocity signal through a SPLP filter
	double curr_speed_estimate = sample_variance==0.0 ? 0.0 : sample_covariance / sample_variance;
	speed_estimate += front_filter_alpha * ( curr_speed_estimate - speed_estimate );
	
	return speed_estimate;
}

/** 
* Resets the speed estimator for new trajecotry
* @return 0 on success, 1 on failure
*/
int Speed_Estimator::reset()
{
	// Clear the memory
	front_location_history.clear();
	observation_time_history.clear();
	
	// Populate observation histories
	for( int i = 0; i < sequence_length; i++ )
	{
		front_location_history.push_back(0.0);
		observation_time_history.push_back( double(sequence_length - i)*(-0.02) );
	}
	
	// Set x location and speed estimate to 0
	x_loc_estimate = 0.0;
	speed_estimate = 0.0;
	
	return 0;
}

//******************************************************************** USER INTERFACE FUNCTIONS ********************************************************************//
/**
* Prints the finite element solver and simulation parameters to std out
* @param Configuration handler object that contains all loaded and calculated configuration data
*/
void print_params(Config_Handler* config_handler)
{	
	// PPO Agent hyperparameters
	cout << "\nAgent Hyperparameters(\n";
	cout << "  (Autoencoder): " << config_handler->load_path << "\n";
	cout << "  (Steps per Trajectory): " << config_handler->steps_per_trajectory << "\n";
	cout << "  (Trajectories per Batch): " << config_handler->trajectories_per_batch << "\n";
	cout << "  (Epochs per Batch): " << config_handler->epochs_per_batch << "\n";
	cout << "  (Discount Ratio): " << config_handler->gamma << " \n";
	cout << "  (GAE Parameter): " << config_handler->lambda << " \n";
	cout << "  (Clipping Parameter): " << config_handler->epsilon << " \n";
	cout << "  (Start LR): " << config_handler->start_alpha << "\n";
	cout << "  (End LR): " << config_handler->end_alpha << " \n";
	cout << ")\n";
}

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


//******************************************************************** PYTHON API CONVERSION FUNCTIONS ********************************************************************//
/**
* Converts 1D vector<double> to a 1D PyList
* @param The vector used to create list
* @return PyObject pointer pointing at the created list
*/
template <typename T>
PyObject* get_1D_list(T arr)
{
	PyObject *list;
	
	list = PyList_New(arr.size());
	for (unsigned int i = 0; i < arr.size(); i++)
	{
		PyList_SetItem(list, i, PyFloat_FromDouble(arr[i]));
	}
	return list;
}

/**
* Converts 2D vector<vector<int>> to a 2D PyList
* @param The vector used to create list
* @return PyObject pointer pointing at the created list
*/
template <typename T>
PyObject* get_2D_list(T arr)
{
	PyObject *mat, *vec;

	mat = PyList_New(arr.size());
	for (unsigned int i = 0; i < arr.size(); i++)
	{
		vec = PyList_New(arr[0].size());
		for (unsigned int j = 0; j < arr[0].size(); j++)
		{
			PyList_SetItem(vec, j, PyFloat_FromDouble(arr[i][j]));
		}
		PyList_SetItem(mat, i, vec);
	}
	
	return mat;
}

/**
* Converts 3D object to a 3D PyList
* @param The vector used to create list
* @return PyObject pointer pointing at the created list
*/
template <typename T>
PyObject* get_3D_list(T arr)
{
	PyObject *ten, *mat, *vec;

	ten = PyList_New(arr.size());
	for(unsigned int i = 0; i < arr.size(); i++)
	{
		mat = PyList_New(arr[0].size());
		for (unsigned int j = 0; j < arr[0].size(); j++)
		{
			vec = PyList_New(arr[0][0].size());
			for (unsigned int k = 0; k < arr[0][0].size(); k++)
			{
				PyList_SetItem(vec, k, PyFloat_FromDouble(arr[i][j][k]));
			}
			PyList_SetItem(mat, j, vec);
		}
		PyList_SetItem(ten, i, mat);
	}
	
	return ten;
}

/**
* Converts PyList to a 1D vector<T>
* @param The list used to create the vector
* @return 1D vector<T> copy of the PyList
*/
template <typename T>
vector<T> get_vector(PyObject* list)
{
	if (PyList_Size(list) > 0)
	{
		vector<T> out(PyList_Size(list), 0.0);
		for (unsigned int i = 0; i < PyList_Size(list); i++)
		{
			out[i] = PyFloat_AsDouble(PyList_GetItem(list, i));
		}
		return out;
	}
	return vector<T>();
}

//******************************************************************** PYTHON API INITIALIZATION FUNCTIONS ********************************************************************//
/**
* Initializes a PPO agent
* @param Configuration handler pointer that contains all loaded and calculated configuration data
* @param The number of additional inputs beyond the state image given to the agent during action generation
* @param The number of outputs generated by the actor (The number of inputs to the environment)
*/
PyObject* init_agent(Config_Handler* config_handler, int num_addtional_inputs, int num_outputs)
{
	// Define module name
	PyObject* name = PyUnicode_DecodeFSDefault("PPO");

	// Initialize module
	PyObject* module = PyImport_Import(name);
	if (module == NULL)
	{
		fprintf(stderr, "\nFailed to find ppo module.\n");
		PyErr_Print();
		Py_DECREF(name);
		return NULL;
	}
	Py_DECREF(name);

	// Load dictionary of module methods and variables
	PyObject* dict = PyModule_GetDict(module);
	if (dict == NULL)
	{
		fprintf(stderr, "\nFailed to load ppo module dictionary.\n");
		PyErr_Print();
		Py_DECREF(module);
		return NULL;
	}
	Py_DECREF(module);

	// Get the initialization function from the module dictionary
	PyObject* init = PyDict_GetItemString(dict, "Agent");
	if (init == NULL || !PyCallable_Check(init))
	{
		fprintf(stderr, "\nFailed to find PPO Agent __init__ function.\n");
		PyErr_Print();
		Py_DECREF(dict);
		if (init != NULL) { Py_DECREF(init); }
		return NULL;
	}
	Py_DECREF(dict);

	// Build the initialization arguments
	PyObject* init_args = PyTuple_New(12);
	PyTuple_SetItem(init_args, 0, PyLong_FromLong(num_addtional_inputs));
	PyTuple_SetItem(init_args, 1, PyLong_FromLong(num_outputs));
	PyTuple_SetItem(init_args, 2, PyLong_FromLong(config_handler->steps_per_trajectory));
	PyTuple_SetItem(init_args, 3, PyLong_FromLong(config_handler->trajectories_per_batch));
	PyTuple_SetItem(init_args, 4, PyLong_FromLong(config_handler->epochs_per_batch));
	PyTuple_SetItem(init_args, 5, PyFloat_FromDouble(config_handler->gamma));
	PyTuple_SetItem(init_args, 6, PyFloat_FromDouble(config_handler->lambda));
	PyTuple_SetItem(init_args, 7, PyFloat_FromDouble(config_handler->epsilon));
	PyTuple_SetItem(init_args, 8, PyFloat_FromDouble(config_handler->start_alpha));
	PyTuple_SetItem(init_args, 9, PyFloat_FromDouble(config_handler->decay_rate));
	string load_path_str = config_handler->load_path;
	const char* load_path = load_path_str.c_str();
	PyTuple_SetItem(init_args, 10, PyUnicode_DecodeFSDefault(load_path));
	if (config_handler->reset_std)
	{
		PyTuple_SetItem(init_args, 11, Py_True);
	}
	else
	{
		PyTuple_SetItem(init_args, 11, Py_False);
	}

	// Initialize ppo object
	PyObject* object = PyObject_CallObject(init, init_args);
	if (object == NULL)
	{
		fprintf(stderr, "\nFailed to call agent __init__ function.\n");
		PyErr_Print();
		Py_DECREF(init);
		Py_DECREF(init_args);
		return NULL;
	}
	Py_DECREF(init);
	Py_DECREF(init_args);

	// return the class
	return object;
}

/**
* Initializes the save_render_plot class of the PPO module
* @return pyobject pointer to loaded class on success, NULL on failure
*/
PyObject* init_save_render_plot()
{
	// Define module name
	PyObject* name = PyUnicode_DecodeFSDefault("PPO");

	// Initialize module
	PyObject* module = PyImport_Import(name);
	if (module == NULL)
	{
		fprintf(stderr, "\nFailed to find PPO module:\n");
		PyErr_Print();
		Py_DECREF(name);
		return NULL;
	}
	Py_DECREF(name);

	// Load dictionary of module methods and variables
	PyObject* dict = PyModule_GetDict(module);
	if (dict == NULL)
	{
		fprintf(stderr, "\nFailed to load PPO module dictionary:\n");
		PyErr_Print();
		Py_DECREF(module);
		return NULL;
	}
	Py_DECREF(module);

	// Get the initialization function from the module dictionary
	PyObject* init = PyDict_GetItemString(dict, "Save_Plot_Render");
	if (init == NULL || !PyCallable_Check(init))
	{
		fprintf(stderr, "\nFailed to find Save_Plot_Render __init__ function:\n");
		PyErr_Print();
		Py_DECREF(dict);
		if (init != NULL) { Py_DECREF(init); }
		return NULL;
	}
	Py_DECREF(dict);

	// Initialize autoencoder object
	PyObject* object = PyObject_CallObject(init, NULL);
	if (object == NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render __init__ function:\n");
		PyErr_Print();
		Py_DECREF(init);
		return NULL;
	}
	Py_DECREF(init);
	
	// return the class
	return object;
}


//******************************************************************** PYTHON API CALL METHOD FUNCTIONS ********************************************************************//
/**
* Stores training curves to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param vector containing the actor reward curve
* @param vector containing the critic learning curve
* @return 0 on success, 1 on failure
*/
int store_training_curves(PyObject* save_render_plot, vector<double> r_per_episode, vector<double> value_error)
{
	// Convert inputs
	PyObject* py_r_per_episode = get_1D_list<vector<double>>(r_per_episode);
	PyObject* py_value_error = get_1D_list<vector<double>>(value_error);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_training_curves", "(O,O)", py_r_per_episode, py_value_error);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_training_curves function:\n");
		PyErr_Print();
		Py_DECREF(py_r_per_episode);
		Py_DECREF(py_value_error);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_r_per_episode);
	Py_DECREF(py_value_error);
	return 0;
}

/**
* Stores learning rate curves to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param vector containing the actor lr curve
* @param vector containing the critic lr curve
* @return 0 on success, 1 on failure
*/
int store_lr_curves(PyObject* save_render_plot, vector<double> actor_lr, vector<double> critic_lr)
{
	// Convert inputs
	PyObject* py_actor_lr = get_1D_list<vector<double>>(actor_lr);
	PyObject* py_critic_lr = get_1D_list<vector<double>>(critic_lr);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_lr_curves", "(O,O)", py_actor_lr, py_critic_lr);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_lr_curves function:\n");
		PyErr_Print();
		Py_DECREF(py_actor_lr);
		Py_DECREF(py_critic_lr);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_actor_lr);
	Py_DECREF(py_critic_lr);
	return 0;
}

/**
* Stores stdev history to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param vector containing x rate stdev data
* @param vector containing y rate stdev data
* @param vector containing magnitude stdev data
* @return 0 on success, 1 on failure
*/
int store_stdev_history(PyObject* save_render_plot, vector<double> x_stdev, vector<double> y_stdev, vector<double> mag_stdev)
{
	// Convert inputs
	PyObject* py_x_loc_stdev = get_1D_list<vector<double>>(x_stdev);
	PyObject* py_y_loc_stdev = get_1D_list<vector<double>>(y_stdev);
	PyObject* py_mag_stdev = get_1D_list<vector<double>>(mag_stdev);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_stdev_history", "(O,O,O)", py_x_loc_stdev, py_y_loc_stdev, py_mag_stdev);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_stdev_history function:\n");
		PyErr_Print();
		Py_DECREF(py_x_loc_stdev);
		Py_DECREF(py_y_loc_stdev);
		Py_DECREF(py_mag_stdev);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_x_loc_stdev);
	Py_DECREF(py_y_loc_stdev);
	Py_DECREF(py_mag_stdev);
	return 0;
}

/**
* Stores input history to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param vector containing input x location history
* @param vector containing input y location history
* @param vector containing input magnitude percent history
* @param vector containing total power history
* @return 0 on success, 1 on failure
*/
int store_input_history(PyObject* save_render_plot, vector<double> input_location_x, vector<double> input_location_y, vector<double> input_percent, vector<double> power)
{
	// Convert inputs
	PyObject* py_input_location_x = get_1D_list<vector<double>>(input_location_x);
	PyObject* py_input_location_y = get_1D_list<vector<double>>(input_location_y);
	PyObject* py_input_percent = get_1D_list<vector<double>>(input_percent);
	PyObject* py_power = get_1D_list<vector<double>>(power);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_input_history", "(O,O,O,O)", py_input_location_x, py_input_location_y, py_input_percent, py_power);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_input_history function:\n");
		PyErr_Print();
		Py_DECREF(py_input_location_x);
		Py_DECREF(py_input_location_y);
		Py_DECREF(py_input_percent);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_input_location_x);
	Py_DECREF(py_input_location_y);
	Py_DECREF(py_input_percent);
	return 0;
}

/**
* Stores field history to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param vector containing temperature field history
* @param vector containing cure field history
* @return 0 on success, 1 on failure
*/
int store_field_history(PyObject* save_render_plot, vector<vector<vector<double>>> temperature_field, vector<vector<vector<double>>> cure_field, 
vector<vector<vector<double>>> fine_temperature_field, vector<vector<vector<double>>> fine_cure_field, vector<vector<double>> fine_mesh_loc)
{
	
	// Convert inputs
	PyObject* py_temperature_field = get_3D_list<vector<vector<vector<double>>>>(temperature_field);
	PyObject* py_cure_field = get_3D_list<vector<vector<vector<double>>>>(cure_field);
	PyObject* py_fine_temperature_field = get_3D_list<vector<vector<vector<double>>>>(fine_temperature_field);
	PyObject* py_fine_cure_field = get_3D_list<vector<vector<vector<double>>>>(fine_cure_field);
	PyObject* py_fine_mesh_loc = get_2D_list<vector<vector<double>>>(fine_mesh_loc);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_field_history", "(O,O,O,O,O)", py_temperature_field, py_cure_field, py_fine_temperature_field, py_fine_cure_field, py_fine_mesh_loc);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_field_history function:\n");
		PyErr_Print();
		Py_DECREF(py_temperature_field);
		Py_DECREF(py_cure_field);
		Py_DECREF(py_fine_temperature_field);
		Py_DECREF(py_fine_cure_field);
		Py_DECREF(py_fine_mesh_loc);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_temperature_field);
	Py_DECREF(py_cure_field);
	Py_DECREF(py_fine_temperature_field);
	Py_DECREF(py_fine_cure_field);
	Py_DECREF(py_fine_mesh_loc);
	return 0;
}

/**
* Stores front history to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param vector containing front location x and y indicies history
* @param vector containing front location fit history
* @param vector containing front speed field history
* @param vector containing front temperature history
* @param vector containing front shape parameter history
* @return 0 on success, 1 on failure
*/
int store_front_history(PyObject* save_render_plot, vector<vector<vector<double>>> front_curve, vector<vector<double>> front_fit, vector<double> front_velocity, vector<double> front_temperature, vector<double> front_shape_param)
{
	// Convert inputs
	PyObject* py_front_curve = get_3D_list<vector<vector<vector<double>>>>(front_curve);
	PyObject* py_front_fit = get_2D_list<vector<vector<double>>>(front_fit);
	PyObject* py_front_velocity = get_1D_list<vector<double>>(front_velocity);
	PyObject* py_front_temperature = get_1D_list<vector<double>>(front_temperature);
	PyObject* py_front_shape_param = get_1D_list<vector<double>>(front_shape_param);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_front_history", "(O,O,O,O,O)", py_front_curve, py_front_fit, py_front_velocity, py_front_temperature, py_front_shape_param);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_front_history function:\n");
		PyErr_Print();
		Py_DECREF(py_front_curve);
		Py_DECREF(py_front_fit);
		Py_DECREF(py_front_velocity);
		Py_DECREF(py_front_temperature);
		Py_DECREF(py_front_shape_param);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_front_curve);
	Py_DECREF(py_front_fit);
	Py_DECREF(py_front_velocity);
	Py_DECREF(py_front_temperature);
	Py_DECREF(py_front_shape_param);
	return 0;
}

/**
* Stores target, time, and best reward values to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param vector containing target velocity as a function of time
* @param vector containing time history
* @param vector containing reward history
* @return 0 on success, 1 on failure
*/
int store_target_and_time(PyObject* save_render_plot, vector<double> target, vector<double> time, vector<vector<double>> reward)
{
	// Convert inputs
	PyObject* py_target = get_1D_list<vector<double>>(target);
	PyObject* py_time = get_1D_list<vector<double>>(time);
	PyObject* py_reward = get_2D_list<vector<vector<double>>>(reward);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_target_and_time", "(O,O,O)", py_target, py_time, py_reward);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_target_and_time function:\n");
		PyErr_Print();
		Py_DECREF(py_target);
		Py_DECREF(py_time);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_target);
	Py_DECREF(py_time);
	return 0;
}

/**
* Stores top layer of mesh to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param x coordinates of top layer of mesh
* @param y coordinates of top layer of mesh
* @return 0 on success, 1 on failure
*/
int store_top_mesh(PyObject* save_render_plot, vector<vector<double>> mesh_x_z0, vector<vector<double>> mesh_y_z0)
{
	// Convert inputs
	PyObject* py_mesh_x_z0 = get_2D_list<vector<vector<double>>>(mesh_x_z0);
	PyObject* py_mesh_y_z0 = get_2D_list<vector<vector<double>>>(mesh_y_z0);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_top_mesh", "(O,O)", py_mesh_x_z0, py_mesh_y_z0);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_top_mesh function:\n");
		PyErr_Print();
		Py_DECREF(py_mesh_x_z0);
		Py_DECREF(py_mesh_y_z0);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_mesh_x_z0);
	Py_DECREF(py_mesh_y_z0);
	return 0;
}

/**
* Stores input parameters to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param peak irradiance (W/m^2) of input
* @param exponential constant used to calculate input field density
* @return 0 on success, 1 on failure
*/
int store_input_params(PyObject* save_render_plot, double max_input_mag, double exp_const)
{
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_input_params", "(d,d)", max_input_mag, exp_const);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_input_params function:\n");
		PyErr_Print();
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	return 0;

}

/**
* Stores simulation options  to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param whether front speed was the control target
* @param boolean flag that indicates whether the x and y actions were rates or positions
* @param String containing fds and train_agent config values
* @return 0 on success, 1 on failure
*/
int store_options(PyObject* save_render_plot, bool control_speed, bool input_rates, string& configs_string)
{
	// Convert inputs
	int int_control_speed = control_speed ? 1 : 0;
	int int_input_rates = input_rates ? 1 : 0;
	const char* configs_c_str = configs_string.c_str();
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_options", "(i, i, s)", int_control_speed, int_input_rates ,configs_c_str);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_options function:\n");
		PyErr_Print();
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	return 0;
}


//******************************************************************** PYTHON API SAVE, PLOT, RENDER FUNCTIONS ********************************************************************//
/**
* Calls the save, plot, and render functions of the save_render_plot class in the PPO module, decrefs agent and save_render_plot
* @param pointer to the save_render_plot class in the PPO module
* @param Python pointer pointing at the trained agent
* @return 0 on success, 1 on failure
*/
int save_agent_results(PyObject* save_render_plot, PyObject* agent)
{
	// Save
	if(PyObject_CallMethod(save_render_plot, "save", "O", agent) == NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's save function:\n");
		PyErr_Print();
		if (agent != NULL) { Py_DECREF(agent); }
		if (save_render_plot != NULL) { Py_DECREF(save_render_plot); }
		return 1;
	}
	
	// Plot
	if(PyObject_CallMethodObjArgs(save_render_plot, PyUnicode_DecodeFSDefault("plot"), NULL) == NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's plot function:\n");
		PyErr_Print();
		if (agent != NULL) { Py_DECREF(agent); }
		if (save_render_plot != NULL) { Py_DECREF(save_render_plot); }
		return 1;
	}
	
	// Render
	if(PyObject_CallMethodObjArgs(save_render_plot, PyUnicode_DecodeFSDefault("render"), NULL) == NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's render function:\n");
		PyErr_Print();
		if (agent != NULL) { Py_DECREF(agent); }
		if (save_render_plot != NULL) { Py_DECREF(save_render_plot); }
		return 1;
	}
	
	// Free memory
	Py_DECREF(agent);
	Py_DECREF(save_render_plot);
	return 0;
}


//******************************************************************** TRAINING LOOP ********************************************************************//
/**
* Runs a set of trajectories using the PPO policy, updates the PPO agent, and collects relevant training data
* @param The finite element solver object used to propogate time
* @param Configuration handler object that contains all loaded and calculated configuration data
* @param The class used to estimate front speed based on front location estimations
* @param The ppo agent being trained
* @param The save render and plotting class of the ppo agent being trained
* @param The time at which simulation was started
* @return 0 on success, 1 on failure
*/
int run(Finite_Difference_Solver* FDS, Config_Handler* config_handler, Speed_Estimator* estimator, PyObject* agent, PyObject* save_render_plot, auto &start_time)
{
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
	for (int i = 0; i < config_handler->total_trajectories; i++)
	{
		// Initialize simulation variables
		bool done = false;
		double action_1=0.0, stdev_1=0.0, action_2=0.0, stdev_2=0.0, action_3=0.0, stdev_3=0.0, reward;
		vector<double> reward_arr;
		vector<vector<double>> state_image;
		bool run_agent, observe_speed;
		int step_in_trajectory = 0;
		
		// User readout
		print_training_info(i, config_handler->total_trajectories, prev_episode_reward, config_handler->steps_per_trajectory, r_per_episode, best_episode_reward);
		prev_episode_reward = total_reward;

		// Reset
		FDS->reset();
		estimator->reset();
		
		// Simulation loop
		while (!done)
		{
			// Determine what to run this simulation step
			run_agent = (step_in_trajectory % config_handler->steps_per_agent_cycle == 0) || (step_in_trajectory==0);
 			observe_speed = (step_in_trajectory % config_handler->steps_per_speed_estimator_frame == 0) || (step_in_trajectory==0);
			step_in_trajectory++;
			
			// Take a snapshot of the temperature field to update the front speed estimation
			if (observe_speed)
			{
				// Gather temperature state data
				state_image = FDS->get_coarse_temp_z0(true);
				
				// Add observation to speed estimator
				estimator->observe(state_image, FDS->get_curr_sim_time());
			}
			
			// Run the agent
			if (run_agent)
			{
				// Gather temperature state data
				state_image = FDS->get_coarse_temp_z0(true);
				PyObject* py_state_image = get_2D_list<vector<vector<double>>>(state_image);
				
				// Get speed from speed estimator and calculate error
				double front_speed = estimator->estimate();
				double front_speed_error = front_speed / FDS->get_curr_target();
				
				// Combine all additional inputs to PPO agent
				vector<double> additional_ppo_inputs = vector<double>(1, 0.0);
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
				done = FDS->step(action_1, action_2, config_handler->input_rates, action_3);
				
				// Combine the action data
				vector<double> actions = vector<double>(3,0.0);
				actions[0] = action_1;
				actions[1] = action_2;
				actions[2] = action_3;
				PyObject* py_actions = get_1D_list<vector<double>>(actions);

				// Update the agent
				reward_arr = FDS->get_reward();
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
				vector<double> curr_critic_loss_and_lr = get_vector<double>(py_critic_loss_and_lr);
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
				done = FDS->step(action_1, action_2, config_handler->input_rates, action_3);
			}
			
		}

		// Update the best trajectory memory
		prev_episode_reward = total_reward - prev_episode_reward;
		if (prev_episode_reward > best_episode_reward)
		{
			best_episode_reward = prev_episode_reward;
		}

		// Store actor training data
		r_per_episode.push_back(prev_episode_reward/(double)config_handler->steps_per_trajectory);
		if (config_handler->input_rates)
		{
			x_stdev.push_back(0.5*FDS->get_max_input_slew_speed()*stdev_1);
			y_stdev.push_back(0.5*FDS->get_max_input_slew_speed()*stdev_2);		
		}
		else
		{
			x_stdev.push_back(0.5*FDS->get_coarse_x_len()*stdev_1);
			y_stdev.push_back(0.5*FDS->get_coarse_y_len()*stdev_2);		
		}
		mag_stdev.push_back(FDS->get_peak_input_mag()*stdev_3);

		// Final user readout
		if (i == config_handler->total_trajectories - 1) { print_training_info(i, config_handler->total_trajectories, prev_episode_reward, config_handler->steps_per_trajectory, r_per_episode, best_episode_reward); }
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
	bool run_agent, save_frame, observe_speed;
	int step_in_trajectory = 0;
	vector<vector<double>> state_image;
	vector<double> input_state;
	
	// Reset
	FDS->reset();
	estimator->reset();
		
	while (!done)
	{
		// Determine what to run this simulation step
		run_agent = (step_in_trajectory % config_handler->steps_per_agent_cycle == 0) || (step_in_trajectory==0);
		observe_speed = (step_in_trajectory % config_handler->steps_per_speed_estimator_frame == 0) || (step_in_trajectory==0);
		save_frame = (step_in_trajectory % config_handler->steps_per_render_frame == 0) || (step_in_trajectory==0);
		step_in_trajectory++;
		
		// Take a snapshot
		if (observe_speed)
		{
			// Gather temperature state data
			state_image = FDS->get_coarse_temp_z0(true);
			
			// Add observation to speed estimator
			estimator->observe(state_image, FDS->get_curr_sim_time());
		}
		
		// Update the logs
		if (save_frame)
		{
			// Get environment data
			input_state = FDS->get_input_state(false);
			
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
			front_velocity.push_back(FDS->get_front_vel());
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
			state_image = FDS->get_coarse_temp_z0(true);
			PyObject* py_state_image = get_2D_list<vector<vector<double>>>(state_image);
			
			// Get speed from speed estimator and calculate error
			double front_speed = estimator->estimate();
			double front_speed_error = front_speed / FDS->get_curr_target();
			
			// Combine all additional inputs to PPO agent
			vector<double> additional_ppo_inputs = vector<double>(1, 0.0);
			additional_ppo_inputs[0] = front_speed_error;
			PyObject* py_additional_ppo_inputs = get_1D_list<vector<double>>(additional_ppo_inputs);
			
			// Gather input data
			PyObject* py_inputs = get_1D_list<vector<double>>(FDS->get_input_state(true));
		
			// Get agent action based on temperature state data
			PyObject* py_action = PyObject_CallMethod(agent, "get_greedy_action", "(O,O,O)", py_state_image, py_additional_ppo_inputs, py_inputs);
			if (py_action == NULL)
			{
				fprintf(stderr, "\nFailed to call get greedy action function.\n");
				PyErr_Print();
				Py_DECREF(py_state_image);
				Py_DECREF(py_additional_ppo_inputs);
				Py_DECREF(py_inputs);
				return 1;
			}
			
			// Get the agent commanded action
			action_1 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 0));
			action_2 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 1));
			action_3 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 2));

			// Step the environment
			done = FDS->step(action_1, action_2, config_handler->input_rates, action_3);
			
			// Release the python memory
			Py_DECREF(py_state_image);
			Py_DECREF(py_additional_ppo_inputs);
			Py_DECREF(py_inputs);
			Py_DECREF(py_action);
		}
		
		// Step the environment 
		if (!run_agent)
		{
			// Step the environment based on the previously commanded action
			done = FDS->step(action_1, action_2, config_handler->input_rates, action_3);
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
	if(store_options(save_render_plot, FDS->get_control_mode(), config_handler->input_rates, config_handler->configs_string) == 1) {return 1;}

	// Stop clock and print duration
	duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("\nData conversion took: %.1f seconds.", duration);

	// Save, plot, and render
	start_time = chrono::high_resolution_clock::now();
	return save_agent_results(save_render_plot, agent);
}


//******************************************************************** MAIN LOOP ********************************************************************//
int main()
{	
	// Load configuration data
	Config_Handler* config_handler;
	try
	{
		config_handler = new Config_Handler();
	}
	catch (int e)
	{
		cout << "An exception occurred. Exception num " << e << '\n';
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

	// Initialize front speed estimator
	Speed_Estimator* estimator = new Speed_Estimator(config_handler, FDS);

	// Init py environment
	Py_Initialize();
	PyRun_SimpleString("import  sys");
	PyRun_SimpleString("sys.path.append('../py_src/')");
    
	// Init agent
	PyObject* agent = init_agent(config_handler, 1, 3);
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
	print_params(config_handler);
	FDS->print_params();

	// Train agent
	cout << "\nSimulating...\n";
	auto start_time = chrono::high_resolution_clock::now();
	if (run(FDS, config_handler, estimator, agent, save_render_plot, start_time) == 1)
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
