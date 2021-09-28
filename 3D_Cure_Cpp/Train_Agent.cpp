#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "Finite_Difference_Solver.h"

using namespace std;


//******************************************************************** CONFIGURATION FUNCTIONS ********************************************************************//
/**
* Loads parameters from .cfg file
* @return 0 on success, 1 on failure
*/
int load_config(string& autoencoder_path_str, string& estimator_path_str, int& total_trajectories, int& steps_per_trajectory, int& trajectories_per_batch, int& epochs_per_batch, bool& reset_std, double& gamma, double& lamb, double& epsilon, double& start_alpha, double& end_alpha, double& frame_rate)
{
	// Load from config file
	ifstream config_file;
	config_file.open("config_files/train_agent.cfg");
	string config_dump;
	string bool_dump;
	if (config_file.is_open())
	{
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> autoencoder_path_str;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> estimator_path_str;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
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
		config_file >> config_dump >> bool_dump;
		if (bool_dump.compare("true")==0)
		{
			reset_std = true;
		}
		else if (bool_dump.compare("false")==0)
		{
			reset_std = false;
		}
		else
		{
			cout << "\nReset stdev configuration not recognized.";
			return 1;
		}
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> gamma;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> lamb;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> epsilon;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> start_alpha;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> end_alpha;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> frame_rate;
	}
	else
	{
		cout << "Unable to open config_files/train_agent.cfg." << endl;
		return 1;
	}
	config_file.close();
	return 0;
}


//******************************************************************** USER INTERFACE FUNCTIONS ********************************************************************//
/**
* Prints the finite element solver and simulation parameters to std out
* @param path to autoencoder used to define encoder half of PPO agent
* @param number of RL agent actions taken per trajectory 
* @param number of trajectories per RL optimization batch
* @param number of optimization steps taken on each RL optimization batch
* @param RL discount ratio
* @param RL GAE parameter
* @param RL clipping parameter
* @param starting learning rate of RL agent
* @param ending learning rate of RL agent
*/
void print_params(const char* autoencoder_path, int steps_per_trajectory, int trajectories_per_batch, int epochs_per_batch, double gamma, double lamb, double epsilon, double start_alpha, double end_alpha)
{	
	// PPO Agent hyperparameters
	cout << "\nAgent Hyperparameters(\n";
	cout << "  (Autoencoder): " << autoencoder_path << "\n";
	cout << "  (Steps per Trajectory): " << steps_per_trajectory << "\n";
	cout << "  (Trajectories per Batch): " << trajectories_per_batch << "\n";
	cout << "  (Epochs per Batch): " << epochs_per_batch << "\n";
	cout << "  (Discount Ratio): " << gamma << " \n";
	cout << "  (GAE Parameter): " << lamb << " \n";
	cout << "  (Clipping Parameter): " << epsilon << " \n";
	cout << "  (Start LR): " << start_alpha << "\n";
	cout << "  (End LR): " << end_alpha << " \n";
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
void print_training_info(int curr_trajectory, int total_trajectories, double prev_episode_reward, 
int steps_per_trajectory, vector<double> r_per_episode, double best_episode)
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
* Converts PyList to a 1D vector<double>
* @param The list used to create the vector
* @return 1D vector<double> copy of the PyList
*/
vector<double> get_vector(PyObject* list)
{
	if (PyList_Size(list) > 0)
	{
		vector<double> out(PyList_Size(list), 0.0);
		for (unsigned int i = 0; i < PyList_Size(list); i++)
		{
			out[i] = PyFloat_AsDouble(PyList_GetItem(list, i));
		}
		return out;
	}
	return vector<double>();
}

//******************************************************************** PYTHON API INITIALIZATION FUNCTIONS ********************************************************************//
/**
* Initializes the python PPO agent
* @param The number of normalized temperature frames used per state
* @param The number of inputs to the system
* @param The number of (state, action, reward) tuples per trajectory
* @param The number of trajectories per batch
* @param The number of optimization epochs taken per batch
* @param The discount ratio
* @param GAE lambda
* @param PPO clipping ratio
* @param Initial learning rate
* @param Learning rate exponential decay rate
* @param Path to autoencoder loaded
* @param Boolean flag indicating whether to reset actor stdev
* @return PyObject pointer pointing at the initialized PPO agent on success, NULL on failure
*/
PyObject* init_agent(long num_states, long num_inputs, long steps_per_trajectory, long trajectories_per_batch,
long epochs_per_batch, double gamma, double lamb, double epsilon, double alpha, double decay_rate, const char* autoencoder_path, bool reset_std)
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
	PyTuple_SetItem(init_args, 0, PyLong_FromLong(num_states));
	PyTuple_SetItem(init_args, 1, PyLong_FromLong(num_inputs));
	PyTuple_SetItem(init_args, 2, PyLong_FromLong(steps_per_trajectory));
	PyTuple_SetItem(init_args, 3, PyLong_FromLong(trajectories_per_batch));
	PyTuple_SetItem(init_args, 4, PyLong_FromLong(epochs_per_batch));
	PyTuple_SetItem(init_args, 5, PyFloat_FromDouble(gamma));
	PyTuple_SetItem(init_args, 6, PyFloat_FromDouble(lamb));
	PyTuple_SetItem(init_args, 7, PyFloat_FromDouble(epsilon));
	PyTuple_SetItem(init_args, 8, PyFloat_FromDouble(alpha));
	PyTuple_SetItem(init_args, 9, PyFloat_FromDouble(decay_rate));
	PyTuple_SetItem(init_args, 10, PyUnicode_DecodeFSDefault(autoencoder_path));
	if (reset_std)
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


/**
* Initializes the python speed estimator
* @param Path to trained speed estimator
* @return PyObject pointer pointing at the initialized speed estimator on success, NULL on failure
*/
PyObject* init_speed_estimator(const char* load_path)
{
	// Define module name
	PyObject* name = PyUnicode_DecodeFSDefault("Speed_Estimator");

	// Initialize module
	PyObject* module = PyImport_Import(name);
	if (module == NULL)
	{
		fprintf(stderr, "\nFailed to find Speed_Estimator module.\n");
		PyErr_Print();
		Py_DECREF(name);
		return NULL;
	}
	Py_DECREF(name);

	// Load dictionary of module methods and variables
	PyObject* dict = PyModule_GetDict(module);
	if (dict == NULL)
	{
		fprintf(stderr, "\nFailed to load Speed_Estimator module dictionary.\n");
		PyErr_Print();
		Py_DECREF(module);
		return NULL;
	}
	Py_DECREF(module);

	// Get the initialization function from the module dictionary
	PyObject* init = PyDict_GetItemString(dict, "Estimator");
	if (init == NULL || !PyCallable_Check(init))
	{
		fprintf(stderr, "\nFailed to find Speed_Estimator Estimator __init__ function.\n");
		PyErr_Print();
		Py_DECREF(dict);
		if (init != NULL) { Py_DECREF(init); }
		return NULL;
	}
	Py_DECREF(dict);

	// Build the initialization arguments
	PyObject* init_args = PyTuple_New(8);
	PyTuple_SetItem(init_args, 0, PyUnicode_DecodeFSDefault(""));
	PyTuple_SetItem(init_args, 1, PyLong_FromLong(0));
	PyTuple_SetItem(init_args, 2, PyLong_FromLong(0));
	PyTuple_SetItem(init_args, 3, PyLong_FromLong(0));
	PyTuple_SetItem(init_args, 4, PyLong_FromLong(0));
	PyTuple_SetItem(init_args, 5, PyLong_FromLong(0));
	PyTuple_SetItem(init_args, 6, PyLong_FromLong(0));
	PyTuple_SetItem(init_args, 7, PyUnicode_DecodeFSDefault(load_path));

	// Initialize ppo object
	PyObject* object = PyObject_CallObject(init, init_args);
	if (object == NULL)
	{
		fprintf(stderr, "\nFailed to call Estimator __init__ function.\n");
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
int store_stdev_history(PyObject* save_render_plot, vector<double> x_loc_stdev, vector<double> y_loc_stdev, vector<double> mag_stdev)
{
	// Convert inputs
	PyObject* py_x_loc_stdev = get_1D_list<vector<double>>(x_loc_stdev);
	PyObject* py_y_loc_stdev = get_1D_list<vector<double>>(y_loc_stdev);
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
* @return 0 on success, 1 on failure
*/
int store_input_history(PyObject* save_render_plot, vector<double> input_location_x, vector<double> input_location_y, vector<double> input_percent)
{
	// Convert inputs
	PyObject* py_input_location_x = get_1D_list<vector<double>>(input_location_x);
	PyObject* py_input_location_y = get_1D_list<vector<double>>(input_location_y);
	PyObject* py_input_percent = get_1D_list<vector<double>>(input_percent);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_input_history", "(O,O,O)", py_input_location_x, py_input_location_y, py_input_percent);
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
* @param vector containing front location x indicies history
* @param vector containing front location y indicies history
* @param vector containing front speed field history
* @param vector containing front temperature field history
* @return 0 on success, 1 on failure
*/

int store_front_history(PyObject* save_render_plot, vector<vector<vector<double>>> front_curve, vector<double> front_velocity, vector<double> front_temperature, vector<double> front_shape_param)
{
	// Convert inputs
	PyObject* py_front_curve = get_3D_list<vector<vector<vector<double>>>>(front_curve);
	PyObject* py_front_velocity = get_1D_list<vector<double>>(front_velocity);
	PyObject* py_front_temperature = get_1D_list<vector<double>>(front_temperature);
	PyObject* py_front_shape_param = get_1D_list<vector<double>>(front_shape_param);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_front_history", "(O,O,O,O)", py_front_curve, py_front_velocity, py_front_temperature, py_front_shape_param);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_front_history function:\n");
		PyErr_Print();
		Py_DECREF(py_front_curve);
		Py_DECREF(py_front_velocity);
		Py_DECREF(py_front_temperature);
		Py_DECREF(py_front_shape_param);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_front_curve);
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
* @return 0 on success, 1 on failure
*/
int store_options(PyObject* save_render_plot, bool control_speed)
{
	// Convert inputs
	int int_control_speed = control_speed ? 1 : 0;
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_options", "(i)", int_control_speed);
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


//******************************************************************** PYTHON API SPEED ESTIMATOR FUNCTIONS ********************************************************************//
/**
* Calls the speed estimator's get params
* @param pointer to the speed_estimator class
* @return 0 on success, 1 on failure
*/
int get_estimator_params(PyObject* speed_estimator, int&sequence_length, double&image_spacing)
{	
	
	PyObject* py_params = PyObject_CallMethod(speed_estimator, "get_params", NULL);
	if(py_params == NULL)
	{
		fprintf(stderr, "\nFailed to call Estimator's get_params function:\n");
		PyErr_Print();
		return 1;
	}
	
	sequence_length = PyLong_AsLong(PyTuple_GetItem(py_params, 0));
	image_spacing = PyFloat_AsDouble(PyTuple_GetItem(py_params, 1));
	
	Py_DECREF(py_params);
	return 0;
}

/**
* Calls the speed estimator's get speed function
* @param pointer to the speed_estimator class
* @return estimated speed
*/
int estimate_error(PyObject* error_estimator, deque<vector<vector<double>>> image_sequence, double&estimated_error)
{
	PyObject* py_image_sequence = get_3D_list<deque<vector<vector<double>>>>(image_sequence);
	
	
	PyObject* py_estimated_error = PyObject_CallMethod(error_estimator, "get_estimate", "O", py_image_sequence);
	if(py_estimated_error == NULL)
	{
		fprintf(stderr, "\nFailed to call Estimator's get_estimate function:\n");
		PyErr_Print();
		Py_DECREF(py_image_sequence);
		return 1;
	}
	
	estimated_error = PyFloat_AsDouble(py_estimated_error);
	
	Py_DECREF(py_image_sequence);
	Py_DECREF(py_estimated_error);
	return 0;
}


//******************************************************************** TRAINING LOOP ********************************************************************//
/**
* Runs a set of trajectories using the PPO policy, updates the PPO agent, and collects relevant training data
* @param The finite element solver object used to propogate time
* @param The ppo agent being trained
* @param The pretrained front error estimator
* @param The save render and plotting class of the ppo agent being trained
* @param The total number of trajectories to be executed
* @param The number of simulation steps taken per single agent cycle
* @param The number of simulation steps taken per single render frame
* @param The number of temperature frames used per state
* @return 0 on success, 1 on failure
*/
int run(Finite_Difference_Solver* FDS, PyObject* agent, PyObject* error_estimator, PyObject* save_render_plot, int total_trajectories, int steps_per_agent_cycle, int steps_per_frame, int steps_per_trajectory, auto &start_time)
{
	// Agent training data storage
	vector<double> r_per_episode;
	vector<double> critic_loss;
	vector<double> actor_lr;
	vector<double> critic_lr;
	vector<double> x_loc_stdev;
	vector<double> y_loc_stdev;
	vector<double> mag_rate_stdev;

	// Simulation set variables
	double total_reward = 0.0;
	double best_episode_reward = 0.0;
	double prev_episode_reward = 0.0;
	
	// Image sequence variables
	int sequence_length;
	double image_spacing;
	if(get_estimator_params(error_estimator, sequence_length, image_spacing) == 1) {return 1;}
	int steps_per_image_sequence_frame = (int) round(image_spacing / FDS->get_coarse_time_step());
 
	// Run a set of episodes
	for (int i = 0; i < total_trajectories; i++)
	{
		// Initialize simulation variables
		bool done = false;
		double action_1=0.0, stdev_1=0.0, action_2=0.0, stdev_2=0.0, action_3=0.0, stdev_3=0.0, reward, estimated_error, monomer_burn_temp;
		vector<double> reward_arr;
		bool run_agent, take_snapshot;
		int step_in_trajectory = 0;
		deque<vector<vector<double>>> image_sequence;
		vector<vector<double>> state_image;
		
		// User readout
		print_training_info(i, total_trajectories, prev_episode_reward, steps_per_trajectory, r_per_episode, best_episode_reward);
		prev_episode_reward = total_reward;

		// Reset environment
		FDS->reset();
		
		// Populate the initial image_sequence
		for (int i = 0; i < sequence_length; i++)
		{
			image_sequence.push_back(FDS->get_norm_coarse_temp_z0());
		}

		// Simulation loop
		while (!done)
		{
			// Determine what to run this simulation step
			run_agent = (step_in_trajectory % steps_per_agent_cycle == 0) || (step_in_trajectory==0);
			take_snapshot = (step_in_trajectory % steps_per_image_sequence_frame == 0);
			step_in_trajectory++;
			
			// Take a snapshot
			if (take_snapshot)
			{
				image_sequence.push_back(FDS->get_norm_coarse_temp_z0());
				image_sequence.pop_front();
				if (estimate_error(error_estimator, image_sequence, estimated_error) == 1) {return 1;}
				estimated_error = estimated_error / FDS->get_curr_target();
			}
			
			// Run the agent
			if (run_agent)
			{
				// Gather temperature state data
				state_image = FDS->get_coarse_temp_z0();
				PyObject* py_state_image = get_2D_list<vector<vector<double>>>(state_image);
				
				// Gather reference temperature
				monomer_burn_temp = FDS->get_monomer_burn_temp();
				
				// Gather target error data
				PyObject *py_estimated_error;
				py_estimated_error = PyList_New(1);
				PyList_SetItem(py_estimated_error, 0, PyFloat_FromDouble(estimated_error));
				
				// Gather input data
				PyObject* py_inputs = get_1D_list<vector<double>>(FDS->get_input());
				
				// Get agent action based on temperature state data
				PyObject* py_action_and_stdev = PyObject_CallMethod(agent, "get_action", "(O,f,O,O)", py_state_image, monomer_burn_temp, py_estimated_error, py_inputs);
				if (py_action_and_stdev == NULL)
				{
					fprintf(stderr, "\nFailed to call get action function.\n");
					PyErr_Print();
					Py_DECREF(py_state_image);
					Py_DECREF(py_estimated_error);
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

				// Update the agent and collect critic loss data
				reward_arr = FDS->get_reward();
				reward = reward_arr[0];
				PyObject* py_critic_loss_and_lr = PyObject_CallMethod(agent, "update_agent", "(O,f,O,O,O,f)", py_state_image, monomer_burn_temp, py_estimated_error, py_inputs, py_actions, reward);
				if (py_critic_loss_and_lr == NULL)
				{
					fprintf(stderr, "\nFailed to update agent\n");
					PyErr_Print();
					Py_DECREF(py_state_image);
					Py_DECREF(py_estimated_error);
					Py_DECREF(py_inputs);
					Py_DECREF(py_action_and_stdev);
					Py_DECREF(py_actions);
					return 1;
				}
				vector<double> curr_critic_loss_and_lr = get_vector(py_critic_loss_and_lr);
				if (curr_critic_loss_and_lr.size() > 0)
				{
					actor_lr.push_back(curr_critic_loss_and_lr[0]);
					critic_lr.push_back(curr_critic_loss_and_lr[1]);
					for (unsigned int i = 2; i < curr_critic_loss_and_lr.size(); i++)
					{
						critic_loss.push_back(curr_critic_loss_and_lr[i]);
					}
				}
				Py_DECREF(py_critic_loss_and_lr);
	
				// Update reward
				total_reward = total_reward + reward;
				
				// Release the python memory
				Py_DECREF(py_state_image);
				Py_DECREF(py_estimated_error);
				Py_DECREF(py_inputs);
				Py_DECREF(py_action_and_stdev);
				Py_DECREF(py_actions);
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
		x_loc_stdev.push_back(FDS->get_max_input_slew_speed()*stdev_1);
		y_loc_stdev.push_back(FDS->get_max_input_slew_speed()*stdev_2);
		mag_rate_stdev.push_back(FDS->get_peak_input_mag()*FDS->get_max_input_mag_percent_rate()*stdev_3);

		// Final user readout
		if (i == total_trajectories - 1) { print_training_info(i, total_trajectories, prev_episode_reward, steps_per_trajectory, r_per_episode, best_episode_reward); }
	}
	
	// Trajectory variables
	vector<double> input_location_x;
	vector<double> input_location_y;
	vector<double> input_percent;
	vector<double> time;
	vector<double> target;
	vector<vector<double>> reward;
	vector<double> front_velocity;
	vector<double> front_temperature;
	vector<double> front_shape_param;
	vector<vector<double>> fine_mesh_loc;
	vector<vector<vector<double>>> front_curve;
	vector<vector<vector<double>>> temperature_field;
	vector<vector<vector<double>>> cure_field;
	vector<vector<vector<double>>> fine_temperature_field;
	vector<vector<vector<double>>> fine_cure_field;
	
	// Run the simulation loop again to generate trajectory data
	bool done = false;
	double action_1=0.0, action_2=0.0, action_3=0.0, estimated_error, monomer_burn_temp;
	bool run_agent, save_frame, take_snapshot;
	int step_in_trajectory = 0;
	deque<vector<vector<double>>> image_sequence;
	vector<vector<double>> state_image;
	FDS->reset();
	
	// Populate the initial image_sequence
	for (int i = 0; i < sequence_length; i++)
	{
		image_sequence.push_back(FDS->get_norm_coarse_temp_z0());
	}
		
		
	while (!done)
	{
		// Determine what to run this simulation step
		run_agent = (step_in_trajectory % steps_per_agent_cycle == 0) || (step_in_trajectory==0);
		take_snapshot = (step_in_trajectory % steps_per_image_sequence_frame == 0);
		save_frame = (step_in_trajectory % steps_per_frame == 0) || (step_in_trajectory==0);
		step_in_trajectory++;
		
		// Take a snapshot
		if (take_snapshot)
		{
			image_sequence.push_back(FDS->get_norm_coarse_temp_z0());
			image_sequence.pop_front();
			if (estimate_error(error_estimator, image_sequence, estimated_error) == 1) {return 1;}
			estimated_error = estimated_error / FDS->get_curr_target();
		}
		
		// Update the logs
		if (save_frame)
		{
			// Get environment data
			vector<double> input_location = FDS->get_input_location();
			
			// Store simulation input data
			input_location_x.push_back(input_location[0]);
			input_location_y.push_back(input_location[1]);
			input_percent.push_back(FDS->get_input_percent());
			
			// Store simualtion target and time data
			time.push_back(FDS->get_curr_sim_time());
			target.push_back(FDS->get_curr_target());
			reward.push_back(FDS->get_reward());
			
			// Store simulation front data
			front_velocity.push_back(FDS->get_front_vel());
			front_temperature.push_back(FDS->get_front_temp());
			front_shape_param.push_back(FDS->get_front_shape_param());
			front_curve.push_back(FDS->get_front_curve());
			
			// Store fine mesh data
			fine_mesh_loc.push_back(FDS->get_fine_mesh_loc());
			
			// Store simulation field data
			temperature_field.push_back(FDS->get_coarse_temp_z0());
			cure_field.push_back(FDS->get_coarse_cure_z0());
			fine_temperature_field.push_back(FDS->get_fine_temp_z0());
			fine_cure_field.push_back(FDS->get_fine_cure_z0());
			
		}
		
		// Run the agent
		if (run_agent)
		{
			// Gather temperature state data
			state_image = FDS->get_coarse_temp_z0();
			PyObject* py_state_image = get_2D_list<vector<vector<double>>>(state_image);
			
			// Gather reference temperature
			monomer_burn_temp = FDS->get_monomer_burn_temp();
				
			// Gather target error data
			PyObject *py_estimated_error;
			py_estimated_error = PyList_New(1);
			PyList_SetItem(py_estimated_error, 0, PyFloat_FromDouble(estimated_error));
				
			// Gather input data
			PyObject* py_inputs = get_1D_list<vector<double>>(FDS->get_input());
			
			// Get agent action based on temperature state data
			PyObject* py_action = PyObject_CallMethod(agent, "get_greedy_action", "(O,f,O,O)", py_state_image, monomer_burn_temp, py_estimated_error, py_inputs);
			if (py_action == NULL)
			{
				fprintf(stderr, "\nFailed to call get greedy action function.\n");
				PyErr_Print();
				Py_DECREF(py_state_image);
				Py_DECREF(py_inputs);
				return 1;
			}
			
			// Get the agent commanded action
			action_1 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 0));
			action_2 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 1));
			action_3 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 2));

			// Step the environment
			done = FDS->step(action_1, action_2, action_3);
			
			// Release the python memory
			Py_DECREF(py_state_image);
			Py_DECREF(py_inputs);
			Py_DECREF(py_action);
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
	if(store_stdev_history(save_render_plot, x_loc_stdev, y_loc_stdev, mag_rate_stdev) == 1) {return 1;}
	if(store_input_history(save_render_plot, input_location_x, input_location_y, input_percent) == 1) {return 1;}
	if(store_field_history(save_render_plot, temperature_field, cure_field, fine_temperature_field, fine_cure_field, fine_mesh_loc) == 1) {return 1;}
	if(store_front_history(save_render_plot, front_curve, front_velocity, front_temperature, front_shape_param) == 1) {return 1;}
	if(store_target_and_time(save_render_plot, target, time, reward) == 1) {return 1;}
	if(store_top_mesh(save_render_plot, FDS->get_coarse_x_mesh_z0(), FDS->get_coarse_y_mesh_z0()) == 1) {return 1;}
	if(store_input_params(save_render_plot, FDS->get_peak_input_mag(), FDS->get_input_const()) == 1) {return 1;}
	if(store_options(save_render_plot, FDS->get_control_mode()) == 1) {return 1;}

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
	// Load parameters
	string autoencoder_path_str;
	string estimator_path_str;
	int total_trajectories;
	int steps_per_trajectory;
	int trajectories_per_batch;
	int epochs_per_batch;
	bool reset_std;
	double gamma;
	double lamb;
	double epsilon;
	double start_alpha;
	double end_alpha;
	double frame_rate;
	if (load_config(autoencoder_path_str, estimator_path_str, total_trajectories, steps_per_trajectory, trajectories_per_batch, epochs_per_batch, reset_std, gamma, lamb, epsilon, start_alpha, end_alpha, frame_rate) == 1) { return 1; }
	const char* autoencoder_path = autoencoder_path_str.c_str();
	const char* estimator_path = estimator_path_str.c_str();

	// Initialize FDS
	Finite_Difference_Solver* FDS;
	try
	{
		FDS = new Finite_Difference_Solver();
	}
	catch (int e)
	{
		cout << "An exception occurred. Exception num " << e << '\n';
		return 1;
	}

	// Calculated agent parameters
	double decay_rate = pow(end_alpha/start_alpha, (double)trajectories_per_batch/(double)total_trajectories);
	double agent_execution_period = (FDS->get_sim_duration() / (double)steps_per_trajectory);
	int steps_per_agent_cycle = (int) round(agent_execution_period / FDS->get_coarse_time_step());

	// Calculated rendering parameters
	int steps_per_frame = (int) round(1.0 / (FDS->get_coarse_time_step() * frame_rate));
	steps_per_frame = steps_per_frame <= 0 ? 1 : steps_per_frame;

	// Init py environment
	Py_Initialize();
	PyRun_SimpleString("import  sys");
	PyRun_SimpleString("sys.path.append('./')");
    
	// Init agent
	PyObject* agent = init_agent(1, 3, steps_per_trajectory, trajectories_per_batch, epochs_per_batch, gamma, lamb, epsilon, start_alpha, decay_rate, autoencoder_path, reset_std);
	if (agent == NULL) { Py_FinalizeEx(); return 1; }
    
    	// Init save_render_plot
	PyObject* save_render_plot = init_save_render_plot();
	if (save_render_plot == NULL) { Py_FinalizeEx(); return 1; }

	// Init speed estimator
	PyObject* error_estimator = init_speed_estimator(estimator_path);
	if (error_estimator == NULL) { Py_FinalizeEx(); return 1; }

	// Print parameters to stdout
	print_params(autoencoder_path, steps_per_trajectory, trajectories_per_batch, epochs_per_batch, gamma, lamb, epsilon, start_alpha, end_alpha);
	FDS->print_params();

	// Train agent
	cout << "\nSimulating...\n";
	auto start_time = chrono::high_resolution_clock::now();
	if (run(FDS, agent, error_estimator, save_render_plot, total_trajectories, steps_per_agent_cycle, steps_per_frame, steps_per_trajectory, start_time) == 1) { return 1; };
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("Saving and Rendering Took: %.1f seconds.\n\nDone!", duration);
	
	// Finish
	Py_FinalizeEx();
	return 0;
}
