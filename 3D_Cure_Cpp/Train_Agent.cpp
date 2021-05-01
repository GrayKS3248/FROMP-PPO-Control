#define PY_SSIZE_T_CLEAN
#include "Finite_Element_Solver.h"
#include <Python.h>
#include <string>
#include <iomanip>
#include <sstream>
using namespace std;


//******************************************************************** DEBUGGUNG FUNCTIONS ********************************************************************//
/**
* Prints to std out a representation of a 2D vector
* @param The array to be printed
*/
void print_2D(vector<vector<double> > arr, unsigned int len)
{
	// Find the largest order number
	int max_order = 0;
	int curr_order = 0;
	for (unsigned int i = 0; i < len; i++)
	{
		for (unsigned int j = 0; j < arr[0].size(); j++)
		{
			curr_order = (int) floor(log10(abs(arr[i][j])));
			max_order = curr_order > max_order ? curr_order : max_order;
		}
	}

	for (unsigned int i = 0; i <len; i++)
	{
		for (unsigned int j = 0; j < arr[0].size(); j++)
		{
			if (arr[i][j] == 0.0)
			{
				for (int i = 0; i <= max_order; i++)
				{
					cout << " ";
				}
			}
			else
			{
				curr_order = (int) floor(log10(abs(arr[i][j])));
				curr_order = curr_order < 0 ? 0 : curr_order;
				for (int i = 0; i < max_order-curr_order; i++)
				{
					cout << " ";
				}
			}
			if (arr[i][j] > 0.0)
			{
				printf(" %.2f ", arr[i][j]);
			}
			else
			{
				printf("%.2f ", arr[i][j]);
			}
		}
		cout << endl;
	}
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
PyObject* get_1D_list(vector<double> arr)
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
* Converts 2D vector<vector<double>> to a 2D PyList
* @param The vector used to create list
* @return PyObject pointer pointing at the created list
*/
PyObject* get_2D_list(vector<vector<double>> arr)
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
* Converts 3D vector<vector<vector<double>>> to a 3D PyList
* @param The vector used to create list
* @return PyObject pointer pointing at the created list
*/
PyObject* get_3D_list(vector<vector<vector<double>>> arr)
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
* @param The number of (state, action, reward) tuples per trajectory
* @param The number of trajectories per batch
* @param The number of optimization epochs taken per batch
* @param The discount ratio
* @param GAE lambda
* @param PPO clipping ratio
* @param Initial learning rate
* @param Learning rate exponential decay rate
* @param Path to autoencoder loaded
* @return PyObject pointer pointing at the initialized PPO agent on success, NULL on failure
*/
PyObject* init_agent(long steps_per_trajectory, long trajectories_per_batch, long epochs_per_batch, 
double gamma, double lamb, double epsilon, double alpha, double decay_rate,
const char* autoencoder_path)
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
	PyObject* init_args = PyTuple_New(9);
	PyTuple_SetItem(init_args, 0, PyLong_FromLong(steps_per_trajectory));
	PyTuple_SetItem(init_args, 1, PyLong_FromLong(trajectories_per_batch));
	PyTuple_SetItem(init_args, 2, PyLong_FromLong(epochs_per_batch));
	PyTuple_SetItem(init_args, 3, PyFloat_FromDouble(gamma));
	PyTuple_SetItem(init_args, 4, PyFloat_FromDouble(lamb));
	PyTuple_SetItem(init_args, 5, PyFloat_FromDouble(epsilon));
	PyTuple_SetItem(init_args, 6, PyFloat_FromDouble(alpha));
	PyTuple_SetItem(init_args, 7, PyFloat_FromDouble(decay_rate));
	PyTuple_SetItem(init_args, 8, PyUnicode_DecodeFSDefault(autoencoder_path));

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
	PyObject* object = PyObject_CallNoArgs(init);
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
	PyObject* py_r_per_episode = get_1D_list(r_per_episode);
	PyObject* py_value_error = get_1D_list(value_error);
	
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
* Stores stdev history to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param vector containing x rate stdev data
* @param vector containing y rate stdev data
* @param vector containing magnitude stdev data
* @return 0 on success, 1 on failure
*/
int store_stdev_history(PyObject* save_render_plot, vector<double> x_rate_stdev, vector<double> y_rate_stdev, vector<double> mag_stdev)
{
	// Convert inputs
	PyObject* py_x_rate_stdev = get_1D_list(x_rate_stdev);
	PyObject* py_y_rate_stdev = get_1D_list(y_rate_stdev);
	PyObject* py_mag_stdev = get_1D_list(mag_stdev);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_stdev_history", "(O,O,O)", py_x_rate_stdev, py_y_rate_stdev, py_mag_stdev);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_stdev_history function:\n");
		PyErr_Print();
		Py_DECREF(py_x_rate_stdev);
		Py_DECREF(py_y_rate_stdev);
		Py_DECREF(py_mag_stdev);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_x_rate_stdev);
	Py_DECREF(py_y_rate_stdev);
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
	PyObject* py_input_location_x = get_1D_list(input_location_x);
	PyObject* py_input_location_y = get_1D_list(input_location_y);
	PyObject* py_input_percent = get_1D_list(input_percent);
	
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
int store_field_history(PyObject* save_render_plot, vector<vector<vector<double>>> temperature_field, vector<vector<vector<double>>> cure_field)
{
	// Convert inputs
	PyObject* py_temperature_field = get_3D_list(temperature_field);
	PyObject* py_cure_field = get_3D_list(cure_field);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_field_history", "(O,O)", py_temperature_field, py_cure_field);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_field_history function:\n");
		PyErr_Print();
		Py_DECREF(py_temperature_field);
		Py_DECREF(py_cure_field);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_temperature_field);
	Py_DECREF(py_cure_field);
	return 0;
}

/**
* Stores front history to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param vector containing front location history
* @param vector containing front speed field history
* @param vector containing front temperature field history
* @return 0 on success, 1 on failure
*/
int store_front_history(PyObject* save_render_plot, vector<vector<vector<double>>> front_location, 
vector<vector<vector<double>>> front_velocity, vector<vector<vector<double>>> front_temperature)
{
	// Convert inputs
	PyObject* py_front_location = get_3D_list(front_location);
	PyObject* py_front_velocity = get_3D_list(front_velocity);
	PyObject* py_front_temperature = get_3D_list(front_temperature);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_front_history", "(O,O,O)", py_front_location, py_front_velocity, py_front_temperature);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_front_history function:\n");
		PyErr_Print();
		Py_DECREF(py_front_location);
		Py_DECREF(py_front_velocity);
		Py_DECREF(py_front_temperature);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_front_location);
	Py_DECREF(py_front_velocity);
	Py_DECREF(py_front_temperature);
	return 0;
}

/**
* Stores target, time, and best reward values to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param vector containing target velocity as a function of time
* @param vector containing time history
* @return 0 on success, 1 on failure
*/
int store_target_and_time(PyObject* save_render_plot, vector<double> target, vector<double> time)
{
	// Convert inputs
	PyObject* py_target = get_1D_list(target);
	PyObject* py_time = get_1D_list(time);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_target_and_time", "(O,O)", py_target, py_time);
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
	PyObject* py_mesh_x_z0 = get_2D_list(mesh_x_z0);
	PyObject* py_mesh_y_z0 = get_2D_list(mesh_y_z0);
	
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
* Stores leftmost layer of mesh to the save_render_plot class
* @param pointer to the save_render_plot class in the PPO module
* @param y coordinates of leftmost layer of mesh
* @param z coordinates of leftmost layer of mesh
* @return 0 on success, 1 on failure
*/
int store_left_mesh(PyObject* save_render_plot, vector<vector<double>> mesh_y_x0, vector<vector<double>> mesh_z_x0)
{
	// Convert inputs
	PyObject* py_mesh_y_x0 = get_2D_list(mesh_y_x0);
	PyObject* py_mesh_z_x0 = get_2D_list(mesh_z_x0);
	
	// Call function
	PyObject* result = PyObject_CallMethod(save_render_plot, "store_left_mesh", "(O,O)", py_mesh_y_x0, py_mesh_z_x0);
	if (result==NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's store_left_mesh function:\n");
		PyErr_Print();
		Py_DECREF(py_mesh_y_x0);
		Py_DECREF(py_mesh_z_x0);
		return 1;
	}
	
	// Free memory
	Py_DECREF(result);
	Py_DECREF(py_mesh_y_x0);
	Py_DECREF(py_mesh_z_x0);
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
	if(PyObject_CallMethodNoArgs(save_render_plot, PyUnicode_DecodeFSDefault("plot")) == NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's plot function:\n");
		PyErr_Print();
		if (agent != NULL) { Py_DECREF(agent); }
		if (save_render_plot != NULL) { Py_DECREF(save_render_plot); }
		return 1;
	}
	
	// Render
	if(PyObject_CallMethodNoArgs(save_render_plot, PyUnicode_DecodeFSDefault("render")) == NULL)
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
* @param The ppo agent being trained
* @param The save render and plotting class of the ppo agent being trained
* @param The total number of trajectories to be executed
* @param The number of simulation steps taken per single agent cycle
* @param The number of simulation steps taken per single render frame
* @param The number of agent cycles steps in each trajectory
* @return 0 on success, 1 on failure
*/
int run(Finite_Element_Solver* FES, PyObject* agent, PyObject* save_render_plot, int total_trajectories, int steps_per_agent_cycle, int steps_per_frame, int steps_per_trajectory)
{
	// Determine how many frames there are in each trajectory
	int frames_per_trajectory = (int)round((double)FES->get_target_vector_arr_size() / (double)steps_per_frame);
	
	// Agent training data storage
	vector<double> r_per_episode;
	vector<double> critic_loss;
	vector<double> x_rate_stdev;
	vector<double> y_rate_stdev;
	vector<double> mag_rate_stdev;

	// Best trajectory data
	vector<double> input_location_x;
	vector<double> input_location_y;
	vector<double> input_percent;
	vector<double> time;
	vector<double> target;
	vector<vector<vector<double>>> temperature_field(FES->get_num_vert_length(), vector<vector<double>>(FES->get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> cure_field(FES->get_num_vert_length(), vector<vector<double>>(FES->get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> front_location(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> front_velocity(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> front_temperature(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));

	// Current trajectory data
	vector<double> curr_input_location_x;
	vector<double> curr_input_location_y;
	vector<double> curr_input_percent;
	vector<double> curr_time;
	vector<double> curr_target;
	vector<vector<vector<double>>> curr_temperature_field(FES->get_num_vert_length(), vector<vector<double>>(FES->get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> curr_cure_field(FES->get_num_vert_length(), vector<vector<double>>(FES->get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> curr_front_location(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> curr_front_velocity(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> curr_front_temperature(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));

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
		bool run_agent, save_frame;
		int step_in_trajectory = 0;
		
		// User readout
		print_training_info(i, total_trajectories, prev_episode_reward, steps_per_trajectory, r_per_episode, best_episode_reward);
		prev_episode_reward = total_reward;

		// Reset environment
		FES->reset();

		// Simulation loop
		while (!done)
		{
			// Determine what to run this simulation step
			run_agent = (step_in_trajectory % steps_per_agent_cycle == 0) || (step_in_trajectory==0);
			save_frame = (step_in_trajectory % steps_per_frame == 0) || (step_in_trajectory==0);
			step_in_trajectory++;
			
			// Update the logs
			if (save_frame)
			{
				// Get environment data
				vector<double> input_location = FES->get_input_location();
				vector<vector<double>> temp_mesh = FES->get_temp_mesh();
				vector<vector<double>> cure_mesh = FES->get_cure_mesh();
				vector<vector<double>> front_loc = FES->get_front_loc();
				vector<vector<double>> front_vel = FES->get_front_vel();
				vector<vector<double>> front_temp = FES->get_front_temp();
				
				// Store simulation data
				curr_input_location_x.push_back(input_location[0]);
				curr_input_location_y.push_back(input_location[1]);
				curr_input_percent.push_back(FES->get_input_percent());
				curr_time.push_back(FES->get_current_time());
				curr_target.push_back(FES->get_current_target());
				
				// Store environment temperature data
				for (int i = 0; i < FES->get_num_vert_length(); i++)
				{
					for (int j = 0; j < FES->get_num_vert_width(); j++)
					{
						curr_temperature_field[i][j][curr_target.size()-1] = temp_mesh[i][j];
						curr_cure_field[i][j][curr_target.size()-1] = cure_mesh[i][j];
					}
				}
				
				// Store environment front data
				for (int j = 0; j < FES->get_num_vert_width(); j++)
				{
					for (int k = 0; k < FES->get_num_vert_depth(); k++)
					{
						curr_front_location[j][k][curr_target.size()-1] = front_loc[j][k];
						curr_front_velocity[j][k][curr_target.size()-1] = front_vel[j][k];
						curr_front_temperature[j][k][curr_target.size()-1] = front_temp[j][k];
					}
				}
			}
			
			// Run the agent
			if (run_agent)
			{
				// Gather temperature state data
				vector<vector<double>> norm_temp_mesh = FES->get_norm_temp_mesh();
				PyObject* py_norm_temp_mesh = get_2D_list(norm_temp_mesh);
				double input_x_loc = FES->get_input_location()[0];
				double input_y_loc = FES->get_input_location()[1];
				double input_percent = FES->get_input_percent();
				
				// Get agent action based on temperature state data
				PyObject* py_action = PyObject_CallMethod(agent, "get_action", "O,f,f,f", py_norm_temp_mesh, input_x_loc, input_y_loc, input_percent);
				if (py_action == NULL)
				{
					fprintf(stderr, "\nFailed to call get action function.\n");
					PyErr_Print();
					Py_DECREF(py_norm_temp_mesh);
					return 1;
				}
				
				// Get the agent commanded action
				action_1 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 0));
				action_2 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 2));
				action_3 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 4));
				
				// Get the agent's stdev
				stdev_1 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 1));
				stdev_2 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 3));
				stdev_3 = PyFloat_AsDouble(PyTuple_GetItem(py_action, 5));

				// Step the environment
				done = FES->step(action_1, action_2, action_3);

				// Update the agent and collect critic loss data
				reward = FES->get_reward();
				PyObject* py_critic_loss = PyObject_CallMethod(agent, "update_agent", "(O,f,f,f,f,f,f,f)", py_norm_temp_mesh, input_x_loc, input_y_loc, input_percent, action_1, action_2, action_3, reward);
				if (py_critic_loss == NULL)
				{
					fprintf(stderr, "\nFailed to update agent\n");
					PyErr_Print();
					Py_DECREF(py_norm_temp_mesh);
					Py_DECREF(py_action);
					return 1;
				}
				vector<double> curr_critic_loss = get_vector(py_critic_loss);
				for (unsigned int i = 0; i < curr_critic_loss.size(); i++)
				{
					critic_loss.push_back(curr_critic_loss[i]);
				}
				Py_DECREF(py_critic_loss);
	
				// Update reward
				total_reward = total_reward + reward;
				
				// Release the python memory
				Py_DECREF(py_norm_temp_mesh);
				Py_DECREF(py_action);
			}
			
			// Step the environment 
			if (!run_agent)
			{
				// Step the environment based on the previously commanded action
				done = FES->step(action_1, action_2, action_3);
			}
			
		}

		// Update the best trajectory memory
		prev_episode_reward = total_reward - prev_episode_reward;
		if (prev_episode_reward > best_episode_reward || i == 0)
		{
			best_episode_reward = prev_episode_reward;
			input_location_x = curr_input_location_x;
			input_location_y = curr_input_location_y;
			input_percent = curr_input_percent;
			time = curr_time;
			target = curr_target;
			temperature_field = curr_temperature_field;
			cure_field = curr_cure_field;
			front_location = curr_front_location;
			front_velocity = curr_front_velocity;
			front_temperature = curr_front_temperature;
		}

		// Reset the current trajectory memory
		curr_input_location_x.clear();
		curr_input_location_y.clear();
		curr_input_percent.clear();
		curr_time.clear();
		curr_target.clear();

		// Update the logs
		r_per_episode.push_back(prev_episode_reward/(double)steps_per_trajectory);
		x_rate_stdev.push_back(FES->get_max_input_loc_rate()*stdev_1);
		y_rate_stdev.push_back(FES->get_max_input_loc_rate()*stdev_2);
		mag_rate_stdev.push_back(FES->get_max_input_mag()*FES->get_max_input_mag_percent_rate()*stdev_3);

		// Final user readout
		if (i == total_trajectories - 1) { print_training_info(i, total_trajectories, prev_episode_reward, steps_per_trajectory, r_per_episode, best_episode_reward); }
	}
	
	// Store all relevant data to save render and plot module
	if(store_training_curves(save_render_plot, r_per_episode, critic_loss) == 1) {return 1;}
	if(store_stdev_history(save_render_plot, x_rate_stdev, y_rate_stdev, mag_rate_stdev) == 1) {return 1;}
	if(store_input_history(save_render_plot, input_location_x, input_location_y, input_percent) == 1) {return 1;}
	if(store_field_history(save_render_plot, temperature_field, cure_field) == 1) {return 1;}
	if(store_front_history(save_render_plot, front_location, front_velocity, front_temperature) == 1) {return 1;}
	if(store_target_and_time(save_render_plot, target, time) == 1) {return 1;}
	if(store_top_mesh(save_render_plot, FES->get_mesh_x_z0(), FES->get_mesh_y_z0()) == 1) {return 1;}
	if(store_left_mesh(save_render_plot, FES->get_mesh_y_x0(), FES->get_mesh_z_x0()) == 1) {return 1;}
	if(store_input_params(save_render_plot, FES->get_max_input_mag(), FES->get_exp_const()) == 1) {return 1;}
	if(store_options(save_render_plot, FES->get_control_speed()) == 1) {return 1;}

	// Save, plot, and render
	return save_agent_results(save_render_plot, agent);
}


//******************************************************************** MAIN LOOP ********************************************************************//
int main()
{	
	// Agent load parameters
	const char* autoencoder_path = "results/ks3_obj1_bn64_U";
	
	// Agent training parameter
	int total_trajectories = 500;
	int steps_per_trajectory = 100;
	int trajectories_per_batch = 20;
	int epochs_per_batch = 20;
	
	// Agent hyperparameters
	double gamma = 0.99;
	double lamb = 0.95;
	double epsilon = 0.20;
	double start_alpha = 1.0e-3;
	double end_alpha = 7.5e-4;

	// Rendering parameters
	double frame_rate = 30.0;

	// Initialize FES
	Finite_Element_Solver FES = Finite_Element_Solver();

	// Calculated agent parameters
	double decay_rate = pow(end_alpha/start_alpha, (double)trajectories_per_batch/(double)total_trajectories);
	double agent_execution_period = (FES.get_sim_duration() / (double)steps_per_trajectory);
	int steps_per_agent_cycle = (int) round(agent_execution_period / FES.get_time_step());

	// Calculated rendering parameters
	int steps_per_frame = (int) round(1.0 / (FES.get_time_step() * frame_rate));
	steps_per_frame = steps_per_frame <= 0 ? 1 : steps_per_frame;

	// Init py environment
	Py_Initialize();
	
	// Init agent
	PyObject* agent = init_agent(steps_per_trajectory, trajectories_per_batch, epochs_per_batch, gamma, lamb, epsilon, start_alpha, decay_rate, autoencoder_path);
	if (agent == NULL) { Py_FinalizeEx(); return 1; }

	// Init save_render_plot
	PyObject* save_render_plot = init_save_render_plot();
	if (save_render_plot == NULL) { Py_FinalizeEx(); return 1; }

	// Print parameters to stdout and log
	print_params(autoencoder_path, steps_per_trajectory, trajectories_per_batch, epochs_per_batch, gamma, lamb, epsilon, start_alpha, end_alpha);
	FES.print_params();

	// Train agent
	auto start_time = chrono::high_resolution_clock::now();
	if (run(&FES, agent, save_render_plot, total_trajectories, steps_per_agent_cycle, steps_per_frame, steps_per_trajectory) == 1) { return 1; };
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("Training Took: %.1f seconds.\n", duration);
	
	// Finish
	Py_FinalizeEx();
	return 0;
}
