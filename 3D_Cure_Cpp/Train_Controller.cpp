#define PY_SSIZE_T_CLEAN
#include "Finite_Element_Solver.h"
#include <Python.h>
#include <string>
#include <iomanip>
#include <sstream>

using namespace std;

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

/**
* Prints the finite element solver and simulation parameters to std out
* @param size of the encoder bottleneck
* @param number of autoencoder frames saved per trajecotry
* @param number of frames per autoencoder optimization epoch
* @param starting learning rate of autoencoder
* @param ending learning rate of autoencoder
* @param x dimension of images sent to autoencoder
* @param y dimension of images sent to autoencoder
* @param the objective function used to update the autoencoder
* @param number of RL agent actions taken per trajectory 
* @param number of trajectories per RL optimization batch
* @param number of optimization steps taken on each RL optimization batch
* @param RL discount ratio
* @param RL GAE parameter
* @param RL clipping parameter
* @param starting learning rate of RL agent
* @param ending learning rate of RL agent
*/
void print_params(int encoded_size, int samples_per_trajectory, int samples_per_batch, double ae_start_alpha, double ae_end_alpha, int x_dim, int y_dim, long objective_fnc, int steps_per_trajectory, int trajectories_per_batch, int num_epochs, double gamma, double lamb, double epsilon, double rl_start_alpha, double rl_end_alpha)
{
	// Autoencoder hyperparameters
	cout << "\nAutoencoder Hyperparameters(\n";
	cout << "  (Objective Fnc): " << objective_fnc << "\n";
	cout << "  (Bottleneck): " << encoded_size << "\n";
	cout << "  (Image Dimenstions): " << x_dim << " x " << y_dim << "\n";
	cout << "  (Samples per Trajectory): " << samples_per_trajectory << "\n";
	cout << "  (Samples per Batch): " << samples_per_batch << " \n";
	cout << "  (Start LR): " << ae_start_alpha << "\n";
	cout << "  (End LR): " << ae_end_alpha << " \n";
	cout << ")\n";
	
	// PPO Agent hyperparameters
	cout << "\nPPO Agent Hyperparameters(\n";
	cout << "  (Steps per Trajectory): " << steps_per_trajectory << "\n";
	cout << "  (Trajectories per Batch): " << trajectories_per_batch << "\n";
	cout << "  (Optimizations per Batch): " << num_epochs << "\n";
	cout << "  (Discount Ratio): " << gamma << " \n";
	cout << "  (GAE Parameter): " << lamb << " \n";
	cout << "  (Clipping Parameter): " << epsilon << " \n";
	cout << "  (Start LR): " << rl_start_alpha << "\n";
	cout << "  (End LR): " << rl_end_alpha << " \n";
	cout << ")\n";
}

/**
* Initializes the python PPO agent
* @param The length of the state array
* @param The total number of agent steps taken per trajectory
* @param The number of trajectories per optimization batch
* @param The number of (state,reward) pairs per optimization step
* @param The number of optimization steps taken per optimization batch
* @param The discount ratio
* @param GAE lambda
* @param PPO clipping ratio
* @param Initial learning rate
* @param Learning rate exponential decay rate
* @param Path to previous agent to load
* @return PyObject pointer pointing at the initialized PPO agent on success, NULL on failure
*/
PyObject* init_agent(long num_states, long steps_per_trajectory, long trajectories_per_batch, long minibatch_size, long num_epochs, double gamma, double lamb, double epsilon, double alpha, double decay_rate, bool  load_agent, bool reset_stdev, const char* agent_path)
{
	// Define module name
	PyObject* name = PyUnicode_DecodeFSDefault("PPO_Agent_3_Output");

	// Initialize module
	PyObject* module = PyImport_Import(name);
	if (module == NULL)
	{
		fprintf(stderr, "\nFailed to find agent module.\n");
		PyErr_Print();
		Py_DECREF(name);
		return NULL;
	}
	Py_DECREF(name);

	// Load dictionary of module methods and variables
	PyObject* dict = PyModule_GetDict(module);
	if (dict == NULL)
	{
		fprintf(stderr, "\nFailed to load agent module dictionary.\n");
		PyErr_Print();
		Py_DECREF(module);
		return NULL;
	}
	Py_DECREF(module);

	// Get the initialization function from the module dictionary
	PyObject* init = PyDict_GetItemString(dict, "PPO_Agent");
	if (init == NULL || !PyCallable_Check(init))
	{
		fprintf(stderr, "\nFailed to find agent __init__ function.\n");
		PyErr_Print();
		Py_DECREF(dict);
		if (init != NULL) { Py_DECREF(init); }
		return NULL;
	}
	Py_DECREF(dict);

	// Convert load agent and reset stdev bool to Py pointer
	PyObject* py_load_agent;
	if (load_agent)
	{
		py_load_agent = PyLong_FromLong(1);
	}
	else
	{
		py_load_agent = PyLong_FromLong(0);
	}
	PyObject* py_reset_stdev;
	if (reset_stdev)
	{
		py_reset_stdev = PyLong_FromLong(1);
	}
	else
	{
		py_reset_stdev = PyLong_FromLong(0);
	}

	// Build the initialization arguments
	PyObject* init_args = PyTuple_New(13);
	PyTuple_SetItem(init_args, 0, PyLong_FromLong(num_states));
	PyTuple_SetItem(init_args, 1, PyLong_FromLong(steps_per_trajectory));
	PyTuple_SetItem(init_args, 2, PyLong_FromLong(trajectories_per_batch));
	PyTuple_SetItem(init_args, 3, PyLong_FromLong(minibatch_size));
	PyTuple_SetItem(init_args, 4, PyLong_FromLong(num_epochs));
	PyTuple_SetItem(init_args, 5, PyFloat_FromDouble(gamma));
	PyTuple_SetItem(init_args, 6, PyFloat_FromDouble(lamb));
	PyTuple_SetItem(init_args, 7, PyFloat_FromDouble(epsilon));
	PyTuple_SetItem(init_args, 8, PyFloat_FromDouble(alpha));
	PyTuple_SetItem(init_args, 9, PyFloat_FromDouble(decay_rate));
	PyTuple_SetItem(init_args, 10, py_load_agent);
	PyTuple_SetItem(init_args, 11, py_reset_stdev);
	PyTuple_SetItem(init_args, 12, PyUnicode_DecodeFSDefault(agent_path));
	Py_DECREF(py_load_agent);
	Py_DECREF(py_reset_stdev);

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
* Initializes the python autoencoder
* @param Initial learning rate
* @param Learning rate exponential decay rate
* @param The x dimension of the frames sent to the autoencoder
* @param The y dimension of the frames sent to the autoencoder
* @param Number of filters used in first convolutional layer
* @param Number of filters used in second convolutional layer
* @param Length of the 1D compressed array in autoencoder
* @param Number of layers at decoder output
* @param Number of frames stored before a single stochastic gradient descent step
* @param Objective fnc type (must be less than or equal to num_output_layers) 1->rebuilt temp, 2->rebuilt temp and front, 3->rebuilt temp, front, and cure
* @param Whether or not to load a previous AE
* @param Path to previous AE to load
* @return PyObject pointer pointing at the initialized autoencoder on success, NULL on failure
*/
PyObject* init_autoencoder(double start_alpha, double decay_rate, long x_dim, long y_dim, long num_filter_1, long num_filter_2, long encoded_size, long num_output_layers, long frame_buffer, long objective_fnc, bool load_autoencoder, const char* load_path)
{
	// Define module name
	PyObject* name = PyUnicode_DecodeFSDefault("Autoencoder");

	// Initialize module
	PyObject* module = PyImport_Import(name);
	if (module == NULL)
	{
		fprintf(stderr, "\nFailed to find autoencoder module:\n");
		PyErr_Print();
		Py_DECREF(name);
		return NULL;
	}
	Py_DECREF(name);

	// Load dictionary of module methods and variables
	PyObject* dict = PyModule_GetDict(module);
	if (dict == NULL)
	{
		fprintf(stderr, "\nFailed to load autoencoder module dictionary:\n");
		PyErr_Print();
		Py_DECREF(module);
		return NULL;
	}
	Py_DECREF(module);

	// Get the initialization function from the module dictionary
	PyObject* init = PyDict_GetItemString(dict, "Autoencoder");
	if (init == NULL || !PyCallable_Check(init))
	{
		fprintf(stderr, "\nFailed to find autoencoder __init__ function:\n");
		PyErr_Print();
		Py_DECREF(dict);
		if (init != NULL) { Py_DECREF(init); }
		return NULL;
	}
	Py_DECREF(dict);

	// Build the initialization arguments
	PyObject* init_args = PyTuple_New(10);
	PyTuple_SetItem(init_args, 0, PyFloat_FromDouble(start_alpha));
	PyTuple_SetItem(init_args, 1, PyFloat_FromDouble(decay_rate));
	PyTuple_SetItem(init_args, 2, PyLong_FromLong(x_dim));
	PyTuple_SetItem(init_args, 3, PyLong_FromLong(y_dim));
	PyTuple_SetItem(init_args, 4, PyLong_FromLong(num_filter_1));
	PyTuple_SetItem(init_args, 5, PyLong_FromLong(num_filter_2));
	PyTuple_SetItem(init_args, 6, PyLong_FromLong(encoded_size));
	PyTuple_SetItem(init_args, 7, PyLong_FromLong(num_output_layers));
	PyTuple_SetItem(init_args, 8, PyLong_FromLong(frame_buffer));
	PyTuple_SetItem(init_args, 9, PyLong_FromLong(objective_fnc));

	// Initialize autoencoder object
	PyObject* object = PyObject_CallObject(init, init_args);
	if (object == NULL)
	{
		fprintf(stderr, "\nFailed to call autoencoder __init__ function:\n");
		PyErr_Print();
		Py_DECREF(init);
		Py_DECREF(init_args);
		return NULL;
	}
	Py_DECREF(init);
	Py_DECREF(init_args);

	// Load previous AE at path
	if (load_autoencoder)
	{
		PyObject* load_result = PyObject_CallMethod(object, "load", "s", load_path);
		if (load_result == NULL)
		{
			fprintf(stderr, "\nFailed to call autoencoder load function:\n");
			PyErr_Print();
			return NULL;
		}
		Py_DECREF(load_result);
	}
	
	// return the class
	return object;
}

/**
* Saves MSE loss training data, last frame buffer, and trained autoencoder NN. Frees autoencoder
* @param Vector containing MSE training data
* @param Pointer to the trained autoencoder
* @return 0 on success, 1 on failure
*/
int save_autoencoder_training_data(vector<double> MSE_loss, PyObject* autoencoder)
{
	// Convert autoencoder training results
	cout << "\n\nConverting autoencoder results..." << endl;
	PyObject* py_MSE_loss = get_1D_list(MSE_loss);
	
	// Run save and display
	if (PyObject_CallMethod(autoencoder, "display_and_save", "O", py_MSE_loss) == NULL)
	{
		fprintf(stderr, "\nFailed to call display and save autoencoder function:\n");
		PyErr_Print();
		if (autoencoder != NULL) { Py_DECREF(autoencoder); }
		if (py_MSE_loss != NULL) { Py_DECREF(py_MSE_loss); }
		return 1;
	}
	
	// Free python memory
	Py_DECREF(autoencoder);
	Py_DECREF(py_MSE_loss);
	return 0;
}

/**
* Saves agent training data, best controlled trajectory information, and trained agent NN. Frees agent
* @return 0 on success, 1 on failure
*/
int save_agent_training_data(PyObject* trained_agent, vector<double> r_per_episode, vector<double> x_rate_stdev, vector<double> y_rate_stdev, vector<double> mag_stdev, vector<double> value_error,
vector<double> best_input_location_x, vector<double> best_input_location_y, vector<double> best_input_percent, vector<double> best_sim_time, vector<double> best_target, vector<vector<vector<double>>> best_temperature_field, 
vector<vector<vector<double>>> best_cure_field, vector<vector<vector<double>>> best_front_temperature, vector<vector<vector<double>>> best_front_location, vector<vector<vector<double>>> best_front_velocity, 
double best_episode, Finite_Element_Solver* FES, bool render)
{
	// Init save and render module
	cout << "\nConverting agent results..." << endl;
	PyObject* module_name = PyUnicode_DecodeFSDefault("Save_Render");
	PyObject* module = PyImport_Import(module_name);
	if (module == NULL)
	{
		fprintf(stderr, "\nFailed to find agent save and render module.\n");
		PyErr_Print();
		Py_DECREF(module_name);
		return 1;
	}
	PyObject* fnc = PyObject_GetAttrString(module,"Run");
	if (fnc == NULL || !PyCallable_Check(fnc))
	{
		fprintf(stderr, "\nFailed to find agent save and render function.\n");
		PyErr_Print();
		Py_DECREF(module_name);
		Py_DECREF(module);
		if (fnc != NULL) { Py_DECREF(fnc); }
		return 1;
	}
	Py_DECREF(module_name);
	Py_DECREF(module);
	
	// Convert results
	PyObject* py_r_per_episode = get_1D_list(r_per_episode);
	PyObject* py_x_rate_stdev = get_1D_list(x_rate_stdev);
	PyObject* py_y_rate_stdev = get_1D_list(y_rate_stdev);
	PyObject* py_mag_stdev = get_1D_list(mag_stdev);
	PyObject* py_value_error = get_1D_list(value_error);
	PyObject* py_best_input_location_x = get_1D_list(best_input_location_x);
	PyObject* py_best_input_location_y = get_1D_list(best_input_location_y);
	PyObject* py_best_input_percent = get_1D_list(best_input_percent);
	PyObject* py_best_sim_time = get_1D_list(best_sim_time);
	PyObject* py_best_target = get_1D_list(best_target);
	PyObject* py_best_temperature_field = get_3D_list(best_temperature_field);
	PyObject* py_best_cure_field = get_3D_list(best_cure_field);
	PyObject* py_best_front_temperature = get_3D_list(best_front_temperature);
	PyObject* py_best_front_location = get_3D_list(best_front_location);
	PyObject* py_best_front_velocity = get_3D_list(best_front_velocity);
	PyObject* py_best_episode = PyFloat_FromDouble(best_episode);
	PyObject* py_mesh_x_z0 =  get_2D_list(FES->get_mesh_x_z0());
	PyObject* py_mesh_y_z0 =  get_2D_list(FES->get_mesh_y_z0());
	PyObject* py_max_input_mag = PyFloat_FromDouble(FES->get_max_input_mag());
	PyObject* py_exp_const = PyFloat_FromDouble(FES->get_exp_const());
	PyObject* py_mesh_y_x0 =  get_2D_list(FES->get_mesh_y_x0());
	PyObject* py_mesh_z_x0 =  get_2D_list(FES->get_mesh_z_x0());
	PyObject* py_control_speed;
	if (FES->get_control_speed())
	{
		py_control_speed = PyLong_FromLong(1);
	}
	else
	{
		py_control_speed = PyLong_FromLong(0);
	}
	PyObject* py_render;
	if (render)
	{
		py_render = PyLong_FromLong(1);
	}
	else
	{
		py_render = PyLong_FromLong(0);
	}
	
	
	// Create args for run fucntion
	PyObject* args = PyTuple_New(25);
	PyTuple_SetItem(args, 0, trained_agent);
	PyTuple_SetItem(args, 1, py_r_per_episode);
	PyTuple_SetItem(args, 2, py_x_rate_stdev);
	PyTuple_SetItem(args, 3, py_y_rate_stdev);
	PyTuple_SetItem(args, 4, py_mag_stdev);
	PyTuple_SetItem(args, 5, py_value_error);
	PyTuple_SetItem(args, 6, py_best_input_location_x);
	PyTuple_SetItem(args, 7, py_best_input_location_y);
	PyTuple_SetItem(args, 8, py_best_input_percent);
	PyTuple_SetItem(args, 9, py_best_sim_time);
	PyTuple_SetItem(args, 10, py_best_target);
	PyTuple_SetItem(args, 11, py_best_temperature_field);
	PyTuple_SetItem(args, 12, py_best_cure_field);
	PyTuple_SetItem(args, 13, py_best_front_location);
	PyTuple_SetItem(args, 14, py_best_front_velocity);
	PyTuple_SetItem(args, 15, py_best_front_temperature);
	PyTuple_SetItem(args, 16, py_best_episode);
	PyTuple_SetItem(args, 17, py_mesh_x_z0);
	PyTuple_SetItem(args, 18, py_mesh_y_z0);
	PyTuple_SetItem(args, 19, py_max_input_mag);
	PyTuple_SetItem(args, 20, py_exp_const);
	PyTuple_SetItem(args, 21, py_mesh_y_x0);
	PyTuple_SetItem(args, 22, py_mesh_z_x0);
	PyTuple_SetItem(args, 23, py_control_speed);
	PyTuple_SetItem(args, 24, py_render);
	
	// Run save and render
	if ((args==NULL) || (PyObject_CallObject(fnc, args) == NULL))
	{
		fprintf(stderr, "\nFailed to call save and render function.\n");
		PyErr_Print();
		Py_DECREF(fnc);
		if (args != NULL) { Py_DECREF(args); }
		return 1;
	}
	
	// Free memory
	Py_DECREF(fnc);
	Py_DECREF(args);
	return 0;
}

/**
* Solves for the total number of frames to be rendered during a single trajectory
* @param the environment used for simulation
* @param Total number of trajectory steps taken between each frame capture
* @return number of frames to be rendered each trajectory
*/
int get_num_frames_per_trajectory(Finite_Element_Solver* FES, int steps_per_frame)
{
	int frames_per_trajectory = 0;
	int frame_index = 0;
	bool frame_done = false;
	while(!frame_done)
	{
		frame_done = (frame_index == (int)FES->get_target_vector_arr_size() - 1);
		if ((frame_index % steps_per_frame == 0) || (frame_index==0))
		{
			frames_per_trajectory++;
		}
		frame_index++;
	}
	
	return frames_per_trajectory;
}

/**
* Generates random set of frames to be used to update autoencoder
* @param total number of frames generated by 1 trajectory
* @param Number of frames taken each trajectory
* @return vector of update frame indices
*/
vector<int> get_update_frames(int tot_num_sim_steps, int samples_per_trajectory)
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
* Prints to stdout a readout of the current RL agent training process
* @param the current trajecotry being simulated
* @param the total number of trajectories to be simulated
* @param the total reward given for the previous trajectory
* @param number of RL agent actions taken per trajectory 
* @param vector containing episodic reward memory
* @param the best episodic reward to date
* @param vecotr containing all MSE losses
*/
void print_training_info(int curr_trajectory, int total_trajectories, double prev_episode_reward, int steps_per_trajectory, vector<double> r_per_episode, double best_episode, vector<double> MSE_loss)
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
	
	// Autoencoder MSE loss sub messege
	string msg6 = "| AE Loss: ";
	stream.str(string());
	if (MSE_loss.size() <= 0)
	{
		stream << fixed << setprecision(3) << 0.0;
	}
	else
	{
		stream << fixed << setprecision(3) << MSE_loss.back();
	}
	msg6.append(stream.str());
	if (msg6.length() < 18)
	{
		msg6.append(18 - msg6.length(), ' ');
	}
	
	// Print all sub messeges
	cout << msg1+msg2+msg3+msg4+msg5+msg6 << "|\r";
}

/**
* Runs a set of trajectories using the PPO policy, updates the PPO agent, and collects relevant training data
* @param The finite element solver object used to propogate time
* @param Whether to render the saved best trajectory
* @param The ppo agent that defines the policy
* @param The total number of trajectories to be executed
* @param The number of simulation steps taken per single agent cycle
* @param The number of simulation steps taken per single save frame
* @param The number of agent cycles steps in each trajecotry
* @param The autoencoder used to encode temperature state data
* @param Total number of steps per simulation
* @param Number of frames sampled each trajecotry by autoencoder
* @return 0 on success, 1 on failure
*/
int run(Finite_Element_Solver* FES, bool render, PyObject* agent, int total_trajectories, int steps_per_agent_cycle, int steps_per_frame, int steps_per_trajectory, PyObject* autoencoder, int tot_num_sim_steps, int samples_per_trajectory)
{

	// Determine how many frames there are in each trajectory
	int frames_per_trajectory = get_num_frames_per_trajectory(FES, steps_per_frame);

	// Agent training data storage
	vector<double> r_per_episode;
	vector<double> x_rate_stdev;
	vector<double> y_rate_stdev;
	vector<double> mag_stdev;
	vector<double> value_error;

	// Autoencoder training data storage
	vector<double> MSE_loss;

	// Best trajecotry data
	vector<double> best_input_location_x;
	vector<double> best_input_location_y;
	vector<double> best_input_percent;
	vector<double> best_sim_time;
	vector<double> best_target;
	vector<vector<vector<double>>> best_temperature_field(FES->get_num_vert_length(), vector<vector<double>>(FES->get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> best_cure_field(FES->get_num_vert_length(), vector<vector<double>>(FES->get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> best_front_location(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> best_front_velocity(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> best_front_temperature(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));

	// Current trajectory data
	vector<double> curr_input_location_x;
	vector<double> curr_input_location_y;
	vector<double> curr_input_percent;
	vector<double> curr_sim_time;
	vector<double> curr_target;
	vector<vector<vector<double>>> curr_temperature_field(FES->get_num_vert_length(), vector<vector<double>>(FES->get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> curr_cure_field(FES->get_num_vert_length(), vector<vector<double>>(FES->get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> curr_front_location(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> curr_front_velocity(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));
	vector<vector<vector<double>>> curr_front_temperature(FES->get_num_vert_width(), vector<vector<double>>(FES->get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));

	// Simulation set variables
	double total_reward = 0.0;
	double best_episode = 0.0;
	double prev_episode_reward = 0.0;

	// Run a set of episodes
	for (int i = 0; i < total_trajectories; i++)
	{
		// Initialize simulation variables
		bool done = false;
		double action_1=0.0, stdev_1=0.0, action_2=0.0, stdev_2=0.0, action_3=0.0, stdev_3=0.0, reward;
		bool run_agent, save_frame, update_encoder;
		int step_in_trajectory = 0;
		
		// Select random set of frames to be used to update autoencoder
		vector<int> update_frames = get_update_frames(tot_num_sim_steps, samples_per_trajectory);
		int curr_save_frame_ind = 0;
		int curr_save_frame = update_frames[curr_save_frame_ind];
		
		// User readout
		print_training_info(i, total_trajectories, prev_episode_reward, steps_per_trajectory, r_per_episode, best_episode, MSE_loss);
		prev_episode_reward = total_reward;

		// Reset environment
		FES->reset();

		// Simulation loop
		while (!done)
		{
			// Determine what to run this simulation step
			run_agent = (step_in_trajectory % steps_per_agent_cycle == 0) || (step_in_trajectory==0);
			save_frame = (step_in_trajectory % steps_per_frame == 0) || (step_in_trajectory==0);
			update_encoder = (step_in_trajectory == curr_save_frame);
			step_in_trajectory++;
			
			// Run the agent
			if (run_agent)
			{
				// Gather temperature state data
				vector<vector<double>> norm_temp_mesh = FES->get_norm_temp_mesh();
				PyObject* py_norm_temp_mesh = get_2D_list(norm_temp_mesh);
				
				// Encode and convert temperature state data
				PyObject* py_encoded_state = PyObject_CallMethod(autoencoder, "encode", "O", py_norm_temp_mesh);
				if (py_encoded_state == NULL)
				{
					fprintf(stderr, "\nFailed to call update autoencoder function.\n");
					PyErr_Print();
					Py_DECREF(py_norm_temp_mesh);
					return 1;
				}
				vector<double> encoded_state = get_vector(py_encoded_state);
				
				// Append input location and magnitude data to encoded state
				FES->append_input(&encoded_state);
				
				// Get agent action based on total encoded state
				PyObject* py_state = get_1D_list(encoded_state);
				PyObject* py_action = PyObject_CallMethod(agent, "get_action", "O", py_state);
				if (py_action == NULL)
				{
					fprintf(stderr, "\nFailed to call get action function.\n");
					PyErr_Print();
					Py_DECREF(py_norm_temp_mesh);
					Py_DECREF(py_encoded_state);
					Py_DECREF(py_state);
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

				// Update the agent
				reward = FES->get_reward();
				PyObject* py_result = PyObject_CallMethod(agent, "update_agent", "(O,f,f,f,f)", py_state, PyFloat_FromDouble(action_1), PyFloat_FromDouble(action_2), PyFloat_FromDouble(action_3), PyFloat_FromDouble(reward));
				if (py_result == NULL)
				{
					fprintf(stderr, "\nFailed to update agent\n");
					PyErr_Print();
					Py_DECREF(py_norm_temp_mesh);
					Py_DECREF(py_encoded_state);
					Py_DECREF(py_state);
					Py_DECREF(py_action);
					return 1;
				}
	
				// Update reward
				total_reward = total_reward + reward;
				
				// Release the python memory
				Py_DECREF(py_norm_temp_mesh);
				Py_DECREF(py_encoded_state);
				Py_DECREF(py_state);
				Py_DECREF(py_action);
				Py_DECREF(py_result);
			}
			else
			{
				// Step the environment based on the previously commanded action
				done = FES->step(action_1, action_2, action_3);
			}

			// Update the encoder
			if (update_encoder)
			{
				// Update which frame is to be saved next
				curr_save_frame_ind++;
				if (curr_save_frame_ind < samples_per_trajectory)
				{
					curr_save_frame = update_frames[curr_save_frame_ind];
				}
				
				// Collect frame data
				vector<vector<double>> norm_temp_mesh = FES->get_norm_temp_mesh();
				vector<vector<double>> cure_mesh = FES->get_cure_mesh();
				PyObject* py_norm_temp_mesh = get_2D_list(norm_temp_mesh);
				PyObject* py_cure_mesh = get_2D_list(cure_mesh);
						
				// Send frame data to autoencoder (it will automatically update when data buffer is full)
				PyObject* py_MSE_loss = PyObject_CallMethod(autoencoder, "update", "(O,O)", py_norm_temp_mesh, py_cure_mesh);
				if (py_MSE_loss == NULL)
				{
					fprintf(stderr, "\nFailed to call update autoencoder function.\n");
					PyErr_Print();
					Py_DECREF(py_norm_temp_mesh);
					Py_DECREF(py_cure_mesh);
					return 1;
				}
				
				// Store training data
				MSE_loss.push_back(PyFloat_AsDouble(py_MSE_loss));
				
				// Free python memory
				Py_DECREF(py_MSE_loss);
				Py_DECREF(py_norm_temp_mesh);
				Py_DECREF(py_cure_mesh);
			}

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
				curr_sim_time.push_back(FES->get_current_time());
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
			
		}

		// Update the best trajectory memory
		prev_episode_reward = total_reward - prev_episode_reward;
		if (prev_episode_reward > best_episode || i == 0)
		{
			best_episode = prev_episode_reward;
			best_input_location_x = curr_input_location_x;
			best_input_location_y = curr_input_location_y;
			best_input_percent = curr_input_percent;
			best_sim_time = curr_sim_time;
			best_target = curr_target;
			best_temperature_field = curr_temperature_field;
			best_cure_field = curr_cure_field;
			best_front_location = curr_front_location;
			best_front_velocity = curr_front_velocity;
			best_front_temperature = curr_front_temperature;
		}

		// Reset the current trajectory memory
		curr_input_location_x.clear();
		curr_input_location_y.clear();
		curr_input_percent.clear();
		curr_sim_time.clear();
		curr_target.clear();

		// Update the logs
		r_per_episode.push_back(prev_episode_reward/(double)steps_per_trajectory);
		x_rate_stdev.push_back(FES->get_loc_rate_scale()*stdev_1);
		y_rate_stdev.push_back(FES->get_loc_rate_scale()*stdev_2);
		mag_stdev.push_back(FES->get_max_input_mag()*FES->get_mag_scale()*stdev_3);

		// Final user readout
		if (i == total_trajectories - 1) { print_training_info(i, total_trajectories, prev_episode_reward, steps_per_trajectory, r_per_episode, best_episode, MSE_loss); }
	}

	// Gather the trajectory value error from the agent
	PyObject* py_value_error = PyObject_GetAttr(agent, PyUnicode_DecodeFSDefault("value_estimation_error"));
	if (py_value_error == NULL)
	{
		fprintf(stderr, "\nFailed to gather value estimation error training data\n");
		PyErr_Print();
		return 1;
	}
	value_error = get_vector(py_value_error);
	Py_DECREF(py_value_error);

	// Save autoencoder and autoencoder training data
	if (save_autoencoder_training_data(MSE_loss, autoencoder)==1) { return 1; }
	
	// Save agent and agent training data
	return save_agent_training_data(agent, r_per_episode, x_rate_stdev, y_rate_stdev, mag_stdev, value_error,
	best_input_location_x, best_input_location_y, best_input_percent, best_sim_time, best_target, best_temperature_field, 
	best_cure_field, best_front_temperature, best_front_location, best_front_velocity, best_episode, FES, render);
}

int main()
{	
	// Agent load parameters
	bool load_agent = false;
	bool reset_stdev = false;
	const char* agent_path = "results/PPO_1";
	
	// Agent training parameter
	int total_trajectories = 100;
	int steps_per_trajectory = 100;
	int trajectories_per_batch = 10;
	int num_epochs = 10;
	double rl_start_alpha = 1.0e-3;
	double rl_end_alpha = 5.0e-4;
	
	// Agent hyperparameters
	double gamma = 0.99;
	double lamb = 0.95;
	double epsilon = 0.20;

	// Autoencoder load parameters
	bool load_ae = true;
	const char* ae_path = "validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-2";
	
	// Autoencoder training parameters
	int samples_per_trajectory = 5;
	int samples_per_batch = 100;
	double ae_start_alpha = 5.0e-5;
	double ae_end_alpha = 1.0e-6;
	
	// Autoencoder NN parameters
	long num_filter_1 = 8;
	long num_filter_2 = 16;
	int encoded_size = 64;
	long num_output_layers = 3;
	long objective_fnc = 3;

	// Rendering parameters
	bool render = false;
	double frame_rate = 30.0;

	// Initialize FES
	Finite_Element_Solver FES = Finite_Element_Solver(encoded_size);

	// Calculated agent parameters
	double rl_decay_rate = pow(rl_end_alpha/rl_start_alpha, (double)trajectories_per_batch/(double)total_trajectories);
	int minibatch_size = (int) round(((double)trajectories_per_batch * (double)steps_per_trajectory) / (double)num_epochs);
	double agent_execution_period = (FES.get_sim_duration() / (double)steps_per_trajectory);
	int steps_per_agent_cycle = (int) round(agent_execution_period / FES.get_time_step());
	
	// Calculated autoencoder parameters
	int x_dim = FES.get_num_vert_length();
	int y_dim = FES.get_num_vert_width();
	double ae_decay_rate = pow(ae_end_alpha/ae_start_alpha, 1.0/((double)total_trajectories*(double)samples_per_trajectory));
	int tot_num_sim_steps = (int)floor(FES.get_sim_duration()/FES.get_time_step());

	// Calculated rendering parameters
	int steps_per_frame = (int) round(1.0 / (FES.get_time_step() * frame_rate));
	steps_per_frame = steps_per_frame <= 0 ? 1 : steps_per_frame;

	// Init py environment
	Py_Initialize();
	
	// Init agent
	PyObject* agent = init_agent(FES.get_num_state(), steps_per_trajectory, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, rl_start_alpha, rl_decay_rate, load_agent, reset_stdev, agent_path);
	if (agent == NULL) { Py_FinalizeEx(); return 1; }
	
	// Init autoencoder
	PyObject* autoencoder = init_autoencoder(ae_start_alpha, ae_decay_rate, x_dim, y_dim, num_filter_1, num_filter_2, encoded_size, num_output_layers, samples_per_batch, objective_fnc, load_ae, ae_path);
	if (autoencoder == NULL) { Py_DECREF(agent); Py_FinalizeEx(); return 1; }

	// Print parameters to stdout
	print_params(encoded_size, samples_per_trajectory, samples_per_batch, ae_start_alpha, ae_end_alpha, x_dim, y_dim, objective_fnc, steps_per_trajectory, trajectories_per_batch, num_epochs, gamma, lamb, epsilon, rl_start_alpha, rl_end_alpha);
	FES.print_params();

	// Train agent
	cout << "\nTraining agent..." << endl;
	auto start_time = chrono::high_resolution_clock::now();
	if (run(&FES, render, agent, total_trajectories, steps_per_agent_cycle, steps_per_frame, steps_per_trajectory, autoencoder, tot_num_sim_steps, samples_per_trajectory) == 1) { return 1; };

	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("\nTraining Took: %.1f seconds.\n", duration);
	
	// Close the py environment
	Py_FinalizeEx();
	cout << "Done!";
	return 0;
}
