#define PY_SSIZE_T_CLEAN
#include "Finite_Element_Solver.h"
#include <Python.h>
#include <string>
#include <iomanip>
#include <sstream>

using namespace std;

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
* Prints the finite element solver and simulation parameters to std out
* @param size of the encoder bottleneck
* @param number of autoencoder frames saved per trajecotry
* @param number of frames per autoencoder optimization epoch
* @param starting learning rate of autoencoder
* @param ending learning rate of autoencoder
* @param x dimension of images sent to autoencoder
* @param y dimension of images sent to autoencoder
* @param the objective function used to update the autoencoder
*/
void print_params(int encoded_size, int samples_per_trajectory, int samples_per_batch, double start_alpha, double end_alpha, int x_dim, int y_dim, long objective_fnc)
{
	// Hyperparameters
	cout << "\nHyperparameters(\n";
	cout << "  (Objective Fnc): " << objective_fnc << "\n";
	cout << "  (Bottleneck): " << encoded_size << "\n";
	cout << "  (Image Dimenstions): " << x_dim << " x " << y_dim << "\n";
	cout << "  (Samples per Trajectory): " << samples_per_trajectory << "\n";
	cout << "  (Samples per Batch): " << samples_per_batch << " \n";
	cout << "  (Start LR): " << start_alpha << "\n";
	cout << "  (End LR): " << end_alpha << " \n";
	cout << ")\n";
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
* @param vector containing all MSE losses
*/
void print_training_info(int curr_trajectory, int total_trajectories, vector<double> MSE_loss)
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
	
	// Autoencoder MSE loss sub messege
	string msg3 = "| MSE Loss: ";
	stream.str(string());
	if (MSE_loss.size() <= 0)
	{
		stream << fixed << setprecision(3) << 0.0;
	}
	else
	{
		stream << fixed << setprecision(3) << MSE_loss.back();
	}
	msg3.append(stream.str());
	if (msg3.length() < 19)
	{
		msg3.append(19 - msg3.length(), ' ');
	}
	
	// Print all sub messeges
	cout << msg1+msg2+msg3 << "|\r";
}

/**
* Runs a set of trajectories using a random policy and simultaneously trains and autoencoder to create a reduced state representation
* @param The finite element solver object used to propogate time
* @param The autoencoder that is being trained
* @param The total number of trajectories to be executed
* @param The number of simulation steps taken per single cycle of control application
* @return 0 on success, 1 on failure
*/
int run(Finite_Element_Solver* FES, PyObject* autoencoder, int total_trajectories, int steps_per_cycle, int tot_num_sim_steps, int samples_per_trajectory)
{

	// Declare variable tracking MSE_loss during training
	vector<double> MSE_loss;

	// Run a set of episodes
	for (int i = 0; i < total_trajectories; i++)
	{
		// Declare simulation variables
		bool done = false;
		double action_1=0.0, action_2=0.0, action_3=0.0;
		bool apply_control, update_encoder;
		int step_in_trajectory = 0;
		
		// Select random set of frames to be used to update autoencoder
		vector<int> update_frames = get_update_frames(tot_num_sim_steps, samples_per_trajectory);
		int curr_save_frame_ind = 0;
		int curr_save_frame = update_frames[curr_save_frame_ind];

		// User readout
		print_training_info(i, total_trajectories, MSE_loss);

		// Reset environment
		FES->reset();
		
		// Simulation for loop
		while (!done)
		{
			// Determine what to run this simulation step
			apply_control = (step_in_trajectory % steps_per_cycle == 0) || (step_in_trajectory==0);
			update_encoder = step_in_trajectory == curr_save_frame;

			// Run the random controller
			if (apply_control)
			{

				// Get a random action
				action_1 = 20.0 * (2.0 * ((double)rand()/(double)RAND_MAX - 0.5));
				action_2 = 20.0 * (2.0 * ((double)rand()/(double)RAND_MAX - 0.5));
				action_3 = 20.0 * (2.0 * ((double)rand()/(double)RAND_MAX - 0.5));

				// Step the environment
				done = FES->step(action_1, action_2, action_3);
			}
			else
			{
				// Step the environment
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

			// Update the current state and the step in episode
			step_in_trajectory++;

		}

		// Final user readout
		if (i == total_trajectories - 1) { print_training_info(i, total_trajectories, MSE_loss); }
	}

	// Save autoencoder and autoencoder training data
	return save_autoencoder_training_data(MSE_loss, autoencoder);
}

int main()
{	
	// Autoencoder load parameters
	bool load = true;
	const char* path = "validation/DCPD_GC2_Autoencoder/0%_Cropped/1-8-16_64_aux-2";
	
	// Autoencoder training parameters
	int total_trajectories = 2;
	int samples_per_trajectory = 20;
	int samples_per_batch = 100;
	double start_alpha = 1.0e-3;
	double end_alpha = 1.0e-5;
	
	// Autoencoder NN parameters
	long num_filter_1 = 8;
	long num_filter_2 = 16;
	int encoded_size = 64;
	long num_output_layers = 3;
	long objective_fnc = 3;

	// Initialize FES
	Finite_Element_Solver FES = Finite_Element_Solver(encoded_size);

	// Calculated parameters
	int x_dim = FES.get_num_vert_length();
	int y_dim = FES.get_num_vert_width();
	double decay_rate = pow(end_alpha/start_alpha, 1.0/((double)total_trajectories*(double)samples_per_trajectory));
	double execution_period = (FES.get_sim_duration() / (double)samples_per_trajectory);
	int steps_per_cycle = (int) round(execution_period / FES.get_time_step());
	int tot_num_sim_steps = (int)floor(FES.get_sim_duration()/FES.get_time_step());
	
	// Init py environment
	Py_Initialize();
	
	// Init autoencoder
	PyObject* autoencoder = init_autoencoder(start_alpha, decay_rate, x_dim, y_dim, num_filter_1, num_filter_2, encoded_size, num_output_layers, samples_per_batch, objective_fnc, load, path);
	if (autoencoder == NULL) { Py_FinalizeEx(); return 1; }

	// Print simulation parameters
	print_params(encoded_size, samples_per_trajectory, samples_per_batch, start_alpha, end_alpha, x_dim, y_dim, objective_fnc);
	FES.print_params();

	// Train autoencoder
	cout << "\nTraining autoencoder...\n";
	auto start_time = chrono::high_resolution_clock::now();
	if (run(&FES, autoencoder, total_trajectories, steps_per_cycle, tot_num_sim_steps, samples_per_trajectory) == 1) { return 1; }

	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("\nTraining Took: %.1f seconds.\n", duration);

	// Close the py environment
	Py_FinalizeEx();
	cout << "Done!";
	return 0;
}
