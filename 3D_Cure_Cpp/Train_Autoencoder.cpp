#define PY_SSIZE_T_CLEAN
#include "Finite_Element_Solver.h"
#include <Python.h>
#include <string>
#include <iomanip>
#include <sstream>

using namespace std;

/**
* Prints the finite element solver and simulation parameters to std out
*/
void print_params(int encoder_output_size, int samples_per_trajectory, int samples_per_batch, double start_alpha, double end_alpha, int x_dim, int y_dim)
{
	// Hyperparameters
	cout << "\nHyperparameters(\n";
	cout << "  (Bottleneck): " << encoder_output_size << "\n";
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
* @param Whether or not a previous autoencoder's NN is loaded into the current autoencoder
* @return PyObject pointer pointing at the initialized autoencoder
*/
PyObject* init_autoencoder(double start_alpha, double decay_rate, long x_dim, long y_dim, long num_filter_1, long num_filter_2, long encoder_output_size, long num_output_layers, long frame_buffer, bool load_autoencoder)
{
	// Define module name
	PyObject* name = PyUnicode_DecodeFSDefault("Autoencoder");

	// Initialize module
	PyObject* module = PyImport_Import(name);
	if (module == NULL)
	{
		PyErr_Print();
		fprintf(stderr, "Failed to find autoencoder module.\n");
		Py_DECREF(name);
		Py_DECREF(module);
		return NULL;
	}
	Py_DECREF(name);

	// Load dictionary of module methods and variables
	PyObject* dict = PyModule_GetDict(module);
	if (dict == NULL)
	{
		PyErr_Print();
		fprintf(stderr, "Failed to load autoencoder module dictionary.\n");
		Py_DECREF(module);
		Py_DECREF(dict);
		return NULL;
	}
	Py_DECREF(module);

	// Get the initialization function from the module dictionary
	PyObject* init = PyDict_GetItemString(dict, "Autoencoder");
	if (init == NULL || !PyCallable_Check(init))
	{
		PyErr_Print();
		fprintf(stderr, "Failed to find autoencoder __init__ function.\n");
		Py_DECREF(dict);
		Py_DECREF(init);
		return NULL;
	}
	Py_DECREF(dict);

	// Convert load autoencoder bool to Py pointer
	PyObject* py_load_autoencoder;
	if (load_autoencoder)
	{
		py_load_autoencoder = PyLong_FromLong(1);
	}
	else
	{
		py_load_autoencoder = PyLong_FromLong(0);
	}

	// Build the initialization arguments
	PyObject* init_args = PyTuple_New(10);
	PyTuple_SetItem(init_args, 0, PyFloat_FromDouble(start_alpha));
	PyTuple_SetItem(init_args, 1, PyFloat_FromDouble(decay_rate));
	PyTuple_SetItem(init_args, 2, PyLong_FromLong(x_dim));
	PyTuple_SetItem(init_args, 3, PyLong_FromLong(y_dim));
	PyTuple_SetItem(init_args, 4, PyLong_FromLong(num_filter_1));
	PyTuple_SetItem(init_args, 5, PyLong_FromLong(num_filter_2));
	PyTuple_SetItem(init_args, 6, PyLong_FromLong(encoder_output_size));
	PyTuple_SetItem(init_args, 7, PyLong_FromLong(num_output_layers));
	PyTuple_SetItem(init_args, 8, PyLong_FromLong(frame_buffer));
	PyTuple_SetItem(init_args, 9, py_load_autoencoder);
	Py_DECREF(py_load_autoencoder);

	// Initialize autoencoder object
	PyObject* object = PyObject_CallObject(init, init_args);
	if (object == NULL)
	{
		PyErr_Print();
		fprintf(stderr, "Failed to call autoencoder __init__ function.\n");
		Py_DECREF(init);
		Py_DECREF(init_args);
		Py_DECREF(object);
		return NULL;
	}
	Py_DECREF(init);
	Py_DECREF(init_args);

	// return the class
	return object;
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
* @return Vector containing the training data
*/
vector<double> run(Finite_Element_Solver* FES, PyObject* autoencoder, int total_trajectories, int steps_per_cycle, int tot_num_sim_steps, int samples_per_trajectory)
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
					PyErr_Print();
					fprintf(stderr, "Failed to call update autoencoder function.\n");
					cin.get();
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
		if (i == total_trajectories - 1)
		{
			print_training_info(total_trajectories-1, total_trajectories, MSE_loss);
		}
	}

	// Return autoencoder and training data
	return MSE_loss;
}

int main()
{	
	// Autoencoder hyperparameters
	bool use_extended_state = false;
	int total_trajectories = 5000;
	long num_filter_1 = 12;
	long num_filter_2 = 12;
	int encoder_output_size = 64;
	long num_output_layers = 3;
	int samples_per_trajectory = 20;
	int samples_per_batch = 100;
	double start_alpha = 1.0e-3;
	double end_alpha = 1.0e-5;

	// Initialize FES
	Finite_Element_Solver FES = Finite_Element_Solver(encoder_output_size, use_extended_state);

	// Calculated parameters
	int x_dim = FES.get_num_vert_length();
	int y_dim = FES.get_num_vert_width();
	double decay_rate = pow(end_alpha/start_alpha, 1.0/((double)total_trajectories*(double)samples_per_trajectory));
	double execution_period = (FES.get_sim_duration() / (double)samples_per_trajectory);
	int steps_per_cycle = (int) round(execution_period / FES.get_time_step());
	int tot_num_sim_steps = (int)floor(FES.get_sim_duration()/FES.get_time_step());
	
	// Init autoencoder
	Py_Initialize();
	PyObject* autoencoder = init_autoencoder(start_alpha, decay_rate, x_dim, y_dim, num_filter_1, num_filter_2, encoder_output_size, num_output_layers, samples_per_batch, false);
	if (autoencoder == NULL)
	{
		PyErr_Print();
		fprintf(stderr, "Failed to initialize autoencoder\n");
		cin.get();
		return 1;
	}

	// Print simulation parameters
	print_params(encoder_output_size, samples_per_trajectory, samples_per_batch, start_alpha, end_alpha, x_dim, y_dim);
	FES.print_params();

	// Train autoencoder
	cout << "\nTraining autoencoder..." << endl;
	auto start_time = chrono::high_resolution_clock::now();
	vector<double> MSE_loss = run(&FES, autoencoder, total_trajectories, steps_per_cycle, tot_num_sim_steps, samples_per_trajectory);

	// Stop clock and print duration
	auto end_time = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>( end_time - start_time ).count();
	cout << "Training Took: ";
	printf("%.1f", (double)duration*10e-7);
	cout << " seconds.\n";
	
	// Convert training results
	cout << "\nConverting results..." << endl;
	PyObject* py_MSE_loss = get_1D_list(MSE_loss);
	
	// Run save and display
	if (PyObject_CallMethod(autoencoder, "display_and_save", "O", py_MSE_loss) == NULL)
	{
		PyErr_Print();
		fprintf(stderr, "Failed to call display autoencoder function.\n");
		cin.get();
		return 1;
	}
	
	// End main
	Py_DECREF(autoencoder);
	Py_DECREF(py_MSE_loss);
	Py_FinalizeEx();
	cout << "Done!";
	return 0;
}
