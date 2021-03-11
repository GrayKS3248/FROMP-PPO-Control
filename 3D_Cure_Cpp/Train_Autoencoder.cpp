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
void print_params(int encoder_output_size, int samples_per_trajectory, int samples_per_batch, double start_alpha, double end_alpha)
{
	// Hyperparameters
	cout << "\nHyperparameters(\n";
	cout << "  (Bottleneck): " << encoder_output_size << "\n";
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
* @param Length of the 1D compressed array in autoencoder
* @param Number of frames stored before a single stochastic gradient descent step
* @param Whether or not a previous autoencoder's NN is loaded into the current autoencoder
* @return PyObject pointer pointing at the initialized autoencoder
*/
PyObject* init_autoencoder(double start_alpha, double decay_rate, long x_dim, long y_dim, long encoder_output_size, long frame_buffer, bool load_autoencoder)
{
	// Declare PyObjects
	PyObject *name, *module, *dict, *init, *init_args, *object;

	// Define module name
	name = PyUnicode_DecodeFSDefault("Autoencoder");

	// Initialize module
	module = PyImport_Import(name);
	if (module == NULL)
	{
		PyErr_Print();
		fprintf(stderr, "Failed to find autoencoder module.\n");
		return NULL;
	}
	Py_DECREF(name);

	// Load dictionary of module methods and variables
	dict = PyModule_GetDict(module);
	if (dict == NULL)
	{
		PyErr_Print();
		fprintf(stderr, "Failed to load autoencoder module dictionary.\n");
		return NULL;
	}
	Py_DECREF(module);

	// Get the initialization function from the module dictionary
	init = PyDict_GetItemString(dict, "Autoencoder");
	if (init == NULL || !PyCallable_Check(init))
	{
		PyErr_Print();
		fprintf(stderr, "Failed to find autoencoder __init__ function.\n");
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
	init_args = PyTuple_New(7);
	PyTuple_SetItem(init_args, 0, PyFloat_FromDouble(start_alpha));
	PyTuple_SetItem(init_args, 1, PyFloat_FromDouble(decay_rate));
	PyTuple_SetItem(init_args, 2, PyLong_FromLong(x_dim));
	PyTuple_SetItem(init_args, 3, PyLong_FromLong(y_dim));
	PyTuple_SetItem(init_args, 4, PyLong_FromLong(encoder_output_size));
	PyTuple_SetItem(init_args, 5, PyLong_FromLong(frame_buffer));
	PyTuple_SetItem(init_args, 6, py_load_autoencoder);
	Py_DECREF(py_load_autoencoder);

	// Initialize autoencoder object
	object = PyObject_CallObject(init, init_args);
	if (object == NULL)
	{
		PyErr_Print();
		fprintf(stderr, "Failed to call autoencoder __init__ function.\n");
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
PyObject* get_1D_list(vector<double> state)
{
	PyObject *list = PyList_New(state.size());
	for (unsigned int i = 0; i < state.size(); i++)
	{
		PyList_SetItem(list, i, PyFloat_FromDouble(state[i]));
	}
	return list;
}

/**
* Converts 2D vector<vector<double>> to a 2D PyList
* @param The vector used to create list
* @return PyObject pointer pointing at the created list
*/
PyObject* get_2D_list(vector<vector<double>> state)
{
	PyObject *list = PyList_New(0);
	PyObject *inner_list = PyList_New(0);
	for (unsigned int i = 0; i < state.size(); i++)
	{
		for (unsigned int j = 0; j < state[0].size(); j++)
		{
			PyList_Append(inner_list, PyFloat_FromDouble(state[i][j]));
		}
		PyList_Append(list, inner_list);
		inner_list = PyList_New(0);
	}
	
	return list;
}

/**
* Runs a set of trajectories using a random policy and simultaneously trains and autoencoder to create a reduced state representation
* @param The finite element solver object used to propogate time
* @param The autoencoder that is being trained
* @param The total number of trajectories to be executed
* @param The number of simulation steps taken per single cycle of control application
* @param How many frames are to be gathered from each trajectory
* @return Tuple containing the training data and the trained autoencoder
*/
auto run(Finite_Element_Solver FES, PyObject* autoencoder, int total_trajectories, int steps_per_cycle, int samples_per_trajectory)
{

	// Declare variable tracking MSE_loss during training
	vector<double> MSE_loss;

	// Simulation set variables
	double percent_complete = 0.0;

	// Run a set of episodes
	for (int i = 0; i < total_trajectories; i++)
	{
		// User readout parameters
		percent_complete = total_trajectories > 1 ? 100.0 * i / (total_trajectories-1) : 0.0;

		// User readout
		stringstream stream;
		stream << std::fixed << std::setprecision(1) << percent_complete;
		string msg1 = stream.str();
		msg1.append("% Complete");
		if (msg1.length() < 18)
		{
			msg1.append(18 - msg1.length(), ' ');
		}
		string msg2 = "";
		msg2.append("| Traj: " + to_string(i+1) + "/" + to_string(total_trajectories));
		if (msg2.length() < 22)
		{
			msg2.append(22 - msg2.length(), ' ');
		}
		string msg3 = "| MSE Loss: ";
		stream.str(std::string());
		if (i != 0)
		{
			stream << std::fixed << std::setprecision(3) << MSE_loss.back();
		}
		else
		{
			stream << std::fixed << std::setprecision(3) << 0.0;
		}
		msg3.append(stream.str());
		if (msg3.length() < 20)
		{
			msg3.append(20 - msg3.length(), ' ');
		}
		cout << msg1+msg2+msg3 << "|\r";

		// Reset environment
		vector<double> state = FES.reset();

		// Declare simulation variables
		bool done = false;
		double action_1=0.0, action_2=0.0, action_3=0.0;
		bool apply_control;
		int step_in_trajectory = 0;
		
		// Select a random set of samples_per_trajectory frames to send to autoencoder
		vector<int> all_frames;
		for (int i = 0; i < steps_per_cycle*samples_per_trajectory; i++)
		{
			all_frames.push_back(i);
		}
		random_shuffle (all_frames.begin(), all_frames.end());
		vector<int> save_frames;
		for (int i = 0; i < samples_per_trajectory; i++)
		{
			save_frames.push_back(all_frames[i]);
		}
		sort (save_frames.begin(), save_frames.end());
		int curr_save_frame_ind = 0;
		int curr_save_frame = save_frames[curr_save_frame_ind];

		// Simulation for loop
		while (!done)
		{
			// Determine what to run this simulation step
			apply_control = (step_in_trajectory % steps_per_cycle == 0) || (step_in_trajectory==0);

			// Run the random controller
			if (apply_control)
			{

				// Get a random action
				action_1 = 20.0 * (2.0 * ((double)rand()/(double)RAND_MAX - 0.5));
				action_2 = 20.0 * (2.0 * ((double)rand()/(double)RAND_MAX - 0.5));
				action_3 = 20.0 * (2.0 * ((double)rand()/(double)RAND_MAX - 0.5));

				// Step the environment
				done = FES.step(action_1, action_2, action_3);
			}
			else
			{
				// Step the environment
				done = FES.step(action_1, action_2, action_3);
			}
			if (step_in_trajectory == curr_save_frame)
			{
				// Update which frame is to be saved next
				curr_save_frame_ind++;
				curr_save_frame = save_frames[curr_save_frame_ind];	
				
				// Collect frame data
				vector<vector<double> > norm_temp_mesh = FES.get_norm_temp_mesh();
				PyObject* py_norm_temp_mesh = get_2D_list(norm_temp_mesh);
								
				// Send frame data to autoencoder (it will automatically update when data buffer is full)
				PyObject* py_MSE_loss = PyObject_CallMethod(autoencoder, "update", "O", py_norm_temp_mesh);
				if (py_MSE_loss == NULL)
				{
					PyErr_Print();
					fprintf(stderr, "Failed to call update autoencoder function.\n");
					cin.get();
				}
				MSE_loss.push_back(PyFloat_AsDouble(py_MSE_loss));
				Py_DECREF(py_MSE_loss);
				Py_DECREF(py_norm_temp_mesh);
			}

			// Update the current state and the step in episode
			step_in_trajectory += 1;

		}

		// Final user readout
		if (i == total_trajectories - 1)
		{
			msg1 = "";
			msg1.append("100.0% Complete   ");
			msg2 = "| Traj: ";
			msg2.append(to_string(i+1) + "/" + to_string(total_trajectories));
			if (msg2.length() < 22)
			{
				msg2.append(22 - msg2.length(), ' ');
			}
			msg3 = "| MSE Loss: ";
			stream.str(std::string());
			stream << std::fixed << std::setprecision(3) << MSE_loss.back();
			msg3.append(stream.str());
			if (msg3.length() < 20)
			{
				msg3.append(20 - msg3.length(), ' ');
			}
			cout << msg1+msg2+msg3 << "|\n";
		}
	}

	// Return autoencoder and training data
	auto out = make_tuple(autoencoder, MSE_loss);
	return out;
}

int main()
{
	// Autoencoder hyperparameters
	bool load_autoencoder = false;
	int encoder_output_size = 100;
	int total_trajectories = 5000;
	int samples_per_trajectory = 20;
	int samples_per_batch = 100;
	double start_alpha = 1.0e-3;
	double end_alpha = 5.0e-5;

	// Initialize FES
	Finite_Element_Solver FES = Finite_Element_Solver();

	// Calculated parameters
	double decay_rate = pow(end_alpha/start_alpha, 1.0/((double)total_trajectories*(double)samples_per_trajectory));
	double execution_period = (FES.get_sim_duration() / (double)samples_per_trajectory);
	int steps_per_cycle = (int) round(execution_period / FES.get_time_step());

	// Init autoencoder
	Py_Initialize();
	PyObject* autoencoder = init_autoencoder(start_alpha, decay_rate, (int)ceil((double)FES.get_num_vert_length()/2.0), (int)ceil((double)FES.get_num_vert_width()/2.0), encoder_output_size, samples_per_batch, load_autoencoder);
	if (autoencoder == NULL)
	{
		PyErr_Print();
		fprintf(stderr, "Failed to initialize autoencoder\n");
		cin.get();
		return 1;
	}

	// Print simulation parameters
	print_params(encoder_output_size, samples_per_trajectory, samples_per_batch, start_alpha, end_alpha);
	FES.print_params();

	// Train autoencoder
	cout << "\nTraining autoencoder..." << endl;
	auto start_time = std::chrono::high_resolution_clock::now();
	auto [trained_autoencoder, MSE_loss] = run(FES, autoencoder, total_trajectories, steps_per_cycle, samples_per_trajectory);

	// Stop clock and print duration
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time ).count();
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
	cin.get();
	return 0;
}
