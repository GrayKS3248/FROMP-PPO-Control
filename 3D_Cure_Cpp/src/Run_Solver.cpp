#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "Finite_Difference_Solver.h"

using namespace std;


//******************************************************************** CONFIGURATION FUNCTIONS ********************************************************************//
/**
* Loads parameters from .cfg file
* @return 0 on success, 1 on failure
*/
int load_config(int& input_actions_per_simulation, double& frame_rate)
{
	// Load from config file
	ifstream config_file;
	config_file.open("../config_files/run_solver.cfg");
	string config_dump;
	if (config_file.is_open())
	{
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> input_actions_per_simulation;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> config_dump >> frame_rate;
	}
	else
	{
		cout << "Unable to open ../config_files/run_solver.cfg." << endl;
		return 1;
	}
	config_file.close();
	return 0;
}

/**
* Saves copy of original fds and run_solver config to string
* @param String to which config is saved
* @return 0 on success, 1 on failure
*/
int save_config(string& configs_string)
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
	file_in.open("../config_files/run_solver.cfg", ofstream::in);
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
		cout << "Unable to open ../config_files/run_solver.cfg." << endl;
		return 1;
	}
	
	// Close the file
	file_in.close();
	
	return 0;
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
* Calls the save, plot, and render functions of the save_render_plot class in the PPO module
* @param pointer to the save_render_plot class in the PPO module
* @return 0 on success, 1 on failure
*/
int save_results(PyObject* save_render_plot)
{
	// Save
	if(PyObject_CallMethodObjArgs(save_render_plot, PyUnicode_DecodeFSDefault("save_without_agent"), NULL) == NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's save function:\n");
		PyErr_Print();
		if (save_render_plot != NULL) { Py_DECREF(save_render_plot); }
		return 1;
	}
	
	// Plot
	if(PyObject_CallMethodObjArgs(save_render_plot, PyUnicode_DecodeFSDefault("plot"), NULL) == NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's plot function:\n");
		PyErr_Print();
		if (save_render_plot != NULL) { Py_DECREF(save_render_plot); }
		return 1;
	}
	
	// Render
	if(PyObject_CallMethodObjArgs(save_render_plot, PyUnicode_DecodeFSDefault("render"), NULL) == NULL)
	{
		fprintf(stderr, "\nFailed to call Save_Plot_Render's render function:\n");
		PyErr_Print();
		if (save_render_plot != NULL) { Py_DECREF(save_render_plot); }
		return 1;
	}
	
	// Free memory
	Py_DECREF(save_render_plot);
	return 0;
}


//******************************************************************** TRAINING LOOP ********************************************************************//
/**
* Runs a single FROMP simulation and collects simualtion data
* @param The finite element solver object used to propogate time
* @param The save render and plotting class of the ppo agent being trained
* @param The number of simulation steps taken per single input cycle
* @param The number of simulation steps taken per single render frame
* @param The number of simulation steps taken between each progress update to user
* @param The time at which simulation began
* @param String containing fds and run_solver config values
* @return 0 on success, 1 on failure
*/
int run(Finite_Difference_Solver* FDS, PyObject* save_render_plot, int steps_per_input_cycle, int steps_per_frame, int steps_per_progress_update, auto &start_time, string& configs_string )
{
	// Trajectory data
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
	vector<vector<double>> front_fit;
	vector<vector<vector<double>>> temperature_field;
	vector<vector<vector<double>>> cure_field;
	vector<vector<vector<double>>> fine_temperature_field;
	vector<vector<vector<double>>> fine_cure_field;

	// Initialize simulation variables
	vector<double> input_state;
	bool done = false;
	double action_1=0.0, action_2=0.0, action_3=0.0;
	bool print_progress, run_input, save_frame;
	int step_in_trajectory = 0;

	// Reset environment
	FDS->reset();

	// Simulation loop
	while (!done)
	{
		// Determine what to run this simulation step
		run_input = (step_in_trajectory % steps_per_input_cycle == 0) || (step_in_trajectory==0);
		save_frame = (step_in_trajectory % steps_per_frame == 0) || (step_in_trajectory==0);
		print_progress = (step_in_trajectory % steps_per_progress_update == 0) || (step_in_trajectory==0);
		step_in_trajectory++;
		
		// Print progress
		if(print_progress)
		{
			FDS->print_progress(true);
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
		
		// Step the environment 
		if (run_input)
		{	
			// Generate random actions from -1.0 to 1.0
			action_1 = 2.0*((double)rand()/(double)RAND_MAX - 0.5);
			action_2 = 2.0*((double)rand()/(double)RAND_MAX - 0.5);
			action_3 = 2.0*((double)rand()/(double)RAND_MAX - 0.5);
			
			// Step the environment given the random actions
			done = FDS->step(action_1, action_2, true, action_3);
		}
		else
		{
			// Step the environment based on the previously commanded actions
			done = FDS->step(action_1, action_2, true, action_3);
		}
		
	}
	
	// Final progress report
	FDS->print_progress(false);
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("Simulation took: %.1f seconds.\n\nConverting simulation results...", duration);
	
	// Send all relevant data to save render and plot module
	start_time = chrono::high_resolution_clock::now();
	if(store_input_history(save_render_plot, input_location_x, input_location_y, input_percent) == 1) {return 1;}
	if(store_field_history(save_render_plot, temperature_field, cure_field, fine_temperature_field, fine_cure_field, fine_mesh_loc) == 1) {return 1;}
	if(store_front_history(save_render_plot, front_curve, front_fit, front_velocity, front_temperature, front_shape_param) == 1) {return 1;}
	if(store_target_and_time(save_render_plot, target, time, reward) == 1) {return 1;}
	if(store_top_mesh(save_render_plot, FDS->get_coarse_x_mesh_z0(), FDS->get_coarse_y_mesh_z0()) == 1) {return 1;}
	if(store_input_params(save_render_plot, FDS->get_peak_input_mag(), FDS->get_input_const()) == 1) {return 1;}
	if(store_options(save_render_plot, FDS->get_control_mode(), true, configs_string) == 1) {return 1;}

	// Stop clock and print duration
	duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("\nData conversion took: %.1f seconds.", duration);

	// Save, plot, and render
	start_time = chrono::high_resolution_clock::now();
	return save_results(save_render_plot);
}


//******************************************************************** MAIN LOOP ********************************************************************//
int main()
{	
	// Load parameters
	int input_actions_per_simulation;
	double frame_rate;
	if (load_config(input_actions_per_simulation, frame_rate) == 1) { cin.get(); return 1; }

	// Save config values
	string configs_string;
	if (save_config(configs_string) == 1) 
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

	// Calculated input parameters
	double input_execution_period = (FDS->get_sim_duration() / (double)input_actions_per_simulation);
	int steps_per_input_cycle = (int) round(input_execution_period / FDS->get_coarse_time_step());

	// Calculated rendering parameters
	int steps_per_frame = (int) round(1.0 / (FDS->get_coarse_time_step() * frame_rate));
	steps_per_frame = steps_per_frame <= 0 ? 1 : steps_per_frame;
	
	// Calculated user interface parameters
	int steps_per_progress_update = (int) round(1.0 / (FDS->get_coarse_time_step() * (100.0 / FDS->get_sim_duration())));
	steps_per_progress_update = steps_per_progress_update <= 0 ? 1 : steps_per_progress_update;

	// Init py environment
	Py_Initialize();
	PyRun_SimpleString("import  sys");
	PyRun_SimpleString("sys.path.append('../py_src/')");

	// Init save_render_plot
	PyObject* save_render_plot = init_save_render_plot();
	if (save_render_plot == NULL) 
	{ 
		Py_FinalizeEx(); 
		cin.get(); 
		return 1; 
	}

	// Print parameters to stdout
	FDS->print_params();

	// Run simulation
	cout << "\nSimulating...\n";
	auto start_time = chrono::high_resolution_clock::now();
	if (run(FDS, save_render_plot, steps_per_input_cycle, steps_per_frame, steps_per_progress_update, start_time, configs_string) == 1) 
	{ 
		Py_FinalizeEx(); 
		cin.get(); 
		return 1; 
	};
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("Saving and Rendering Took: %.1f seconds.\n\nDone!", duration);
	
	// Finish
	Py_FinalizeEx();
	cin.get();
	return 0;
}
