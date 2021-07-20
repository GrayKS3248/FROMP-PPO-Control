#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "Finite_Difference_Solver.h"

using namespace std;


//******************************************************************** CONFIGURATION FUNCTIONS ********************************************************************//
/**
* Loads parameters from .cfg file
* @return 0 on success, 1 on failure
*/
int load_config(vector<string>& name_list, vector<string>& initial_cure_list, vector<string>& initial_temp_list, vector<string>& front_speed_list, vector<double>& params)
{
	// Load from config file
	ifstream config_file;
	config_file.open("config_files/tune_params.cfg");
	string string_dump;
	if (config_file.is_open())
	{
		// Simulation parameters
		config_file.ignore(numeric_limits<streamsize>::max(), '}');
		config_file >> string_dump >> params[0];
		
		// Tuning parameter limits
		config_file.ignore(numeric_limits<streamsize>::max(), '}');
		config_file >> string_dump >> params[1];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[2];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[3];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[4];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[5];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[6];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[7];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[8];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[9];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[10];
		
		// Fitness parameters
		config_file.ignore(numeric_limits<streamsize>::max(), '}');
		config_file >> string_dump >> params[11];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[12];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[13];
		
		// Search parameters
		config_file.ignore(numeric_limits<streamsize>::max(), '}');
		config_file >> string_dump >> params[14];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[15];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[16];
		
		// Termination parameters
		config_file.ignore(numeric_limits<streamsize>::max(), '}');
		config_file >> string_dump >> params[17];
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> params[18];
		
		while(true)
		{
			config_file.ignore(numeric_limits<streamsize>::max(), '}');
			
			if(config_file.eof())
			{
				break;
			}
			
			while(true)
			{
				config_file >> string_dump;
				if(string_dump.compare("{END_SET}")==0)
				{
					break;
				}
				else
				{
					name_list.push_back(string_dump);
				
					config_file >> string_dump;
					initial_cure_list.push_back(string_dump);
					
					config_file >> string_dump;
					initial_temp_list.push_back(string_dump);
					
					config_file >> string_dump;
					front_speed_list.push_back(string_dump);
				}
			}
		}
	}
	else
	{
		cout << "Unable to open config_files/tune_params.cfg." << endl;
		return 1;
	}
	config_file.close();
	
	return 0;
}

/**
* Copies original fds config file data to fds_orig.cfg file
* @return 0 on success, 1 on failure
*/
int make_orig_fds_config()
{
	ifstream fds_file_in;
	ofstream fds_file_out;
	fds_file_in.open("config_files/fds.cfg", std::ofstream::in);
	fds_file_out.open("config_files/fds_temp.cfg", std::ofstream::out | std::ofstream::trunc );
	string string_dump;
	if (fds_file_in.is_open() && fds_file_out.is_open())
	{
		// Copy lines
		while(getline(fds_file_in, string_dump))
		{
			fds_file_out << string_dump << "\n";
		}
	}
	else
	{
		cout << "Unable to open config_files/fds.cfg." << endl;
		return 1;
	}
	
	fds_file_in.close();
	fds_file_out.close();
	
	// Rename files
	char old_name_1[] = "config_files/fds.cfg";
	char new_name_1[] = "config_files/fds_orig.cfg";
	char old_name_2[] = "config_files/fds_temp.cfg";
	char new_name_2[] = "config_files/fds.cfg";
	int result = rename(old_name_1, new_name_1) + rename(old_name_2, new_name_2);
	if(result == 0)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

/**
* Edits the FDS config file to run the commanded monomer, initial cure, initial temperature, and target front speed
* @param string of monomoer name
* @param string of initial cure
* @param string of initial temperature 
* @param string of target front speed
* @return 0 on success, 1 on failure
*/
int edit_fds_config(string name, string initial_cure, string initial_temp, string front_speed)
{
	ifstream fds_file_in;
	ofstream fds_file_out;
	fds_file_in.open("config_files/fds_orig.cfg", std::ofstream::in);
	fds_file_out.open("config_files/fds.cfg");
	string string_dump;
	string write_string;
	
	// Set use_input, use_trigger, material, control, target_type, target_speed, tar_speed_dev, tar_temp_dev, initial_temp, temp_deviation, initial_cure, cure_deviation, mean_htc, htc_dev, amb_temp_dev, trigger_len
	if (fds_file_in.is_open() && fds_file_out.is_open())
	{
		
		while(getline(fds_file_in, string_dump))
		{
			if( string_dump.find("use_input") != string::npos )
			{
				write_string = "use_input\tfalse\t[true/false]";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("use_trigger") !=  string::npos )
			{
				write_string = "use_trigger\ttrue\t[true/false]";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("material") !=  string::npos )
			{
				write_string = "material\t"+name+"\t[dcpd_gc1/dcpd_gc2/cod]";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("control") !=  string::npos )
			{
				write_string = "control\tspeed\t[speed/temp]";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("target_type") !=  string::npos )
			{
				write_string = "target_type\tconst\t[const/rand/switch]";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("target_speed") !=  string::npos )
			{
				write_string = "target_speed\t"+front_speed+"\t(Meters / Second)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("tar_speed_dev") !=  string::npos )
			{
				write_string = "tar_speed_dev\t0.0\t(Meters / Second)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("tar_temp_dev") !=  string::npos )
			{
				write_string = "tar_temp_dev\t0.0\t(Kelvin)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("initial_temp") !=  string::npos )
			{
				write_string = "initial_temp\t"+initial_temp+"\t(Kelvin)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("temp_deviation") !=  string::npos )
			{
				write_string = "temp_deviation\t0.0\t(Kelvin)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("initial_cure") !=  string::npos )
			{
				write_string = "initial_cure\t"+initial_cure+"\t(Decimal Percent)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("cure_deviation") !=  string::npos )
			{
				write_string = "cure_deviation\t0.0\t(Decimal Percent)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("mean_htc") !=  string::npos )
			{
				write_string = "mean_htc\t0.0\t(Watts / (Meter ^ 2 * Kelvin))";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("htc_dev") !=  string::npos )
			{
				write_string = "htc_dev\t0.0\t(Watts / (Meter ^ 2 * Kelvin))";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("amb_temp_dev") !=  string::npos )
			{
				write_string = "amb_temp_dev\t0.0\t(Kelvin)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("trigger_len") !=  string::npos )
			{
				write_string = "trigger_len\tmin\t[float value/min] (Seconds)";
				fds_file_out << write_string << "\n";
			}
			else
			{
				fds_file_out << string_dump << "\n";
			}
		}
	}
	else
	{
		cout << "Unable to open config_files/fds.cfg." << endl;
		return 1;
	}
	
	fds_file_in.close();
	fds_file_out.close();
	return 0;
}

/**
* Edits the FDS config file to update the tunable parameters
* @param length of fine mesh in x direction
* @param coarse time step
* @param fine time step multiplier
* @param critical cure rate
* @param transitional cure rate
* @param string of target front speed
* @return 0 on success, 1 on failure
*/
int set_fds_config_tunable_params(double fine_x_len, double time_step, int time_mult, double crit_cure_rate, double trans_cure_rate, string front_speed, double& characteristic_duration)
{
	ifstream fds_file_in;
	ofstream fds_file_out;
	fds_file_in.open("config_files/fds.cfg", std::ofstream::in);
	fds_file_out.open("config_files/fds_temp.cfg");
	string string_dump;
	string write_string;
	
	double coarse_x_len = 0.0;
	const string digits = "0123456789.+-";
	
	// Set fine_x_len, sim_duration, time_step, time_mult, crit_cure_rate, trans_cure_rate
	if (fds_file_in.is_open() && fds_file_out.is_open())
	{
		
		while(getline(fds_file_in, string_dump))
		{
			if( string_dump.find("fine_x_len") != string::npos )
			{
				write_string = "fine_x_len\t" + to_string(fine_x_len) + "\t(Meters)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("coarse_x_len") !=  string::npos )
			{
				fds_file_out << string_dump << "\n";
				unsigned start_pos = string_dump.find_first_of( digits );
				unsigned end_pos = string_dump.find_last_of( "\t" );
				if ( start_pos != string::npos )
				{
					coarse_x_len = stof(string_dump.substr( start_pos, end_pos-start_pos-1 ), NULL);
				}
			}
			else if( string_dump.find("sim_duration") !=  string::npos )
			{
				characteristic_duration = (coarse_x_len / stod(front_speed, NULL));
				write_string = "sim_duration\t" + to_string(0.65 * characteristic_duration) + "\t(Seconds)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("time_step") !=  string::npos )
			{
				write_string = "time_step\t" + to_string(time_step) + "\t(Seconds)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("time_mult") !=  string::npos )
			{
				write_string = "time_mult\t" + to_string(time_mult) + "\t(Fine time steps per coarse time step)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("crit_cure_rate") !=  string::npos )
			{
				write_string = "crit_cure_rate\t" + to_string(crit_cure_rate) + "\t(Decimal Percent)  (All cure rates below this value are assumed to be 0)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("trans_cure_rate") !=  string::npos )
			{
				write_string = "trans_cure_rate\t" + to_string(trans_cure_rate) + "\t(Decimal Percent)  (Cure rate which demarks transition from FE to RK4 integration)";
				fds_file_out << write_string << "\n";
			}
			else
			{
				fds_file_out << string_dump << "\n";
			}
		}
	}
	else
	{
		cout << "Unable to open config_files/fds.cfg." << endl;
		return 1;
	}
	
	fds_file_in.close();
	fds_file_out.close();
	
	// Rename files
	char old_name[] = "config_files/fds_temp.cfg";
	char new_name[] = "config_files/fds.cfg";
	int result = remove(new_name) + rename(old_name, new_name);
	if(result == 0)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

/**
* Resets the fds config file to its original form
* @return 0 on success, 1 on failure
*/
int reset_fds_config()
{
	char old_name_1[] = "config_files/fds.cfg";
	char new_name_1[] = "config_files/fds_tuned.cfg";
	char old_name_2[] = "config_files/fds_orig.cfg";
	char new_name_2[] = "config_files/fds.cfg";
	int result = rename(old_name_1, new_name_1) + rename(old_name_2, new_name_2);
	if(result == 0)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}


//******************************************************************** USER INTERFACE FUNCTIONS ********************************************************************//
/**
* Prints to stdout a readout of the tuning process
* @param Number of times parameters have been updated
* @param X length of fine mesh
* @param Coarse time step
* @param Fine time step multiplier
* @param Critical cure rate
* @param Transitional cure rate
*/
string get_tuning_info(int num_updates, double curr_fine_x_len, double curr_time_step, int curr_time_mult, double curr_crit_cure_rate, double curr_trans_cure_rate)
{	
	// Update point msg
	stringstream stream;
	stream << "Num Updates: " << num_updates;
	string msg1 = stream.str();
	
	// Fine x len msg
	stream.str(string());
	stream << fixed << setprecision(2);
	stream << "Fine X Len: " << 1000.0*curr_fine_x_len << " mm";
	string msg3 = stream.str();
	
	// Coarse time step msg
	stream.str(string());
	stream << fixed << setprecision(2);
	stream << "Time Step: " << 1000.0*curr_time_step << " ms";
	string msg4 = stream.str();
	
	// Fine time step mult msg
	stream.str(string());
	stream << "Time Step Mult: " << curr_time_mult;
	string msg5 = stream.str();

	// Critical cure rate msg
	stream.str(string());
	stream << fixed << setprecision(4);
	stream << "Crit Cure: " << curr_crit_cure_rate;
	string msg6 = stream.str();
	
	// Transitional cure rate msg
	stream.str(string());
	stream << fixed << setprecision(3);
	stream << "Trans Cure: " << curr_trans_cure_rate;
	string msg7 = stream.str();
	
	// Print all sub messeges
	string spacer = " | ";
	string header = "\n\n";
	string footer;
	
	// Concatenate msgs
	string return_str = header + spacer + msg1 + spacer + msg3 + spacer + msg4 + spacer + msg5 + spacer + msg6 + spacer + msg7 + spacer;
	unsigned int target_length = return_str.length()-2;
	while (footer.length() < target_length)
	{
		footer.append("*");
	}
	return_str = return_str + "\n" + footer + "\n";
	
	return return_str;
}

/**
* Prints to stdout a readout of the simulation process
* @param Total number of points used for the tuning process
* @param Current tuning point index
* @param Name of monomer
* @param Tuning point initial cure
* @param Tuning point initial temp
* @param Tuning point target speed
*/
string get_sim_info(unsigned int num_tuning_points, unsigned int curr_tuning_point, string name, string initial_cure, string initial_temp, string front_speed)
{	
	// Tuning point msg
	stringstream stream;
	stream << "Point: " << to_string(curr_tuning_point) << "/" << to_string(num_tuning_points);
	string msg1 = stream.str();
	if (msg1.length() < 12)
	{
		msg1.append(12 - msg1.length(), ' ');
	}
	
	// Monomer msg
	stream.str(string());
	stream << "Monomer: " << name;
	string msg2 = stream.str();
	if (msg2.length() < 17)
	{
		msg2.append(17 - msg2.length(), ' ');
	}
	
	// Initial cure msg
	stream.str(string());
	stream << fixed << setprecision(2);
	stream << "Cure: " << initial_cure;
	string msg3 = stream.str();
	if (msg3.length() < 10)
	{
		msg3.append(10 - msg3.length(), ' ');
	}
	
	// Initial temp msg
	stream.str(string());
	stream << fixed << setprecision(2);
	stream << "Temp: " << initial_temp << " K";
	string msg4 = stream.str();
	if (msg4.length() < 14)
	{
		msg4.append(14 - msg4.length(), ' ');
	}

	// Target front speed msg
	stream.str(string());
	stream << fixed << setprecision(3);
	stream << "Target: " << 1000.0*(stof(front_speed)) << " m/s";
	string msg5 = stream.str();
	if (msg5.length() < 17)
	{
		msg5.append(17 - msg5.length(), ' ');
	}
	
	// Print all sub messeges
	string spacer = " | ";
	return (spacer + msg1 + spacer + msg2 + spacer + msg3 + spacer + msg4 + spacer + msg5 + spacer);
}


//******************************************************************** TRAINING LOOP ********************************************************************//
/**
* Runs a single FROMP simulation and collects simualtion data
* @param The frame rate in per second
* @return 0 on success, 1 on failure
*/
int run(vector<string> &name_list, vector<string> &initial_cure_list, vector<string> &initial_temp_list, vector<string> &front_speed_list, vector<double>& params)
{
	// Populate tunable parameter limit
	double tunable_param_min[5] = { params[1], params[3], params[5], params[7], params[9]  };
	double tunable_param_max[5] = { params[2], params[4], params[6], params[8], params[10] };
	
	// Initialize tunable parameters
	double tunable_param_range[5];
	double tunable_param_curr[5];
	double tunable_param_test[5];
	double tunable_param_momentum[5];
	for(int i = 0; i < 5; i++)
	{
		// Set tunable param ranges
		tunable_param_range[i] = tunable_param_max[i] - tunable_param_min[i];
		
		// Set the initial tunable param value to the middle of the range
		tunable_param_curr[i] = tunable_param_min[i] + 0.50 * tunable_param_range[i];
		tunable_param_momentum[i] = 0.0;
	}
	
	// Name other input parameters
	double frame_rate = params[0];
	double duration_const = params[11];
	double max_stdev_const = params[12];
	double avg_stdev_const = params[13];
	double hyper_radius = params[14];
	double hyper_radius_decay = params[15];
	double momentum_const = params[16];
	int max_num_failed_updates = params[17];
	int max_num_updates = params[18];
	
	// Initialize current fitness values
	double characteristic_duration;
	double avg_sim_duration;
	double max_stdev;
	double avg_stdev;
	double curr_loss = 1.0e10;
	double test_loss = 0.0;
	
	// Calculate the average target speed for stdev normalization
	double avg_target = 0.0;
	for(unsigned int i = 0; i < front_speed_list.size(); i++)
	{
		avg_target+= stod(front_speed_list[i], NULL);
	}
	avg_target = avg_target / (double)front_speed_list.size();
	
	// Initialize fitness history
	vector<double> avg_sim_duration_history;
	vector<double> max_stdev_history;
	vector<double> avg_stdev_history;
	vector<double> loss_history;
	
	// ********************************************************** Parameter tuning loop ********************************************************** //
	bool done_tuning = false;
	int num_updates = 0;
	int num_failed_updates = 0;
	string print_string = "";
	while (!done_tuning)
	{
		// Reset the print string
		print_string = "";
		
		// Take one step in a random direction in hyperspace
		for(int i = 0; i < 5; i++)
		{
			// Normal case for continuous tunable parameters
			if( i != 2 )
			{
				tunable_param_test[i] = tunable_param_curr[i] + hyper_radius * ((2.0*((double)rand()/(double)RAND_MAX)-1.0) + tunable_param_momentum[i]) * tunable_param_range[i];
			}
			
			// Special case for discrete tunable parameters
			else
			{
				int max_step = (int)round( hyper_radius * tunable_param_range[i] );
				max_step = max_step < 1 ? 1 : max_step;
				
				int step = rand() % (max_step + 1);
				step =  ((2.0*((double)rand()/(double)RAND_MAX)-1.0) + tunable_param_momentum[i]) < 0.0 ? -step : step;
				
				tunable_param_test[i] = tunable_param_curr[i] + (double)step;
			}
			
			// Ensure in proper range
			tunable_param_test[i] = tunable_param_test[i] > tunable_param_max[i] ? tunable_param_max[i] : tunable_param_test[i];
			tunable_param_test[i] = tunable_param_test[i] < tunable_param_min[i] ? tunable_param_min[i] : tunable_param_test[i];
		}
		
		// Save the current tuning info as printable string
		print_string.append(get_tuning_info(num_updates, tunable_param_test[0], tunable_param_test[1], tunable_param_test[2], tunable_param_test[3], tunable_param_test[4]));
		
		// Reset current tuning point fitness 
		avg_sim_duration = 0.0;
		max_stdev = 0.0;
		avg_stdev = 0.0;
		
		// Loop through each tuning point
		for(unsigned int i = 0; i < name_list.size(); i++)
		{
			// Modifiy fds config to make current tuning point
			edit_fds_config(name_list[i], initial_cure_list[i], initial_temp_list[i], front_speed_list[i]);
			set_fds_config_tunable_params(tunable_param_test[0], tunable_param_test[1], (int)tunable_param_test[2], tunable_param_test[3], tunable_param_test[4], front_speed_list[i], characteristic_duration);
			
			// Initialize FDS
			Finite_Difference_Solver* FDS;
			try
			{
				FDS = new Finite_Difference_Solver();
			}
			catch (int e)
			{
				reset_fds_config();
				cout << "\nAn exception occurred. Exception num " << e << '\n';
				return 1;
			}
			FDS->reset();

			// Simulation loop
			int steps_per_frame = (int) round(1.0 / (FDS->get_coarse_time_step() * frame_rate));
			steps_per_frame = steps_per_frame < 1 ? 1 : steps_per_frame;
			bool done_simulating = false;
			int step_in_trajectory = 0;
			double curr_stdev = 0.0;
			double population_size = 0.0;
			auto sim_start_time = chrono::high_resolution_clock::now();
			while (!done_simulating)
			{
				// Update the logs
				if ((step_in_trajectory % steps_per_frame == 0) && (FDS->get_progress() >= 50.0))
				{
					// Store front speed stdev data
					curr_stdev += (FDS->get_curr_target() - FDS->get_front_vel()) * (FDS->get_curr_target() - FDS->get_front_vel());
					population_size += 1.0;
				}
				
				// Step the environment 
				done_simulating = FDS->step(0.0, 0.0, 0.0);
				step_in_trajectory++;
			}
			// Calculate the sim duration
			double sim_duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - sim_start_time ).count())*10e-7;
			avg_sim_duration += sim_duration;
			
			// Calculate the stdev
			curr_stdev = sqrt(curr_stdev / population_size);
			max_stdev = curr_stdev > max_stdev ? curr_stdev : max_stdev;
			avg_stdev += curr_stdev;
		}
		
		// Update the average sim duration at current tunable parameters
		avg_sim_duration = avg_sim_duration / ((double)name_list.size() * characteristic_duration);
		avg_sim_duration_history.push_back(avg_sim_duration);
		
		// Update the max stdev from target speed at current tunable parameters
		max_stdev = max_stdev / avg_target;
		max_stdev_history.push_back(max_stdev);
		
		// Update the average stdev from target speed at current tunable parameters
		avg_stdev = avg_stdev / ((double)name_list.size() * avg_target);
		avg_stdev_history.push_back(avg_stdev);
		
		// Calcualte test loss
		test_loss = duration_const*avg_sim_duration + max_stdev_const*max_stdev + avg_stdev_const*avg_stdev;
		
		// If good direction found, set params to that direction, end random search
		if(test_loss < curr_loss)
		{
			// Send loss to print string
			stringstream stream;
			stream.str(string());
			if (test_loss >= 100.0)
			{
				stream << fixed << setprecision(2);
			}
			else if(test_loss >= 10.0)
			{
				stream << fixed << setprecision(3);
			}
			else
			{
				stream << fixed << setprecision(4);
			}
			stream << " | Avg Dur: " << avg_sim_duration;
			stream << " | Max Std: " << max_stdev;
			stream << " | Avg Std: " << avg_stdev;
			stream << " | Loss: " << test_loss << " |";
			print_string.append(stream.str());
			unsigned int target_length = stream.str().length();
			
			// Update the current params
			curr_loss = test_loss;
			for(int i = 0; i < 5; i++)
			{
				tunable_param_momentum[i] += momentum_const * (tunable_param_test[i] - tunable_param_curr[i]) / (hyper_radius * tunable_param_range[i]);
				tunable_param_momentum[i] = tunable_param_momentum[i] * (1.0 - momentum_const);
				tunable_param_curr[i] = tunable_param_test[i];
			}
			
			// Write the momentum
			stream.str(string());
			stream << fixed << setprecision(3);
			stream << "\n | Momentum: <" << tunable_param_momentum[0] << ", " << tunable_param_momentum[1] << ", " << tunable_param_momentum[2] << ", " << tunable_param_momentum[3] << ", " << tunable_param_momentum[4] << ">";
			while (stream.str().length() < target_length)
			{
				stream << " ";
			}
			stream << "|";
			print_string.append(stream.str());
			
			// Iterator the hyper radius
			hyper_radius = hyper_radius * hyper_radius_decay;
			
			// Update update iterators
			num_updates++;
			num_failed_updates = 0;
			
			// Print training data
			cout << print_string;
		}
		else
		{
			num_failed_updates++;
		}
		
		// Detemine whether tuning is complete for not
		done_tuning = (num_updates >= max_num_updates) || (num_failed_updates >= max_num_failed_updates);
	}
	
	return 0;
}


//******************************************************************** MAIN LOOP ********************************************************************//
int main()
{
	// Set randomization seed
	srand(time(NULL));
	
	// Load parameters
	vector<string> name_list;
	vector<string> initial_cure_list;
	vector<string> initial_temp_list;
	vector<string> front_speed_list;
	vector<double> params = vector<double>(19, 0.0);
	if (load_config(name_list, initial_cure_list, initial_temp_list, front_speed_list, params) == 1) { return 1; }
	
	// Make a copy of the original FDS config file
	make_orig_fds_config();

	// Run simulation
	cout << "\n Tuning parameters...\n";
	for(unsigned int i = 0; i < name_list.size(); i++)
	{
		cout << get_sim_info(name_list.size(), i+1, name_list[i], initial_cure_list[i], initial_temp_list[i], front_speed_list[i]) << "\n";
	}
	auto start_time = chrono::high_resolution_clock::now();
	if (run(name_list, initial_cure_list, initial_temp_list, front_speed_list, params) == 1) { return 1; };
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("\n\nTuning took: %.1f seconds.\nDone!", duration);
	
	// Finish
	return reset_fds_config();
}
