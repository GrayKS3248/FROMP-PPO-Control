#include "Finite_Difference_Solver.h"

using namespace std;


//******************************************************************** CONFIGURATION HANDLER CLASS ********************************************************************//
class Config_Handler
{
	public:
		// public variables
		vector<string> name_list;
		vector<string> initial_cure_list;
		vector<string> initial_temp_list;
		vector<string> front_speed_list;
		double frame_rate;
		double min_x_len;
		double max_x_len;
		int min_x_mult;
		int max_x_mult;
		double min_time_step;
		double max_time_step;
		int min_time_mult;
		int max_time_mult;
		double min_tran_cure;
		double max_tran_cure;
		double duration_const;
		double max_stdev_const;
		double avg_stdev_const;
		double learning_rate;
		double learning_rate_decay;
		double momentum_const;
		int max_num_updates;
		double sim_duration;

		// public constructor and destructor
		Config_Handler();
		~Config_Handler();
		
		// public functions
		int set_params(int index);
		int update_tunable_params(double fine_x_len, int x_step_mult, double time_step, int time_mult, double trans_cure_rate);
		
		
	private:
		// private variables
		string orig_fds_config;
		double coarse_x_len = 1.0;
		double curr_front_speed = 1.0;
		
		// private functions
		int copy_to_orig();
		int reset_config();
};


/**
* Constructor loads files from config
*/
Config_Handler::Config_Handler()
{	
	// Load from config file
	ifstream config_file;
	config_file.open("config_files/tune_params.cfg");
	string string_dump;
	if (config_file.is_open())
	{
		// Simulation parameters
		config_file.ignore(numeric_limits<streamsize>::max(), '}');
		config_file >> string_dump >> frame_rate;
		
		// Tuning parameter limits
		config_file.ignore(numeric_limits<streamsize>::max(), '}');
		config_file >> string_dump >> min_x_len;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> max_x_len;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> min_x_mult;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> max_x_mult;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> min_time_step;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> max_time_step;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> min_time_mult;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> max_time_mult;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> min_tran_cure;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> max_tran_cure;
		
		// Fitness parameters
		config_file.ignore(numeric_limits<streamsize>::max(), '}');
		config_file >> string_dump >> duration_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> max_stdev_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> avg_stdev_const;
		
		// Search parameters
		config_file.ignore(numeric_limits<streamsize>::max(), '}');
		config_file >> string_dump >> learning_rate;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> learning_rate_decay;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> momentum_const;
		
		// Termination parameters
		config_file.ignore(numeric_limits<streamsize>::max(), '}');
		config_file >> string_dump >> max_num_updates;
		
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
		throw 2;
	}
	config_file.close();
	
	// Get copy of original fds config file
	if( copy_to_orig() == 1 ) 
	{
		throw 3;
	}
}


/**
* Destructor
*/
Config_Handler::~Config_Handler()
{
	reset_config();
}


/**
* Edits the FDS config file to run the monomer, initial cure, initial temperature, and target front speed of the defined test point index
* Additionally sets all randomizing simulation parameters to 0
* @param index of monomer, cure, temp, and target to be set
* @return 0 on success, 1 on failure
*/
int Config_Handler::set_params(int index)
{
	ofstream file_out;
	istringstream f(orig_fds_config);
	string read_line; 
	string write_line;
	
	// Set use_input, use_trigger, material, control, target_type, target_speed, tar_speed_dev, tar_temp_dev, initial_temp, temp_deviation, initial_cure, cure_deviation, mean_htc, htc_dev, amb_temp_dev, trigger_len
	file_out.open("config_files/fds.cfg");
	if ( file_out.is_open() )
	{
		while ( getline(f, read_line) )
		{
			if( read_line.find("use_input") != string::npos )
			{
				write_line = "use_input\tfalse\t\t[true/false]";
				file_out << write_line << "\n";
			}
			else if( read_line.find("use_trigger") !=  string::npos )
			{
				write_line = "use_trigger\ttrue\t\t[true/false]";
				file_out << write_line << "\n";
			}
			else if( read_line.find("material") !=  string::npos )
			{
				write_line = "material\t"+name_list[index]+"\t[dcpd_gc1/dcpd_gc2/cod]";
				file_out << write_line << "\n";
			}
			else if( read_line.find("control") !=  string::npos )
			{
				write_line = "control\t\tspeed\t\t[speed/temp]";
				file_out << write_line << "\n";
			}
			else if( read_line.find("target_type") !=  string::npos )
			{
				write_line = "target_type\tconst\t\t[const/rand/switch]";
				file_out << write_line << "\n";
			}
			else if( read_line.find("target_speed") !=  string::npos )
			{
				write_line = "target_speed\t"+front_speed_list[index]+"\t(Meters / Second)";
				file_out << write_line << "\n";
				curr_front_speed = stof(front_speed_list[index]);
				sim_duration = 0.65 * (coarse_x_len / curr_front_speed);
			}
			else if( read_line.find("tar_speed_dev") !=  string::npos )
			{
				write_line = "tar_speed_dev\t0.0\t\t(Meters / Second)";
				file_out << write_line << "\n";
			}
			else if( read_line.find("tar_temp_dev") !=  string::npos )
			{
				write_line = "tar_temp_dev\t0.0\t\t(Kelvin)";
				file_out << write_line << "\n";
			}
			else if( read_line.find("initial_temp") !=  string::npos )
			{
				write_line = "initial_temp\t"+initial_temp_list[index]+"\t\t(Kelvin)";
				file_out << write_line << "\n";
			}
			else if( read_line.find("temp_deviation") !=  string::npos )
			{
				write_line = "temp_deviation\t0.0\t\t(Kelvin)";
				file_out << write_line << "\n";
			}
			else if( read_line.find("initial_cure") !=  string::npos )
			{
				write_line = "initial_cure\t"+initial_cure_list[index]+"\t\t(Decimal Percent)";
				file_out << write_line << "\n";
			}
			else if( read_line.find("cure_deviation") !=  string::npos )
			{
				write_line = "cure_deviation\t0.0\t\t(Decimal Percent)";
				file_out << write_line << "\n";
			}
			else if( read_line.find("mean_htc") !=  string::npos )
			{
				write_line = "mean_htc\t0.0\t\t(Watts / (Meter ^ 2 * Kelvin))";
				file_out << write_line << "\n";
			}
			else if( read_line.find("htc_dev") !=  string::npos )
			{
				write_line = "htc_dev\t\t0.0\t\t(Watts / (Meter ^ 2 * Kelvin))";
				file_out << write_line << "\n";
			}
			else if( read_line.find("amb_temp_dev") !=  string::npos )
			{
				write_line = "amb_temp_dev\t0.0\t\t(Kelvin)";
				file_out << write_line << "\n";
			}
			else if( read_line.find("trigger_len") !=  string::npos )
			{
				write_line = "trigger_len\tmin\t\t[float value/min] (Seconds)";
				file_out << write_line << "\n";
			}
			else
			{
				file_out << read_line << "\n";
			}
		}
	}
	else
	{
		cout << "Unable to open config_files/fds.cfg." << endl;
		return 1;
	}
	
	file_out.close();
	
	return 0;
}


/**
* Edits the FDS config file to update the tunable parameters
* @param length of fine mesh in x direction
* @param fine grid x step multiplier
* @param coarse time step
* @param fine time step multiplier
* @param transitional cure rate
* @return 0 on success, 1 on failure
*/
int Config_Handler::update_tunable_params(double fine_x_len, int x_step_mult, double time_step, int time_mult, double trans_cure_rate)
{
	ifstream fds_file_in;
	ofstream fds_file_out;
	fds_file_in.open("config_files/fds.cfg", std::ofstream::in);
	fds_file_out.open("config_files/fds_temp.cfg");
	string string_dump;
	string write_string;
	
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
			else if( string_dump.find("x_step_mult") != string::npos )
			{
				write_string = "x_step_mult\t" + to_string(x_step_mult) + "\t(Meters)";
				fds_file_out << write_string << "\n";
			}
			else if( string_dump.find("sim_duration") !=  string::npos )
			{
				write_string = "sim_duration\t" + to_string(sim_duration) + "\t(Seconds)";
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
* Saves copy of original fds config to private string
* @return 0 on success, 1 on failure
*/
int Config_Handler::copy_to_orig()
{
	// Reset original string
	orig_fds_config = "";
	
	// Read lines from config file
	ifstream file_in;
	file_in.open("config_files/fds.cfg", ofstream::in);
	string read_line;
	
	if (file_in.is_open())
	{
		// Copy lines to original string
		while( getline(file_in, read_line) )
		{
			// Save the coarse x length
			if( read_line.find("coarse_x_len") !=  string::npos )
			{
				size_t start_pos = read_line.find_first_of( "0123456789." );
				size_t end_pos = read_line.find_last_of( "\t" );
				if ( start_pos != string::npos )
				{
					coarse_x_len = stof(read_line.substr( start_pos, end_pos-start_pos-1 ), NULL);
				}
			}
			
			orig_fds_config = orig_fds_config + read_line + "\n";
			
		}
		orig_fds_config.pop_back();
	}
	else
	{
		cout << "Unable to open config_files/fds.cfg." << endl;
		return 1;
	}
	
	// Close the file
	file_in.close();
	
	return 0;
}


/**
* Resets the fds config file to its original value
* @return 0 on success, 1 on failure
*/
int Config_Handler::reset_config()
{
	// Read lines from config file
	ofstream fds_file_out;
	fds_file_out.open("config_files/fds.cfg", ofstream::out | ofstream::trunc);
	if (fds_file_out.is_open())
	{
		fds_file_out << orig_fds_config;
	}
	else
	{
		cout << "Unable to reset config_files/fds.cfg." << endl;
		return 1;
	}
	
	// Close the file
	fds_file_out.close();
	
	return 0;
}


//******************************************************************** LOGGER CLASS ********************************************************************//
class Logger
{
	public:
		// public variables

		// public constructor and destructor
		Logger(Config_Handler* config_handler);
		
		// public functions
		
		
	private:
		// private variables
		
		// private functions
		ofstream file;
};


/**
* Constructor
*/
Logger::Logger(Config_Handler* config_handler)
{
	// Open log file and copy config file to log
	file.open("config_files/log.dat", ofstream::trunc);
	if (file.is_open()) 
	{ 
		file << "{ Configuration Parameters }" << endl;
		file << "Frame Rate: " << config_handler->frame_rate << " frames per second" << endl;
		file << "X Length Range: [" << config_handler->min_x_len << ", " << config_handler->max_x_len << "] meters" << endl;
		file << "X Multiplier Range: [" << config_handler->min_x_mult << ", " << config_handler->max_x_mult << "]" << endl;
		file << "Time Step Range: [" << config_handler->min_time_step << ", " << config_handler->max_time_step << "] seconds" << endl;
		file << "Time Multipler Range: [" << config_handler->min_time_mult << ", " << config_handler->max_time_mult << "]" << endl;
		file << "Transitional Cure Rate Range: [" << config_handler->min_tran_cure << ", " << config_handler->max_tran_cure << "] 1/s" << endl;
		file << "Reward Parameters: [" << config_handler->duration_const << ", " << config_handler->max_stdev_const << ", " << config_handler->avg_stdev_const << "]" << endl;
		file << "Learning Rate: " << config_handler->learning_rate << endl;
		file << "Learning Rate Decay: " << config_handler->learning_rate_decay << endl;
		file << "Momentum Constant: " << config_handler->momentum_const << endl;
		file << "Number of Updates: " << config_handler->max_num_updates << endl << endl;
		
		file << "{ Training Set }" << endl;
		file << "( Monomer, Initial Cure, Initial Temperature, Target Front Speed )" << endl;
		for(unsigned int i = 0; i < config_handler->name_list.size(); i++)
		{
			file << config_handler->name_list[i] << ", " << config_handler->initial_cure_list[i] << ", " << config_handler->initial_temp_list[i] << ", " << config_handler->front_speed_list[i] << endl;
		}
		file << endl;
		
		file << "{ Training Results }" << endl;
		file << "()" << endl;
	}
	else 
	{ 
		cout << "Unable to open config_files/log.dat." << endl; 
		throw 4; 
	}
	
	file.close();
}


//******************************************************************** USER INTERFACE FUNCTIONS ********************************************************************//
/**
* Prints to stdout a readout of the tuning process
* @param Number of times parameters have been updated
* @param X length of fine mesh
* @param x step multiplier
* @param Coarse time step
* @param Transitional cure rate
*/
/* string get_tuning_info(int num_updates, double curr_fine_x_len, int curr_x_step_mult, double curr_time_step, int curr_time_mult, double curr_trans_cure_rate)
{	
	// Update point msg
	stringstream stream;
	stream << "Num Updates: " << num_updates;
	string msg1 = stream.str();
	
	// Fine x len msg
	stream.str(string());
	stream << fixed << setprecision(2);
	stream << "Fine X Len: " << 1000.0*curr_fine_x_len << " mm";
	string msg2 = stream.str();
	
	// X step multiplier msg
	stream.str(string());
	stream << "X Step Mult: " << curr_x_step_mult;
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
	
	// Transitional cure rate msg
	stream.str(string());
	stream << fixed << setprecision(3);
	stream << "Trans Cure: " << curr_trans_cure_rate;
	string msg6 = stream.str();
	
	// Print all sub messeges
	string spacer = " | ";
	string header = "\n\n";
	string footer;
	
	// Concatenate msgs
	string return_str = header + spacer + msg1 + spacer + msg2 + spacer + msg3 + spacer + msg4 + spacer + msg5 + spacer + msg6 + spacer;
	unsigned int target_length = return_str.length()-2;
	while (footer.length() < target_length)
	{
		footer.append("*");
	}
	return_str = return_str + "\n" + footer + "\n";
	
	return return_str;
} */

/**
* Prints to stdout a readout of the simulation process
* @param Total number of points used for the tuning process
* @param Current tuning point index
* @param Name of monomer
* @param Tuning point initial cure
* @param Tuning point initial temp
* @param Tuning point target speed
*/
/* string get_sim_info(unsigned int num_tuning_points, unsigned int curr_tuning_point, string name, string initial_cure, string initial_temp, string front_speed)
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
} */


//******************************************************************** TRAINING LOOP ********************************************************************//
/**
* Runs a single FROMP simulation and collects simualtion data
* @param The frame rate in per second
* @return 0 on success, 1 on failure
*/
int run(Config_Handler* config_handler)
{

}


/* int run(vector<string> &name_list, vector<string> &initial_cure_list, vector<string> &initial_temp_list, vector<string> &front_speed_list, vector<double>& params)
{
	// Open log file
	ofstream log_file;
	log_file.open("config_files/log.dat", ofstream::app);
	if (!log_file.is_open()) 
	{
		cout << "Unable to open config_files/log.dat." << endl; 
		return 1; 
	}
	
	// Initialize tunable parameters
	double tunable_param_min[5] = { params[1], params[3], params[5], params[7], params[9]  };
	double tunable_param_max[5] = { params[2], params[4], params[6], params[8], params[10] };
	bool tunable_param_type[5] = { false, true, false, true, false };
	double tunable_param_range[5];
	double tunable_param_curr[5];
	double tunable_param_test[5];
	double tunable_param_momentum[5];
	for(int i = 0; i < 5; i++)
	{
		// Set tunable param ranges
		tunable_param_range[i] = tunable_param_max[i] - tunable_param_min[i];
		
		// Set the initial tunable param value to a random location in the range
		tunable_param_curr[i] = tunable_param_min[i] + ((double)rand()/(double)RAND_MAX) * tunable_param_range[i];
		if (tunable_param_type[i])
		{
			tunable_param_curr[i] = round(tunable_param_curr[i]);
		}
		tunable_param_momentum[i] = 0.0;
	}
	double grad_dirn[6][5] = { { 0.0, 0.0, 0.0, 0.0, 0.0 }, 
				   { tunable_param_range[0], 0.0, 0.0, 0.0, 0.0 }, 
				   { 0.0, tunable_param_range[1], 0.0, 0.0, 0.0 }, 
				   { 0.0, 0.0, tunable_param_range[2], 0.0, 0.0 }, 
				   { 0.0, 0.0, 0.0, tunable_param_range[3], 0.0 }, 
				   { 0.0, 0.0, 0.0, 0.0, tunable_param_range[4] } };
	
	// Name other parameters
	double frame_rate = params[0];
	double duration_const = params[11];
	double max_stdev_const = params[12];
	double avg_stdev_const = params[13];
	double search_step = params[14];
	double decay_rate = params[15];
	double momentum_const = params[16];
	int max_num_updates = params[17];
	
	// Calculate the average target speed for stdev normalization
	double avg_target = 0.0;
	for(unsigned int i = 0; i < front_speed_list.size(); i++)
	{
		avg_target+= stod(front_speed_list[i], NULL);
	}
	avg_target = avg_target / (double)front_speed_list.size();
	
	// ********************************************************** Parameter tuning loop ********************************************************** //
	bool done_tuning = false;
	int num_updates = 0;
	string print_string;
	while (!done_tuning)
	{
		// Reset the print string
		print_string = "";
		print_string.append(get_tuning_info(num_updates, tunable_param_curr[0], tunable_param_curr[1], tunable_param_curr[2], tunable_param_curr[3], tunable_param_curr[4]));
		
		// Declare gradient values
		double losses[6];
		double gradient[5];
		double grad_signs[5];
		double prev_loss = 1.0e10;
		
		// Declare fitness values
		double avg_sim_duration;
		double max_stdev;
		double avg_stdev;
			
		// Calculate losses around current point
		for(int i = 0; i < 6; i++)
		{
			// Set the current test point's tunable parameters
			for(int j = 0; j < 5; j++)
			{
				// Select random direction to check gradient
				if (i == 0)
				{
					double rand_val = 0.0;
					while( rand_val == 0.0 )
					{
						rand_val = 2.0 * ((double)rand()/(double)RAND_MAX) - 1.0;
					}
					grad_signs[j] = round(rand_val / abs(rand_val));
				}
				
				
				// Discrete case
				if (tunable_param_type[j])
				{
					if ( grad_dirn[i][j] != 0.0 )
					{
						tunable_param_test[j] = tunable_param_curr[j] + grad_signs[j];
					}
					else
					{
						tunable_param_test[j] = tunable_param_curr[j];
					}
				}
				
				// Continuous case
				else
				{
					tunable_param_test[j] = tunable_param_curr[j] + search_step * grad_dirn[i][j] * grad_signs[j];
				}
			}
				
			// Reset fitness values
			avg_sim_duration = 0.0;
			max_stdev = 0.0;
			avg_stdev = 0.0;
	
			// Loop through each tuning point
			for(unsigned int j = 0; j < name_list.size(); j++)
			{
				// Declare characteristic duration
				double characteristic_duration;
				
				// Modifiy fds config to make current tuning point
				edit_fds_config(name_list[j], initial_cure_list[j], initial_temp_list[j], front_speed_list[j]);
				set_fds_config_tunable_params(tunable_param_test[0], (int)tunable_param_test[1], tunable_param_test[2], (int)tunable_param_test[3], tunable_param_test[4], front_speed_list[j], characteristic_duration);
				
				// Initialize FDS
				Finite_Difference_Solver* FDS = new Finite_Difference_Solver();
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
				avg_sim_duration += sim_duration / characteristic_duration;
				
				// Calculate the stdev
				curr_stdev = sqrt(curr_stdev / population_size);
				max_stdev = curr_stdev > max_stdev ? curr_stdev : max_stdev;
				avg_stdev += curr_stdev;
			}
			
			// Update the average sim duration at current tunable parameters
			avg_sim_duration = avg_sim_duration / (double)name_list.size();
			
			// Update the max stdev from target speed at current tunable parameters
			max_stdev = max_stdev / avg_target;
			
			// Update the average stdev from target speed at current tunable parameters
			avg_stdev = avg_stdev / ((double)name_list.size() * avg_target);
			
			// Calcualte test loss
			losses[i] = duration_const*avg_sim_duration + max_stdev_const*max_stdev + avg_stdev_const*avg_stdev;
		}
		
		// Calculate range normalized gradient
		double grad_mag = 0.0;
		for (int i = 0; i < 5; i++)
		{
			
			// Discrete case
			if (tunable_param_type[i])
			{
				// Only take discrete step if you found a better solution
				if (losses[i+1] - losses[0] < 0.0)
				{
					gradient[i] = grad_signs[i] * (losses[i+1] - losses[0]) * tunable_param_range[i];
				}
				else
				{
					gradient[i] = 0.0;
				}
				
			}
			
			// Continuous case
			else
			{
				gradient[i] = grad_signs[i] * (losses[i+1] - losses[0]) / search_step;
			}
			
			grad_mag += gradient[i] * gradient[i];
		}
		grad_mag = sqrt(grad_mag);
		for (int i = 0; i < 5; i++) { gradient[i] = gradient[i] / grad_mag; }
		
		// Take a step in the best direction
		for ( int i = 0; i < 5; i++ )
		{
			double curr_val = tunable_param_curr[i];
			double new_val;
			
			// Discrete case
			if ( tunable_param_type[i] )
			{
				if ( abs(gradient[i]) >= 0.20 )
				{
					new_val = round(tunable_param_curr[i] - gradient[i] / abs(gradient[i]) );
				}
				else
				{
					new_val = tunable_param_curr[i];
				}
			}
			
			// Continuous case
			else
			{
				new_val = tunable_param_curr[i] - gradient[i] * search_step * tunable_param_range[i];
			}
			
			// Update tunable parameters with momentum only if continuous parameters
			if( tunable_param_type[i] )
			{
				tunable_param_curr[i] = new_val;
			}
			else
			{
				tunable_param_curr[i] = new_val + tunable_param_momentum[i];
			}
			
			// Update momentum
			tunable_param_momentum[i] = momentum_const * (new_val - curr_val);
		}
		
		// Send loss to print string
		stringstream stream;
		stream.str(string());
		if (losses[0] >= 100.0)
		{
			stream << fixed << setprecision(2);
		}
		else if(losses[0] >= 10.0)
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
		stream << " | Loss: " << losses[0] << " |";
		print_string.append(stream.str());
		unsigned int target_length = stream.str().length();
		
		// Write the gradient
		stream.str(string());
		stream << fixed << setprecision(3);
		stream << "\n | Gradient: <" << gradient[0] << ", " << gradient[1] << ", " << gradient[2] << ", " << gradient[3] << ", " << gradient[4] << ">";
		while (stream.str().length() < target_length)
		{
			stream << " ";
		}
		stream << "|";
		print_string.append(stream.str());
		
		// Write the momentum
		stream.str(string());
		stream << fixed << setprecision(5);
		stream << "\n | Momentum: <" << 1000.0*tunable_param_momentum[0] << ", " << tunable_param_momentum[1] << ", " << 1000.0*tunable_param_momentum[2] << ", " << tunable_param_momentum[3] << ", " << tunable_param_momentum[4] << ">";
		while (stream.str().length() < target_length)
		{
			stream << " ";
		}
		stream << "|";
		print_string.append(stream.str());
		
		
		// Iterator the step and momentum
		search_step = search_step * decay_rate;
		momentum_const = momentum_const * decay_rate;
		
		// Update update iterators
		num_updates++;
		
		// Print training data
		cout << print_string;
		log_file << print_string;
		
		// Detemine whether tuning is complete for not
		done_tuning = num_updates >= max_num_updates;
		prev_loss = losses[0];
	}
	
	log_file.close();
	return 0;
} */


//******************************************************************** MAIN LOOP ********************************************************************//
int main()
{
	// Set randomization seed
	srand(time(NULL));
	
	// Initialize configuration handler
	Config_Handler* config_handler;
	try
	{
		config_handler = new Config_Handler();
	}
	catch (int e)
	{
		cout << "An exception occurred. Exception num " << e << '\n';
		return 1;
	}
	
	// Initialize logger
	Logger* logger;
	try
	{
		logger = new Logger(config_handler);
	}
	catch (int e)
	{
		cout << "An exception occurred. Exception num " << e << '\n';
		delete config_handler;
		return 1;
	}
	
	// Run simulation
	cout << "Running...\n";
	auto start_time = chrono::high_resolution_clock::now();
	if ( run(config_handler) == 1 ) 
	{ 
		delete config_handler;
		delete logger;
		return 1; 
	}
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("Tuning took: %.1f seconds.\nDone!", duration);
	
	// Delete and return successful
	delete config_handler;
	delete logger;
	return 0;
}
