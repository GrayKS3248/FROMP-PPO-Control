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
		double max_dev_const;
		double stdev_const;
		double learning_rate;
		double learning_rate_decay;
		double momentum_const;
		unsigned int max_num_updates;
		double sim_duration;
		double avg_front_speed;
	
		// public constructor and destructor
		Config_Handler();
		~Config_Handler();
		
		// public functions
		int set_params(unsigned int index);
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
		config_file >> string_dump >> max_dev_const;
		config_file.ignore(numeric_limits<streamsize>::max(), '\n');
		config_file >> string_dump >> stdev_const;
		
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
	
	// Calculate the average target speed for stdev normalization
	avg_front_speed = 0.0;
	for(unsigned int i = 0; i < front_speed_list.size(); i++)
	{
		avg_front_speed += stod(front_speed_list[i], NULL);
	}
	avg_front_speed = avg_front_speed / (double)front_speed_list.size();
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
int Config_Handler::set_params(unsigned int index)
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
				write_string = "x_step_mult\t" + to_string(x_step_mult) + "\t\t(Meters)";
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
				write_string = "time_mult\t" + to_string(time_mult) + "\t\t(Fine time steps per coarse time step)";
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
		file << "Reward Parameters: [" << config_handler->duration_const << ", " << config_handler->max_dev_const << ", " << config_handler->stdev_const << "]" << endl;
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


//******************************************************************** TRAINING LOOP AND FUNCTIONS ********************************************************************//


/**
* Gets a random value between min and max
* @param Lower bound
* @param Upper bound
* @return Random value
*/
template <typename T> T get_rand(T min, T max)
{
	double range = (double)max - (double)min;
	double random = (double)rand() / (double)RAND_MAX;
	double value = (double)min + range*random;
	return (T)value;
}
template <> int get_rand<int>(int min, int max)
{
	double range = (double)max - (double)min;
	double random = (double)rand() / (double)RAND_MAX;
	double value = (double)min + range*random;
	return (int)round(value);
}


/**
* Runs a single FROMP simulation and calculates the loss
* @param Configuration handler directing optimization
* @param Finite difference solver
* @return The loss value of the simulation
*/
double* run_sim(Config_Handler* config_handler, Finite_Difference_Solver* FDS)
{
	// Fitness values
	double curr_dev;
	int population_size = 0;
	double normalized_sim_duration;
	double normalized_max_dev = 0.0;
	double normalized_stdev = 0.0;
	
	// Simulation variables
	int steps_per_frame = (int) round(1.0 / (FDS->get_coarse_time_step() * config_handler->frame_rate));
	steps_per_frame = steps_per_frame < 1 ? 1 : steps_per_frame;
	bool done_simulating = false;
	int step_in_trajectory = 0;
	
	// Reset FDS
	FDS->reset();

	// Simulation loop
	auto sim_start_time = chrono::high_resolution_clock::now();
	while (!done_simulating)
	{
		// Calculate the front velocity stdev at each frame after front stablization
		if ((step_in_trajectory % steps_per_frame == 0) && (FDS->get_progress() >= 53.85))
		{	
			// Calculate deviation data
			curr_dev = (FDS->get_curr_target() - FDS->get_front_vel()) * (FDS->get_curr_target() - FDS->get_front_vel());
			normalized_max_dev = curr_dev > normalized_max_dev ? curr_dev : normalized_max_dev;
			normalized_stdev +=  curr_dev;
			population_size++;
		}
		
		// Step the environment 
		done_simulating = FDS->step(0.0, 0.0, 0.0);
		step_in_trajectory++;
	}
	
	// Calculate and normalize the sim duration
	normalized_sim_duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - sim_start_time ).count())*10e-7;
	normalized_sim_duration = normalized_sim_duration / config_handler->sim_duration;
	
	// Normalize the max deviation
	normalized_max_dev = normalized_max_dev / config_handler->avg_front_speed;
	
	// Calculate the and normalize stdev
	normalized_stdev = sqrt(normalized_stdev / population_size);
	normalized_stdev = normalized_stdev / config_handler->avg_front_speed;
	
	// Calculate the return values
	double* return_val = new double[4];
	return_val[0] = normalized_sim_duration;
	return_val[1] = normalized_max_dev;
	return_val[2] = normalized_stdev;
	return_val[3] = config_handler->duration_const*normalized_sim_duration + config_handler->max_dev_const*normalized_max_dev + config_handler->stdev_const*normalized_stdev;
	return return_val;
}


/**
* Performs gradient descent to optimize tunable parameters
* @param Configuration handler directing optimization
* @param Finite difference solver
* @return 0 on success, 1 on failure
*/
int optimize(Config_Handler* config_handler, Finite_Difference_Solver* FDS)
{
	
	// Initial tunable parameters to random value
	double curr_x_len = get_rand<double>(config_handler->min_x_len, config_handler->max_x_len);
	int curr_x_mult = get_rand<int>(config_handler->min_x_mult, config_handler->max_x_mult);
	double curr_time_step = get_rand<double>(config_handler->min_time_step, config_handler->max_time_step);
	int curr_time_mult = get_rand<int>(config_handler->min_time_mult, config_handler->max_time_mult);
	double curr_tran_cure = get_rand<double>(config_handler->min_tran_cure, config_handler->max_tran_cure);

	// Calculate tunable parameter ranges
	double x_len_range = config_handler->max_x_len - config_handler->min_x_len;
	double time_step_range = config_handler->max_time_step - config_handler->min_time_step;
	double tran_cure_range = config_handler->max_tran_cure - config_handler->min_tran_cure;

	// Set range normalized step size for determining gradient
	double step_size = 0.005;

	// Traning loop
	for ( unsigned int curr_update = 0; curr_update < config_handler->max_num_updates; curr_update++ )
	{
		
		// Calculate losses at each training point +- small radius
		double loss_data[config_handler->front_speed_list.size()][3][2];
		for ( unsigned int i = 0; i < config_handler->front_speed_list.size(); i++ )
		{
			// Assign current training point
			if( config_handler->set_params(i) == 1)
			{
				return 1;
			}
			
			// - x_len loss
			if( config_handler->update_tunable_params(curr_x_len - step_size*x_len_range, curr_x_mult, curr_time_step, curr_time_mult, curr_tran_cure) == 1)
			{
				return 1;
			}
			double* curr_loss_data = run_sim(config_handler, FDS);
			loss_data[i][0][0] = curr_loss_data[3];
			
			// + x_len loss
			if( config_handler->update_tunable_params(curr_x_len + step_size*x_len_range, curr_x_mult, curr_time_step, curr_time_mult, curr_tran_cure) == 1)
			{
				return 1;
			}
			curr_loss_data = run_sim(config_handler, FDS);
			loss_data[i][0][1] = curr_loss_data[3];
			
			// - time_step loss
			if( config_handler->update_tunable_params(curr_x_len, curr_x_mult, curr_time_step - step_size*time_step_range, curr_time_mult, curr_tran_cure) == 1)
			{
				return 1;
			}
			curr_loss_data = run_sim(config_handler, FDS);
			loss_data[i][1][0] = curr_loss_data[3];
			
			// + time_step loss
			if( config_handler->update_tunable_params(curr_x_len, curr_x_mult, curr_time_step + step_size*time_step_range, curr_time_mult, curr_tran_cure) == 1)
			{
				return 1;
			}
			curr_loss_data = run_sim(config_handler, FDS);
			loss_data[i][1][1] = curr_loss_data[3];
			
			// - trans_cure_rate loss
			if( config_handler->update_tunable_params(curr_x_len, curr_x_mult, curr_time_step, curr_time_mult, curr_tran_cure - step_size*tran_cure_range) == 1)
			{
				return 1;
			}
			curr_loss_data = run_sim(config_handler, FDS);
			loss_data[i][2][0] = curr_loss_data[3];
			
			// + trans_cure_rate loss
			if( config_handler->update_tunable_params(curr_x_len, curr_x_mult, curr_time_step, curr_time_mult, curr_tran_cure + step_size*tran_cure_range) == 1)
			{
				return 1;
			}
			curr_loss_data = run_sim(config_handler, FDS);
			loss_data[i][2][1] = curr_loss_data[3];
			
			// Cleanup
			delete curr_loss_data;
		}
		
		// Average the + and - losses from all training points
		double losses[3][2] = { {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0} };
		for ( unsigned int i = 0; i < config_handler->front_speed_list.size(); i++ )
		for ( int j = 0; j < 3; j++ )
		for ( int k = 0; k < 2; k++ )
		{
			losses[j][k] += loss_data[i][j][k];
		}
		for ( int j = 0; j < 3; j++ )
		for ( int k = 0; k < 2; k++ )
		{
			losses[j][k] = losses[j][k] / (double)config_handler->front_speed_list.size();
		}
		
		// Calculate the gradient
		double gradient[3];
		gradient[0] = (losses[0][1] - losses[0][0]) / (2.0 * step_size * x_len_range);
		gradient[1] = (losses[1][1] - losses[1][0]) / (2.0 * step_size * time_step_range);
		gradient[2] = (losses[2][1] - losses[2][0]) / (2.0 * step_size * tran_cure_range);
		
		// Termination condition
		if ( false )
		{
			break;
		}
		
	}
	
	return 0;

}


//******************************************************************** MAIN LOOP ********************************************************************//
int main()
{
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
	
	// Initialize solver
	Finite_Difference_Solver* FDS;
	try
	{
		FDS = new Finite_Difference_Solver();
	}
	catch (int e)
	{
		cout << "An exception occurred. Exception num " << e << '\n';
		delete config_handler;
		delete logger;
		return 1;
	}
	
	// Set randomization seed
	srand(time(NULL));
	
	// Run simulation
	cout << "Running...\n";
	auto start_time = chrono::high_resolution_clock::now();
	if ( optimize(config_handler, FDS) == 1 ) 
	{ 
		delete config_handler;
		delete logger;
		delete FDS;
		return 1; 
	}
	
	// Stop clock and print duration
	double duration = (double)(chrono::duration_cast<chrono::microseconds>( chrono::high_resolution_clock::now() - start_time ).count())*10e-7;
	printf("Tuning took: %.1f seconds.\nDone!", duration);
	
	// Delete and return successful
	delete config_handler;
	delete logger;
	delete FDS;
	return 0;
}
