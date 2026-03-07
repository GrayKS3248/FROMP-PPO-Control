#include "Config_Handler.hpp"

/**
* Constructor config handler. Config handler stores all configuration variables for easy access
* @param Path to cfg file to be loaded
* @param Name of cfg file to be loaded
*/
Config_Handler::Config_Handler(string path, string file)
{
	// Save path and file data
	cfg_path = path;
	cfg_file = file;
	
	// Save a copy of the original cfg file
	store_cfg();
	create_map();
	cfg_orig_string = cfg_string;
	cfg_string = "";
}

/**
* Returns a string copy of the original cfg file
* @return String copy of the original cfg file
*/
string Config_Handler::get_orig_cfg()
{
	return cfg_orig_string;
}

/**
* Returns a string copy of the most recently stored cfg file
* @return String copy of the most recently stored cfg file
*/
string Config_Handler::get_stored_cfg()
{
	return cfg_string;
}

/**
* Save a copy of the current state of the configuration file to a local string
* @return 0 on success, 1 on failure
*/
int Config_Handler::store_cfg()
{
	// Reset original string
	cfg_string = "";
	
	// Read lines from fds config file
	ifstream file_in;
	string path_file = cfg_path + "/" + cfg_file;
	file_in.open(path_file, ofstream::in);
	string read_line;
	if (file_in.is_open())
	{
		// Copy lines to original string
		while( getline(file_in, read_line) )
		{
			cfg_string = cfg_string + read_line + "\n";
		}
		cfg_string.pop_back();
	}
	else
	{
		cout << "Unable to open " << path_file << "\n";
		cin.get();
		return 1;
	}
	
	// Close the file
	file_in.close();
	
	return 0;
}

/**
* Saves copy of original config file to file named file at path
* @param Path to directory where cfg file copy will be saved
* @param Name of cfg file copy
* @param Boolean flag that indicated whether written cfg string will be appended to end of file or whether file will be cleared and only cfg string will be written
* @return 0 on success, 1 on failure
*/
int Config_Handler::save_copy_of_orig_cfg(string save_path, string save_file, bool append)
{
	// Open file
	string path_file = save_path + "/" + save_file;
	ofstream file;
	if (append)
	{
		file.open(path_file, ios::out | ios::app); 
		if (file.is_open())
		{
			file << "\n======================================================================================\n";
		}
	}
	else
	{
		file.open(path_file, ios::out | ios::trunc); 
	}

	// Write to file
	if (file.is_open())
	{
		file << cfg_orig_string;
	}
	else
	{
		cout << "Unable to open " << path_file << "\n";
		cin.get();
		return 1;
	}
	
	// Close file
	file.close();
	return 0;
}

/**
* Creates a dictionary of variable name to string values based on data in cfg file
* @return 0 on success, 1 on failure
*/
int Config_Handler::create_map()
{
	// Reset dictionary
	cfg_dictionary = map<string, string>();
	
	// Read lines from fds config file
	ifstream file_in;
	string path_file = cfg_path + "/" + cfg_file;
	file_in.open(path_file, ofstream::in);
	string read_line;
	if (file_in.is_open())
	{
		// Assign values to dictionary from cfg file
		while( getline(file_in, read_line) )
		{
			// Key value strings
			string key;
			string value;
			
			// Only keep data containing strings
			if (read_line.find("{") != 0 && read_line.find("\n") != 0 && read_line.find("\t") != 0 && read_line.find(" ") != 0)
			{
				
				if (read_line.find("\t") != string::npos)
				{
					key = read_line.substr(0, read_line.find("\t"));
					int curr_char_index = read_line.find("\t");
					while(read_line.substr(curr_char_index).find("\t") == 0)
					{
						curr_char_index++;
					}
					value = read_line.substr(curr_char_index, read_line.substr(curr_char_index).find("\t"));
					
				}
			}
			cfg_dictionary.insert( pair<string,string>(key,value) );
		}
	}
	else
	{
		cout << "Unable to open " << path_file << "\n";
		cin.get();
		return 1;
	}
	
	// Close the file
	file_in.close();
	
	return 0;
	
	
	
	return 0;
}

/**
* Returns the integer value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, int& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 21;
	}
	
	var_value = stoi(cfg_dictionary[var_name]);
	return var_value;
}

/**
* Returns the double value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
double Config_Handler::get_var(string var_name, double& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 20;
	}
	
	var_value = stod(cfg_dictionary[var_name]);
	return var_value;
}

/**
* Returns the float value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
float Config_Handler::get_var(string var_name, float& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 19;
	}
	
	var_value = stof(cfg_dictionary[var_name]);
	return var_value;
}

/**
* Returns the long value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
long Config_Handler::get_var(string var_name, long& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 18;
	}
	
	var_value = stol(cfg_dictionary[var_name]);
	return var_value;
}

/**
* Returns the long double value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
long double Config_Handler::get_var(string var_name, long double& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 17;
	}
	
	var_value = stold(cfg_dictionary[var_name]);
	return var_value;
}

/**
* Returns the unsigned int value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
unsigned int Config_Handler::get_var(string var_name, unsigned int& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 16;
	}
	
	var_value = stoul(cfg_dictionary[var_name]);
	return var_value;
}

/**
* Returns the unsigned long value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
unsigned long Config_Handler::get_var(string var_name, unsigned long& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 15;
	}
	
	var_value = stoul(cfg_dictionary[var_name]);
	return var_value;
}

/**
* Returns the long long value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
long long Config_Handler::get_var(string var_name, long long& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 14;
	}
	
	var_value = stoll(cfg_dictionary[var_name]);
	return var_value;
}

/**
* Returns the unsinged long long value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
unsigned long long Config_Handler::get_var(string var_name, unsigned long long& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 13;
	}
	
	var_value = stoull(cfg_dictionary[var_name]);
	return var_value;
}

/**
* Returns the boolean value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
bool Config_Handler::get_var(string var_name, bool& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 12;
	}
	
	if (cfg_dictionary[var_name].compare("true")==0)
	{
		var_value = true;
	}
	else if (cfg_dictionary[var_name].compare("false")==0)
	{
		var_value = false;
	}
	else
	{
		cout << "\nString value not recognized as boolean";
		cin.get();
		throw 12;
	}
	
	return var_value;
}

/**
* Returns the const char* (c string) value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
const char* Config_Handler::get_var(string var_name, const char*& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 11;
	}
	
	var_value = cfg_dictionary[var_name].c_str();
	return var_value;
}

/**
* Returns the string value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
string Config_Handler::get_var(string var_name, string& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		cin.get();
		throw 10;
	}
	
	var_value = cfg_dictionary[var_name];
	return var_value;
}