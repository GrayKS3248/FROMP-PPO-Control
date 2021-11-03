#pragma once
#include <map>
#include <fstream>
#include <iostream>
#include <string>
#include <typeinfo>
#include <cstring>

using namespace std;

/**
* Speed estimator class takes observations of front locations and times and converts these to front speed estimates
**/
class Config_Handler
{
	public:
		Config_Handler(string path, string file);
		string get_orig_cfg();
		string get_stored_cfg();
		int store_cfg();
		int get_var(string var_name, int& var_value);
		int get_var(string var_name, double& var_value);
		int get_var(string var_name, float& var_value);
		int get_var(string var_name, long& var_value);
		int get_var(string var_name, long double& var_value);
		int get_var(string var_name, unsigned int& var_value);
		int get_var(string var_name, unsigned long& var_value);
		int get_var(string var_name, long long& var_value);
		int get_var(string var_name, unsigned long long& var_value);
		int get_var(string var_name, bool& var_value);
		int get_var(string var_name, const char*& var_value);
		int get_var(string var_name, string& var_value);
	
	private:
		int create_map();
		int read_from_cfg();
		int write_to_cfg();
		map<string, string> cfg_dictionary;
		string cfg_path = "";
		string cfg_file = "";
		string cfg_orig_string = "";
		string cfg_string = "";
};

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
		return 1;
	}
	
	// Close the file
	file_in.close();
	
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
		return 1;
	}
	
	// Close the file
	file_in.close();
	
	return 0;
	
	
	
	return 0;
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
		return 1;
	}
	
	var_value = stoi(cfg_dictionary[var_name]);
	return 0;
}

/**
* Returns the double value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, double& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
	}
	
	var_value = stod(cfg_dictionary[var_name]);
	return 0;
}

/**
* Returns the float value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, float& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
	}
	
	var_value = stof(cfg_dictionary[var_name]);
	return 0;
}

/**
* Returns the long value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, long& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
	}
	
	var_value = stol(cfg_dictionary[var_name]);
	return 0;
}

/**
* Returns the long double value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, long double& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
	}
	
	var_value = stold(cfg_dictionary[var_name]);
	return 0;
}

/**
* Returns the unsigned int value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, unsigned int& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
	}
	
	var_value = stoul(cfg_dictionary[var_name]);
	return 0;
}

/**
* Returns the unsigned long value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, unsigned long& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
	}
	
	var_value = stoul(cfg_dictionary[var_name]);
	return 0;
}

/**
* Returns the long long value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, long long& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
	}
	
	var_value = stoll(cfg_dictionary[var_name]);
	return 0;
}

/**
* Returns the unsinged long long value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, unsigned long long& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
	}
	
	var_value = stoull(cfg_dictionary[var_name]);
	return 0;
}

/**
* Returns the boolean value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, bool& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
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
		return 1;
	}
	
	return 0;
}

/**
* Returns the const char* (c string) value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, const char*& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
	}
	
	var_value = cfg_dictionary[var_name].c_str();
	return 0;
}

/**
* Returns the string value of the string stored at key var_name collected from cfg file
* @param Key of string value to be converted
* @param Reference to variable to be assigned string value at key
* @return 0 on success, 1 on failure
*/
int Config_Handler::get_var(string var_name, string& var_value)
{
	if (cfg_dictionary[var_name].compare(string())==0)
	{
		cout << "Key: " << var_name << " not found.\n";
		return 1;
	}
	
	var_value = cfg_dictionary[var_name];
	return 0;
}