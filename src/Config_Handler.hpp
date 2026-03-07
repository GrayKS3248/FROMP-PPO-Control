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
		int save_copy_of_orig_cfg(string save_path, string save_file, bool append);
		int get_var(string var_name, int& var_value);
		double get_var(string var_name, double& var_value);
		float get_var(string var_name, float& var_value);
		long get_var(string var_name, long& var_value);
		long double get_var(string var_name, long double& var_value);
		unsigned int get_var(string var_name, unsigned int& var_value);
		unsigned long get_var(string var_name, unsigned long& var_value);
		long long get_var(string var_name, long long& var_value);
		unsigned long long get_var(string var_name, unsigned long long& var_value);
		bool get_var(string var_name, bool& var_value);
		const char* get_var(string var_name, const char*& var_value);
		string get_var(string var_name, string& var_value);
	
	private:
		int create_map();
		map<string, string> cfg_dictionary;
		string cfg_path = "";
		string cfg_file = "";
		string cfg_orig_string = "";
		string cfg_string = "";
};