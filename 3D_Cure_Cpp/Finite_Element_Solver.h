#pragma once
#include <math.h>
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

using namespace std;

class Finite_Element_Solver {
	

//******************************************************************** CONSTRUCTOR/DESTRUCTOR ********************************************************************//
public:
Finite_Element_Solver();
~Finite_Element_Solver();


//******************************************************************** GETTERS ********************************************************************//
// Mesh getters
int get_num_vert_length();
int get_num_vert_width();
int get_num_vert_depth();
vector<vector<double>> get_mesh_x_z0();
vector<vector<double>> get_mesh_y_z0();

// Time getters
double get_sim_duration();
double get_time_step();
double get_current_time();

// Input parameters getters
double get_max_input_mag();
double get_exp_const();
double get_max_input_mag_percent_rate();
double get_max_input_loc_rate();

// Input state getters
double get_input_percent();
vector<double> get_input_location();

// Target getters
double get_current_target();
int get_steps_per_episode();

//Sim option getter
bool get_control_speed();

// Temperature and cure getters
vector<vector<double>> get_temp_mesh();
vector<vector<double>> get_norm_temp_mesh();
vector<vector<double>> get_cure_mesh();

// Front state getters
vector<vector<double>> get_front_curve();
double get_front_vel();
double get_front_temp();
double get_front_shape_param();


//******************************************************************** PUBLIC FUNCTIONS ********************************************************************//
void print_params();
void reset();
bool step(double x_loc_rate_action, double y_loc_rate_action, double mag_action);
double get_reward();


//******************************************************************** CONSTANTS ********************************************************************//
private:
// Physical constants
const double gas_const = 8.314;  // Joules / Mol * Kelvin

// DCPD Monomer with GC1 physical parameters
const double DCPD_GC1_thermal_conductivity = 0.15;      // Watts / Meter * Kelvin
const double DCPD_GC1_density = 980.0;                  // Kilograms / Meter ^ 3
const double DCPD_GC1_enthalpy_of_reaction = 352100.0;  // Joules / Kilogram
const double DCPD_GC1_specific_heat = 1440.0;           // Joules / Kilogram * Kelvin
const double DCPD_GC1_pre_exponential = 190985.3;       // 1 / Seconds
const double DCPD_GC1_activiation_energy = 51100.0;     // Joules / Mol
const double DCPD_GC1_model_fit_order = 1.927;          // Unitless
const double DCPD_GC1_autocatalysis_const = 0.365;      // Unitless

// DCPD Monomer with GC2 physical parameters
const double DCPD_GC2_thermal_conductivity = 0.152;     // Watts / Meter * Kelvin
const double DCPD_GC2_density = 980.0;                  // Kilograms / Meter ^ 3
const double DCPD_GC2_enthalpy_of_reaction = 350000.0;  // Joules / Kilogram
const double DCPD_GC2_specific_heat = 1600.0;           // Joules / Kilogram * Kelvin
const double DCPD_GC2_pre_exponential = 8.55e15;        // 1 / Seconds
const double DCPD_GC2_activiation_energy = 110750.0;    // Joules / Mol
const double DCPD_GC2_model_fit_order = 1.72;           // Unitless
const double DCPD_GC2_m_fit = 0.77;                     // Unitless
const double DCPD_GC2_diffusion_const = 14.48;          // Unitless
const double DCPD_GC2_critical_cure = 0.41;             // Decimal Percent

// COD Monomer physical parameters
const double COD_thermal_conductivity = 0.133;     // Watts / Meter * Kelvin
const double COD_density = 882.0;                  // Kilograms / Meter ^ 3
const double COD_enthalpy_of_reaction = 220596.0;  // Joules / Kilogram
const double COD_specific_heat = 1838.5;           // Joules / Kilogram * Kelvin
const double COD_pre_exponential = 2.13e19;        // 1 / Seconds
const double COD_activiation_energy = 132000.0;    // Joules / Mol
const double COD_model_fit_order = 2.5141;         // Unitless
const double COD_m_fit = 0.8173;                   // Unitless
				
// Precalculated arrays for cure rate
double* exp_arr;
double* pow_arr;
double* temp_arr;
double* cure_arr;
double start_temp;
double end_temp;
double temp_step;
int exp_arr_len;
double start_cure;
double end_cure;
double cure_step;
int pow_arr_len;
				
// Laplacian calculation consts for 3rd order 7-stencil
double laplacian_consts_3rd[5][7] = { { 137.0/180.0, -49.0/60.0, -17.0/12.0, 47.0/18.0, -19.0/12.0,  31.0/60.0, -13.0/180.0 }, 
				  { -13.0/180.0,  19.0/15.0,  -7.0/3.0,  10.0/9.0,    1.0/12.0,  -1.0/15.0,   1.0/90.0 }, 
				  {   1.0/90.0,   -3.0/20.0,   3.0/2.0, -49.0/18.0,   3.0/2.0,   -3.0/20.0,   1.0/90.0 },
				  {   1.0/90.0,   -1.0/15.0,   1.0/12.0, 10.0/9.0,   -7.0/3.0,   19.0/15.0, -13.0/180.0 }, 
				  { -13.0/180.0,  31.0/60.0, -19.0/12.0, 47.0/18.0, -17.0/12.0, -49.0/60.0, 137.0/180.0 } };
				  
// Laplacian calculation consts for 2nd order 7-stencil
double laplacian_consts_2nd[3][5] = { { 11.0/12.0, -5.0/3.0,  1.0/2.0,  1.0/3.0, -1.0/12.0 }, 
				      { -1.0/12.0,  4.0/3.0, -5.0/2.0,  4.0/3.0, -1.0/12.0 }, 
				      { -1.0/12.0,  1.0/3.0,  1.0/2.0, -5.0/3.0, 11.0/12.0 } };
				
// Laplacian calculation consts for 1st order 7-stencil substituted into 3rd order laplacian calculation
double laplacian_consts_1st_3rd[5][7] = { { 1, -2,  1,  0,  0,   0,  0 }, 
				          { 0,  1, -2,  1,  0,   0,  0 }, 
				          { 0,  0,  1, -2,  1,   0,  0 }, 
				          { 0,  0,  0,  1, -2,   1,  0 }, 
				          { 0,  0,  0,  0,  1,  -2,  1 } };
				  
// Laplacian calculation consts for 1st order 7-stencil substituted into 2nd order laplacian calculation
double laplacian_consts_1st_2nd[3][5] = { { 1, -2,   1,   0,  0 }, 
				          { 0,  1,  -2,   1,  0 },
				          { 0,  0,   1,  -2,  1 } };


//******************************************************************** CONFIG PARAMETERS ********************************************************************//
// Simulation options
bool control;
bool trigger;
bool use_DCPD_GC1;
bool use_DCPD_GC2;
bool use_COD;
bool control_speed;
bool control_temperature;
bool const_target;
bool random_target;
bool target_switch;
bool laplacian_order_3;

// Mesh parameters
int num_vert_length;
int num_vert_width;
int num_vert_depth;

// Spatial parameters
double length;
double width;
double depth;

// Temporal parameters
double sim_duration;
double time_step;

// Fine mesh parameters
double length_fine;
int fine_time_steps_per_coarse;
int fine_steps_per_coarse_step_x;
int fine_steps_per_coarse_step_y;
int fine_steps_per_coarse_step_z;

// Problem definition
double temperature_limit;
double target_vel;
double vel_rand_scale;
double target_temp;
double temp_rand_scale;

// Front detection
double front_time_const;
double front_min_cure;
double front_max_cure;

// Initial conditions
double initial_temperature;
double initial_cure;
double initial_temp_delta;
double initial_cure_delta;

// Critical cure values
double critical_cure_rate;
double trans_cure_rate;

// Boundary conditions
double htc;
double ambient_temperature;
double default_htc;
double default_ambient_temperature;
double htc_delta;
double ambient_temperature_delta;

// Trigger parameters
double trigger_flux;
double trigger_time;
double trigger_duration;

// Input distribution parameters
double radius_of_input;
double laser_power;
double max_input_mag_percent_rate;
double max_input_loc_rate;

// Reward constants
double input_reward_const;
double overage_reward_const;
double front_shape_reward_const;
double target_reward_const;


//******************************************************************** CALCULATED PARAMETERS ********************************************************************//
// Simulation time and target velocity index
double current_time;
int current_index;
double time_step_fine;

// Front calculation parameters
int front_location_indicies_length;  // Length of memory to store front indices
double front_filter_alpha;

// Monomer physical parameters
double thermal_diffusivity;
double thermal_conductivity;
double enthalpy_of_reaction;
double specific_heat;
double cure_critical_temperature;  // The temperature below which the monomer cure rate is considered to be 0

// Target temporal vectors and the current target
double* target_vector;
double current_target;

// Mesh and step size
double*** mesh_x;
double*** mesh_y;
double*** mesh_z;
double x_step;
double y_step;
double z_step;
int num_vert_length_fine;
int num_vert_width_fine;
int num_vert_depth_fine;
double x_step_fine;
double y_step_fine;
double z_step_fine;
int coarse_steps_per_fine_mesh_x;
int coarse_steps_per_fine_mesh_y;
int coarse_steps_per_fine_mesh_z;
int fine_mesh_start_x_index;
int coarse_mesh_start_x_index;

// Temperature and cure meshes
double*** temp_mesh;
double*** laplacian_mesh;
double*** cure_mesh;
double*** temp_mesh_fine;
double*** cure_mesh_fine;
double*** laplacian_mesh_fine;

// Boundary temperatures
double*** lr_bc_temps;
double*** fb_bc_temps;
double*** tb_bc_temps;
double*** lr_bc_temps_fine;
double*** fb_bc_temps_fine;
double*** tb_bc_temps_fine;

// Front mesh and parameters
double** front_curve;
double*** threadwise_front_curve;
double front_mean_x_loc;
double front_shape_param;
double front_vel;
double front_temp;

// Input magnitude parameters
double exp_const;
double max_input_mag;  // Watts / m^2
double input_percent;  // Decimal percent

// Input location parameters
double min_input_x_loc;
double max_input_x_loc;
double min_input_y_loc;
double max_input_y_loc;
double* input_location;

// Input wattage mesh
double** input_mesh;


//******************************************************************** PRIVATE FUNCTIONS ********************************************************************//
void perturb_mesh(double*** arr, double delta);
void copy_coarse_to_fine();
void step_input(double x_loc_rate_action, double y_loc_rate_action, double mag_percent_rate_action);
void slide_fine_mesh_right();
void copy_fine_to_coarse();
int get_ind(int i);
void update_lr_bc_temps();
void update_fb_bc_temps();
void update_tb_bc_temps();
double get_laplacian(int i, int j, int k);
double get_laplacian_fine(int i, int j, int k);
void step_meshes();
bool step_time();
int load_config();
};
