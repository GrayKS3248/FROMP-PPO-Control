#pragma once
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <deque>
#include <sstream>
#include <iomanip>
#include "matrix_math.hpp"
#include "Config_Handler.hpp"

using namespace std;

class Finite_Difference_Solver {
	

//******************************************************************** CONSTRUCTOR/DESTRUCTOR ********************************************************************//
public:
Finite_Difference_Solver();
~Finite_Difference_Solver();


//******************************************************************** GETTERS ********************************************************************//
// Mesh getters
int get_num_coarse_vert_x();
int get_num_coarse_vert_y();
int get_num_coarse_vert_z();
double get_coarse_x_len();
double get_coarse_y_len();
double get_coarse_z_len();
double get_volume();
double get_surface_area();
vector<vector<double>> get_coarse_x_mesh_z0();
vector<vector<double>> get_coarse_y_mesh_z0();
vector<double> get_fine_mesh_loc();

// Time getters
double get_sim_duration();
int get_num_sim_steps();
double get_coarse_time_step();
double get_curr_sim_time();

// Input getters
vector<double> get_input_state(bool normalize);
double get_peak_input_mag();
double get_input_const();
double get_max_input_mag_percent_rate();
double get_max_input_slew_speed();
double get_trigger_power();
double get_source_power();

// Target getters
double get_curr_target();

// Monomer getters
double get_monomer_burn_temp();
double get_adiabatic_temp_of_rxn();
double get_specific_heat();
double get_density();

// Boundary condition getters
double get_initial_temp();
double get_heat_transfer_coefficient();
double get_ambient_temperature();

// Sim option getter
bool get_control_mode();

// Temperature and cure getters
vector<vector<double>> get_coarse_temp_z0(bool normalize);
vector<vector<double>> get_fine_temp_z0(bool normalize);
vector<vector<double>> get_coarse_cure_z0();
vector<vector<double>> get_fine_cure_z0();

// Front state getters
vector<vector<double>> get_front_curve();
vector<double> get_front_fit(unsigned int order);
double get_front_mean_x_loc(bool normalize);
double get_front_vel(bool normalize);
double get_front_temp(bool normalize);
double get_front_shape_param();


//******************************************************************** PUBLIC FUNCTIONS ********************************************************************//
void print_params();
void print_progress(bool return_carriage);
double get_progress();
void reset();
bool step(double x_cmd, double y_cmd, double mag_cmd);
vector<double> get_reward();


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
				
// Laplacian calculation consts for 6th order 7-stencil
double laplacian_consts[5][7] = { { 137.0/180.0, -49.0/60.0, -17.0/12.0,  47.0/18.0, -19.0/12.0,  31.0/60.0, -13.0/180.0 }, 
			          { -13.0/180.0,  19.0/15.0,   -7.0/3.0,   10.0/9.0,   1.0/12.0,  -1.0/15.0,    1.0/90.0 }, 
			          {    1.0/90.0,  -3.0/20.0,    3.0/2.0, -49.0/18.0,    3.0/2.0,  -3.0/20.0,    1.0/90.0 },
			          {    1.0/90.0,  -1.0/15.0,   1.0/12.0,   10.0/9.0,   -7.0/3.0,  19.0/15.0, -13.0/180.0 }, 
			          { -13.0/180.0,  31.0/60.0, -19.0/12.0,  47.0/18.0, -17.0/12.0, -49.0/60.0, 137.0/180.0 } };


//******************************************************************** CONFIG PARAMETERS ********************************************************************//
// Simulation options
bool input_is_on;
bool using_a_trigger;
int monomer_code;
int control_code;
int target_code;

// Coarse mesh parameters
int num_coarse_vert_x;
int num_coarse_vert_y;
int num_coarse_vert_z;
double coarse_x_len;
double coarse_y_len;
double coarse_z_len;

// Fine mesh parameters
double fine_x_len;
int fine_x_resolution_multiplier;
int fine_y_resolution_multiplier;
int fine_z_resolution_multiplier;

// Temporal parameters
double sim_duration;
double coarse_time_step;
int fine_time_steps_per_coarse_time_step;

// Problem definition
double monomer_burn_temp;
double mean_target_speed;
double max_target_speed_deviation;
double mean_target_temp;
double max_target_temp_deviation;

// Initial conditions
double initial_temp;
double initial_cure;
double max_initial_temp_deviation;
double max_initial_cure_deviation;

// Boundary conditions
double mean_htc;
double max_htc_deviation;
double mean_amb_temp;
double max_amb_temp_deviation;

// Front detection
double front_filter_time_const;
double front_mean_x_loc_history_time_len;
double front_min_cure;
double front_max_cure;
double front_min_cure_rate;

// Critical cure values
double critical_cure_rate;
double transition_cure_rate;

// Precalculated array parameters for cure rate based on temp
double precalc_start_temp;
double precalc_end_temp;
double precalc_temp_step;

// Precalculated array parameters for cure rate based on cure
double precalc_start_cure;
double precalc_end_cure;
double precalc_cure_step;

// Trigger parameters
double trigger_flux;
double trigger_time;
double trigger_duration;

// Input distribution parameters
double radius_of_input;
double input_total_power;
double max_input_mag_percent_rate;
double max_input_slew_speed;

// Reward constants
double input_loc_reward_const;
double input_mag_reward_const;
double max_temp_reward_const;
double front_shape_reward_const;
double target_reward_const;


//******************************************************************** CALCULATED PARAMETERS ********************************************************************//
// Calculated monomer physical parameters
double thermal_diffusivity;
double thermal_conductivity;
double enthalpy_of_reaction;
double specific_heat;
double adiabatic_temp_of_rxn;
double critical_temp;

// Calculated coarse mesh parameters
double*** coarse_x_mesh;
double*** coarse_y_mesh;
double*** coarse_z_mesh;
double*** coarse_temp_mesh;
double*** coarse_laplacian_mesh;
double*** coarse_cure_mesh;
double coarse_x_step;
double coarse_y_step;
double coarse_z_step;
int coarse_x_verts_per_fine_x_len;
int coarse_y_verts_per_fine_y_len;
int coarse_z_verts_per_fine_z_len;
int coarse_x_index_at_fine_mesh_start;

// Calcualted fine mesh parameters
double*** fine_temp_mesh;
double*** fine_laplacian_mesh;
double*** fine_cure_mesh;
int num_fine_vert_x;
int num_fine_vert_y;
int num_fine_vert_z;
double fine_y_len;
double fine_z_len;
double fine_x_step;
double fine_y_step;
double fine_z_step;
int fine_mesh_zero_index;
double fine_mesh_start_loc;
double fine_mesh_end_loc;

// Simulation time and step parameters
double curr_sim_time;
int curr_sim_step;
double fine_time_step;

// Problem definition
double mean_target;
double max_target_deviation;
double* target_arr;

// Calculated boundary conditions
double htc;
double amb_temp;
double min_possible_temp;
double*** coarse_lr_bc_temps;
double*** coarse_fb_bc_temps;
double*** coarse_tb_bc_temps;
double*** fine_lr_bc_temps;
double*** fine_fb_bc_temps;
double*** fine_tb_bc_temps;

// Front calculation parameters
int num_front_instances;
double**  global_front_curve;
double*** thread_front_curve;
double front_vel;
double front_temp;
double front_shape_param;
double front_mean_x_loc;
deque<double> front_mean_x_loc_history;
int front_mean_x_loc_history_len;
double front_mean_x_loc_history_avg;
int max_front_instances;
double front_filter_alpha;

// Precalculated arrays for cure rate
double* precalc_exp_arr;
int precalc_exp_arr_len;
double* precalc_pow_arr;
int precalc_pow_arr_len;
int arg_max_precalc_pow_arr;
double max_precalc_pow_arr;

// Trigger conditions
bool trigger_is_on;

// Calculated input parameters
double input_const;
double peak_input_mag;
double min_input_x_loc;
double max_input_x_loc;
double min_input_y_loc;
double max_input_y_loc;

// Input values
double** input_mesh;
double* input_location;
double input_percent;


//******************************************************************** PRIVATE FUNCTIONS ********************************************************************//
int load_config();
void perturb_mesh(double*** arr, double max_deviation);
void step_trigger();
void step_input(double x_cmd, double y_cmd, double mag_cmd);
int get_ind(int i);
void copy_coarse_to_fine();
void slide_fine_mesh_right();
void copy_fine_to_coarse();
void update_lr_bc_temps();
void update_fb_bc_temps();
void update_tb_bc_temps();
double get_coarse_laplacian(int i, int j, int k);
double get_fine_laplacian(int i, int j, int k);
void step_meshes();
bool step_time();
};