#pragma once
#include <math.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <tuple>
#include <iostream>
#include <chrono>

using namespace std;

class Finite_Element_Solver {

public:
//******************** CONSTRUCTOR ****************************//
Finite_Element_Solver();

//******************** GETTERS ********************************//
int get_num_vert_length();
int get_num_vert_width();
int get_num_vert_depth();
double get_sim_duration();
double get_time_step();
double get_input_percent();
double get_loc_rate_scale();
double get_mag_scale();
double get_max_input_mag();
double get_exp_const();
double get_current_target();
int get_target_vector_arr_size();
double get_current_time();
int get_num_state(int encoded_size);
bool get_control_speed();
void print_params();
vector<double> get_input_location();
vector<vector<double> > get_temp_mesh();
vector<vector<double> > get_norm_temp_mesh(int num_vert_sub_length);
vector<vector<double> > get_cure_mesh();
vector<vector<double> > get_input_mesh();
vector<vector<double> > get_front_loc();
vector<vector<double> > get_front_vel();
vector<vector<double> > get_front_temp();
vector<vector<double> > get_mesh_x_z0();
vector<vector<double> > get_mesh_y_z0();
vector<vector<double> > get_mesh_y_x0();
vector<vector<double> > get_mesh_z_x0();

//******************** FUNCTIONS ******************************//
bool step(double x_loc_rate_action, double y_loc_rate_action, double mag_action);
void reset();
vector<double> get_state(vector<double> encoded_temp, int encoded_size);
double get_reward();

private:
//******************** USER SET PARAMETERS ********************//
// Input type
const bool control = false;

// Trigger type
const bool trigger = true;

// Monomer type (only one can be true)
const bool use_DCPD_GC1 = false;
const bool use_DCPD_GC2 = true;
const bool use_COD = false;

// Control type (only one can be true)
const bool control_speed = true;
const bool control_temperature = false;

// Target type (only one can be true)
const bool const_target = true;
const bool random_target = false;
const bool target_switch = false;

// Mesh parameters
const int num_vert_length = 360;  // Unitless
const int num_vert_width = 40;     // Unitless
const int num_vert_depth = 6;    // Unitless

// Spatial parameters
const double length = 0.05;  // Meters
const double width = 0.01;   // Meters
const double depth = 0.005;  // Meters

// Temporal parameters
const double sim_duration = 60.0;   // Seconds
const double time_step = 0.03;         // Seconds

// Initial conditions
const double initial_temperature = 278.15;  // Kelvin
const double initial_cure = 0.05;           // Decimal Percent
const double initial_temp_delta = 3.0;      // Kelvin
const double initial_cure_delta = 0.005;    // Decimal Percent

// Boundary conditions
const double htc = 10.0;                    // Watts / (Meter ^ 2 * Kelvin)
const double ambient_temperature = 294.15;  // Kelvin

// Problem definition
const double temperature_limit = 523.15;      // Kelvin
const double DCPD_GC1_target_vel = 0.00015;       // Meters / Second
const double DCPD_GC1_vel_rand_scale = 0.000025;  // Meters / Second
const double DCPD_GC1_target_temp = 473.15;       // Kelvin
const double DCPD_GC1_temp_rand_scale = 20.0;     // Kelvin
const double DCPD_GC2_target_vel = 0.0015;       // Meters / Second
const double DCPD_GC2_vel_rand_scale = 0.00025;  // Meters / Second
const double DCPD_GC2_target_temp = 473.15;       // Kelvin
const double DCPD_GC2_temp_rand_scale = 20.0;     // Kelvin
const double COD_target_vel = 0.0005;         // Meters / Second
const double COD_vel_rand_scale = 0.0001;     // Meters / Second
const double COD_target_temp = 408.15;        // Kelvin
const double COD_temp_rand_scale = 20.0;      // Kelvin

// Physical constants
const double gas_const = 8.3144;  // Joules / Mol * Kelvin

// DCPD Monomer with GC1 physical parameters
const double DCPD_GC1_thermal_conductivity = 0.15;      // Watts / Meter * Kelvin
const double DCPD_GC1_density = 980.0;                  // Kilograms / Meter ^ 3
const double DCPD_GC1_enthalpy_of_reaction = 352100.0;  // Joules / Kilogram
const double DCPD_GC1_specific_heat = 1440.0;           // Joules / Kilogram * Kelvin
const double DCPD_GC1_pre_exponential = 190985.3;        // 1 / Seconds
const double DCPD_GC1_activiation_energy = 51100.0;    // Joules / Mol
const double DCPD_GC1_model_fit_order = 1.927;           // Unitless
const double DCPD_GC1_autocatalysis_const = 0.365;                    // Unitless

// DCPD Monomer with GC2 physical parameters
const double DCPD_GC2_thermal_conductivity = 0.15;      // Watts / Meter * Kelvin
const double DCPD_GC2_density = 980.0;                  // Kilograms / Meter ^ 3
const double DCPD_GC2_enthalpy_of_reaction = 350000.0;  // Joules / Kilogram
const double DCPD_GC2_specific_heat = 1600.0;           // Joules / Kilogram * Kelvin
const double DCPD_GC2_pre_exponential = 8.55e15;        // 1 / Seconds
const double DCPD_GC2_activiation_energy = 110750.0;    // Joules / Mol
const double DCPD_GC2_model_fit_order = 1.72;           // Unitless
const double DCPD_GC2_m_fit = 0.777;                    // Unitless
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

// Input distribution parameters
const double radius_of_input = 0.005;       // Meters
const double laser_power = 0.2;             // Watts
const double input_mag_percent_rate = 0.5;  // Decimal Percent / Second
const double max_input_loc_rate = 0.025;    // Meters / Second

// Set trigger condition references
const double DCPD_GC1_trigger_flux_ref = 25500.0;   // Watts / Meter ^ 2
const double DCPD_GC1_trigger_duration_ref = 10.0;  // Seconds
const double DCPD_GC2_trigger_flux_ref = 25500.0;   // Watts / Meter ^ 2
const double DCPD_GC2_trigger_duration_ref = 10.0;  // Seconds
const double COD_trigger_flux_ref = 20000.0;   // Watts / Meter ^ 2
const double COD_trigger_duration_ref = 6.0;  // Seconds
const double trigger_time_ref = 0.0;       // Seconds

// NN Input conversion factors
const double mag_scale = 0.0227;        // Unitless
const double mag_offset = 0.5;          // Unitless
const double loc_rate_scale = 0.001136;  // Unitless
const double loc_rate_offset = 0.0;     // Unitless

// Reward constants
const double max_reward = 2.0;                  // Unitless
const double dist_punishment_const = 0.15;      // Unitless
const double front_rate_reward_const = 1.00;    // Unitless
const double temperature_reward_const = 1.00;    // Unitless
const double input_punishment_const = 0.10;     // Unitless
const double overage_punishment_const = 0.40;   // Unitless
const double integral_punishment_const = 0.10;  // Unitless
const double front_shape_const = 10.0 / width;  // Unitless
const double max_integral = length * width * depth * temperature_limit;                     // Unitless
const double integral_delta = max_integral - length * width * depth * initial_temperature;  // Unitless

//******************** CALCULATED PARAMETERS ********************//
// Simulation time and target velocity index
double current_time;
int current_index;

// Monomer physical parameters
double thermal_diffusivity;
double thermal_conductivity;
double enthalpy_of_reaction;
double specific_heat;

// Target temporal vectors and the current target
vector<double> target_vector;
double current_target;

// Trigger conditions
double trigger_flux;      // Watts / Meter ^ 2
double trigger_time;      // Seconds
double trigger_duration;  // Seconds

// Mesh and step size
vector<vector<vector<double> > > mesh_x;
vector<vector<vector<double> > > mesh_y;
vector<vector<vector<double> > > mesh_z;
double x_step;
double y_step;
double z_step;

// Temperature and cure meshes
vector<vector<vector<double> > > temp_mesh;
vector<vector<vector<double> > > cure_mesh;

// Front mesh and parameters
vector<vector<double> > front_loc;
vector<vector<double> > front_vel;
vector<vector<double> > time_front_last_moved;
vector<vector<double> > front_temp;

// Input magnitude parameters
double exp_const;
double max_input_mag;
double input_percent;

// Input location parameters
double min_input_x_loc;
double max_input_x_loc;
double min_input_y_loc;
double max_input_y_loc;
vector<double> input_location;

// Input wattage mesh
vector<vector<double> > input_mesh;

//******************** FUNCTIONS ********************//
vector<vector<vector<double> > > get_perturbation(vector<vector<vector<double> > > size_array, double delta);
void step_input(double x_loc_rate_action, double y_loc_rate_action, double mag_action);
void step_meshes();
bool step_time();

};
