#pragma once

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>

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
double get_current_target_front_vel();
double get_current_time();
vector<double> get_input_location();
vector<vector<double> > get_temp_mesh();
vector<vector<double> > get_cure_mesh();
vector<vector<double> > get_input_mesh();
vector<vector<double> > get_front_loc();
vector<vector<double> > get_front_vel();
vector<vector<double> > get_mesh_x_z0();
vector<vector<double> > get_mesh_y_z0();
vector<vector<double> > get_mesh_y_x0();
vector<vector<double> > get_mesh_z_x0();

//******************** FUNCTIONS ******************************//
tuple <vector<double>, double, bool> step(double x_loc_rate_action, double y_loc_rate_action, double mag_action);
vector<double> reset();


private:
//******************** USER SET PARAMETERS ********************//
// Simulation parameters
const bool random_target = false;
const bool target_switch = false;
const bool control = false;
const bool trigger = true;

// Mesh parameters
const int num_vert_length = 60;      // Unitless
const int num_vert_width = 12;        // Unitless
const int num_vert_depth = 6;        // Unitless

// Spatial parameters
const double length = 0.05;      // Meters
const double width = 0.01;       // Meters
const double depth = 0.005;      // Meters

// Temporal parameters
const double sim_duration = 240.0;      // Seconds
const double time_step = 0.1;           // Seconds

// Initial conditions
const double initial_temperature = 278.15;      // Kelvin
const double initial_cure = 0.05;               // Decimal Percent
const double initial_temp_delta = 3.0;          // Kelvin
const double initial_cure_delta = 0.005;        // Decimal Percent

// Boundary conditions
const double htc = 10.0;                        // Watts / (Meter ^ 2 * Kelvin)
const double ambient_temperature = 294.15;      // Kelvin

// Problem definition
const double temperature_limit = 573.15;        // Kelvin
const double target = 0.00015;                  // Meters / Second
const double randomizing_scale = 0.000025;      // Meters / Second

// Monomer physical parameters
const double thermal_conductivity = 0.152;         // Watts / Meter * Kelvin
const double density = 980.0;                      // Kilograms / Meter ^ 3
const double enthalpy_of_reaction = 352100.0;      // Joules / Kilogram
const double specific_heat = 1440.0;               // Joules / Kilogram * Kelvin
const double pre_exponential = 190985.325;         // 1 / Seconds
const double activiation_energy = 51100.0;        // Joules / Mol
const double gas_const = 8.3144;                   // Joules / Mol * Kelvin
const double model_fit_order = 1.927;              // Unitless
const double autocatalysis_const = 0.365;          // Unitless

// Input distribution parameters
const double radius_of_input = 0.005;           // Meters
const double laser_power = 0.2;                 // Watts
const double input_mag_percent_rate = 0.5;      // Decimal Percent / Second
const double max_input_loc_rate = 0.025;        // Meters / Second

// Set trigger condition references
const double trigger_flux_r = 25500.0;       // Watts / Meter ^ 2
const double trigger_time_r = 0.0;           // Seconds
const double trigger_duration_r = 10.0;      // Seconds

// NN Input conversion factors
const double mag_scale = 0.0227;            // Unitless
const double mag_offset = 0.5;              // Unitless
const double loc_rate_scale = 2.70e-4;      // Unitless
const double loc_rate_offset = 0.0;         // Unitless

// Reward constants
const double max_reward = 2.0;                      // Unitless
const double dist_punishment_const = 0.15;          // Unitless
const double front_rate_reward_const = 1.26;        // Unitless
const double input_punishment_const = 0.10;         // Unitless
const double overage_punishment_const = 0.40;       // Unitless
const double integral_punishment_const = 0.10;      // Unitless
const double front_shape_const = 10.0 / width;      // Unitless
const double max_integral = length * width * depth * temperature_limit;                         // Unitless
const double integral_delta = max_integral - length * width * depth * initial_temperature;      // Unitless

// Simulation stability limit
const double stab_lim = 10.0 * temperature_limit;

//******************** CALCULATED PARAMETERS ********************//
// Simulation time and target velocity index
double current_time;
int current_index;

// Monomer physical parameters
double thermal_diffusivity;

// Target velocity temporal vector and the current target
vector<double> target_front_vel;
double current_target_front_vel;

// Trigger conditions
double trigger_flux;          // Watts / Meter ^ 2
double trigger_time;          // Seconds
double trigger_duration;      // Seconds

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
vector<double> get_state();
double get_reward();
bool step_time();

};
