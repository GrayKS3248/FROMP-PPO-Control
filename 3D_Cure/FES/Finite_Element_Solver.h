#pragma once

#include <tuple>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <vector>

using namespace std;

class Finite_Element_Solver {

public:
    //******************** CONSTRUCTOR ****************************//
    Finite_Element_Solver();

    //******************** GETTERS ********************************//
    int get_num_vert_length();
    int get_num_vert_width();
    double get_sim_duration();
    double get_time_step();
    double get_input_percent();
    double get_loc_rate_scale();
    double get_mag_scale();
    double get_max_input_mag();
    double get_exp_const();
    double get_current_target_front_vel();
    double get_current_time();
    double* get_input_location();
    double** get_temp_mesh();
    double** get_cure_mesh();
    double** get_front_loc();
    double** get_front_vel();
    double** get_mesh_x();
    double** get_mesh_y();
    double** get_mesh_z();

    //******************** FUNCTIONS ******************************//
    tuple <vector<double>, double, bool> step(double x_loc_rate_action, double y_loc_rate_action, double mag_action);
    vector<double> reset();


private:
    //******************** USER SET PARAMETERS ********************//
    // Simulation parameters
    const bool random_target = false;
    const bool target_switch = false;
    const bool control = true;
    const bool trigger = true;

    // Mesh parameters
    const int num_vert_length = 120;  // Unitless
    const int num_vert_width = 24;    // Unitless
    const int num_vert_depth = 12;    // Unitless

    // Spatial parameters
    const double length = 0.05;  // Meters
    const double width = 0.01;   // Meters
    const double depth = 0.005;  // Meters

    // Temporal parameters
    const double sim_duration = 240.0;  // Seconds
    const double time_step = 0.5;       // Seconds

    // Initial conditions
    const double initial_temperature = 278.15;  // Kelvin
    const double initial_cure = 0.01;           // Decimal Percent

    // Boundary conditions
    const double htc = 6.0;                     // Watts / (Meter ^ 2 * Kelvin)
    const double ambient_temperature = 294.15;  // Kelvin

    // Problem definition
    const double temperature_limit = 573.15;  // Kelvin
    const double target = 0.00015;            // Meters / Second

    // Monomer physical parameters
    const double thermal_conductivity = 0.152;     // Watts / Meter * Kelvin
    const double density = 980.0;                  // Kilograms / Meter ^ 3
    const double enthalpy_of_reaction = 350000.0;  // Joules / Kilogram
    const double specific_heat = 1600.0;           // Joules / Kilogram * Kelvin
    const double pre_exponential = 8.55e5;         // 1 / Seconds
    const double activiation_energy = 110750.0;    // Joules / Mol
    const double gas_const = 8.3144;               // Joules / Mol * Kelvin
    const double m_exp = 0.770;                    // Unitless
    const double n_exp = 1.7215;                   // Unitless
    const double diffusion_const = 14.478;         // Unitless
    const double critical_conversion = 0.405;      // Unitless

    // Input distribution parameters
    const double radius_of_input = 0.005;       // Meters
    const double laser_power = 0.2;             // Watts
    const double input_mag_percent_rate = 0.5;  // Decimal Percent / Second
    const double max_input_loc_rate = 0.025;    // Meters / Second

    // Set trigger conditions
    const double trigger_flux = 20000.0;   // Watts / Meter ^ 2
    const double trigger_time = 0.0;       // Seconds
    const double trigger_duration = 10.0;  // Seconds

    // NN Input conversion factors
    const double mag_scale = 0.0227;        // Unitless
    const double mag_offset = 0.5;          // Unitless
    const double loc_rate_scale = 2.70e-4;  // Unitless
    const double loc_rate_offset = 0.0;     // Unitless

    // Reward constants
    const double max_reward = 2.0;                  // Unitless
    const double dist_punishment_const = 0.15;      // Unitless
    const double front_rate_reward_const = 1.26;    // Unitless
    const double input_punishment_const = 0.10;     // Unitless
    const double overage_punishment_const = 0.40;   // Unitless
    const double integral_punishment_const = 0.10;  // Unitless
    const double front_shape_const = 10.0 / width;  // Unitless

    //******************** CALCULATED PARAMETERS ********************//
    // Simulation time and target velocity index
    double current_time;
    int current_index;

    // Initial condition deltas
    double initial_temp_delta;
    double initial_cure_delta;

    // Randomizing scaling and problem type
    double randomizing_scale;

    // Target velocity temporal vector and the current target
    vector<double> target_front_vel = vector<double>((int)floor(sim_duration/time_step), target);
    int switch_location;
    double switch_vel;
    double current_target_front_vel;

    // Monomer physical parameters
    const double thermal_diffusivity = thermal_conductivity / (specific_heat * density);;

    // Mesh and step size
    vector<vector<vector<double>>> mesh_x = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0)));
    vector<vector<vector<double>>> mesh_y = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0)));
    vector<vector<vector<double>>> mesh_z = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0)));
    double x_step;
    double y_step;
    double z_step;

    // Temperature mesh
    vector<vector<vector<double>>> temp_mesh = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, initial_temperature)));

    // Cure mesh
    vector<vector<vector<double>>> cure_mesh = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, initial_cure)));

    // Front mesh
    vector<vector<vector<bool>>> front_mesh = vector<vector<vector<double>>>(num_vert_length, vector<vector<bool>>(num_vert_width, vector<bool>(num_vert_depth, false)));

    // Front parameters
    vector<vector<double>> front_loc = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
    vector<vector<double>> front_vel = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
    vector<vector<double>> time_front_last_moved = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
    vector<vector<bool>> front_has_started = vector<vector<bool>>(num_vert_width, vector<bool>(num_vert_depth, false));

    // Input magnitude parameters
    double exp_const;
    double max_input_mag;
    double input_percent;

    // Input location parameters
    double min_input_x_loc = 0.0;
    double max_input_x_loc = length;
    double min_input_y_loc = 0.0;
    double max_input_y_loc = width;

    // Reward constants
    const double max_integral = length * width * depth * temperature_limit;
    const double integral_delta = max_integral - length * width * depth * initial_temperature;

    // Input location
    vector<double> input_location(2, 0.0);

    // Input wattage mesh
    vector<vector<double>> input_mesh = vector<vector<double>>(num_vert_length, vector<double>(num_vert_width, 0.0));

    // Simulation stability limit
    const double stab_lim = 20.0 * temperature_limit;

    //******************** FUNCTIONS ********************//
    vector<vector<vector<double>>> get_perturbation(vector<vector<vector<double>>> size_array, double delta);
    void step_input(double x_loc_rate_action, double y_loc_rate_action, double mag_action);
    vector<vector<vector<double>>> step_cure();
    void step_front();
    void step_temperature();
    vector<vector<double>> blockshaped(vector<vector<double>> arr, int nrows, int ncols);
    vector<double> get_state();
    double get_reward();
    bool step_time();

};
