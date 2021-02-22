#include "Finite_Element_Solver.h"

using namespace std;

// Constructor
Finite_Element_Solver::Finite_Element_Solver()
{
    // Set randomization seed
    srand(time(NULL));

    //******************** CALCUALTED PARAMETERS ********************//
    // time
    current_time = 0.0;          // Seconds
    current_index = 0;              // Unitless

    // Monomer physical parameters
    thermal_diffusivity = thermal_conductivity / (specific_heat * density);

    // Calculate the target velocity temporal vector and define the current target
    int sim_steps = (int)(sim_duration / time_step);
    target_front_vel = vector<double>(sim_steps, 0.0);
    if (random_target)
    {
        double new_target = target - 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
        for (int i = 0; i < target_front_vel.size(); i++)
        {
            target_front_vel[i] = new_target;
        }
    }
    else if (target_switch)
    {
        int switch_location = (int) floor((0.20 * (double)rand()/(double)RAND_MAX + 0.40) * (double)(target_front_vel.size() - 1));
        double switch_vel = target_front_vel[switch_location] + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
        for (int i = switch_location; i < target_front_vel.size(); i++)
        {
            target_front_vel[i] = switch_vel;
        }
    }
    current_target_front_vel = target_front_vel[current_index];

    // Set trigger conditions
    if (!trigger)
    {
        trigger_flux = 0.0;      // Watts / Meter ^ 2
        trigger_time = 0.0;      // Seconds
        trigger_duration = 0.0;  // Seconds
    }
    else
    {
      trigger_flux = 25500.0;   // Watts / Meter ^ 2
      trigger_time = 0.0;       // Seconds
      trigger_duration = 10.0;  // Seconds
    }

    // Create mesh and calculate step size
    vector<double> x_range = vector<double>(num_vert_length, 0.0);
    vector<double> y_range = vector<double>(num_vert_width, 0.0);
    vector<double> z_range = vector<double>(num_vert_depth, 0.0);
    mesh_x = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth)));
    mesh_y = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth)));
    mesh_z = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth)));
    for (int i = 0; i < max(num_vert_length, max(num_vert_width, num_vert_depth)); i++)
    {
        if (i < num_vert_length)
        {
            x_range[i] = ((double)i / (double)(num_vert_length - 1)) * length;
        }
        if (i < num_vert_width)
        {
            y_range[i] = ((double)i / (double)(num_vert_width - 1)) * width;
        }
        if (i < num_vert_depth)
        {
            z_range[i] = ((double)i / (double)(num_vert_depth - 1)) * depth;
        }
    }
    for (int i = 0; i < num_vert_length; i++)
    {
        for (int j = 0; j < num_vert_width; j++)
        {
            for (int k = 0; k < num_vert_depth; k++)
            {
                mesh_x[i][j][k] = x_range[i];
                mesh_y[i][j][k] = y_range[j];
                mesh_z[i][j][k] = z_range[k];
            }
        }
    }
    x_step = mesh_x[1][0][0];
    y_step = mesh_y[0][1][0];
    z_step = mesh_z[0][0][1];

    // Init and perturb temperature and cure meshes
    temp_mesh = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, initial_temperature)));
    //temp_mesh = get_perturbation(temp_mesh, initial_temp_delta);
    cure_mesh = vector<vector<vector<double>>>(num_vert_length, vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, initial_cure)));
    //cure_mesh = get_perturbation(cure_mesh, initial_cure_delta);

    // Init front mesh and parameters
    front_mesh = vector<vector<vector<bool>>>(num_vert_length, vector<vector<bool>>(num_vert_width, vector<bool>(num_vert_depth, false)));
    front_loc = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
    front_vel = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
    time_front_last_moved = vector<vector<double>>(num_vert_width, vector<double>(num_vert_depth, 0.0));
    front_has_started = vector<vector<bool>>(num_vert_width, vector<bool>(num_vert_depth, false));

    // Input magnitude parameters
    double sigma = 0.329505114491 * radius_of_input;
    exp_const = -1.0 / (2.0 * sigma * sigma);
    double delta_x = (1.0 / 999.0) * radius_of_input;
    double x = 0.0;
    max_input_mag = 1.01;
    for (int i = 1; i < 999; i++)
    {
        x = ((double)i / 999.0) * radius_of_input;
        max_input_mag += 2.0 * pow(0.01, ((x * x) / (radius_of_input * radius_of_input)));
    }
    max_input_mag = laser_power / (4.0 * (max_input_mag * delta_x / 2.0) * (max_input_mag * delta_x / 2.0));
    input_percent = (double)rand()/(double)RAND_MAX;

    // Input location parameters
    min_input_x_loc = 0.0;
    max_input_x_loc = length;
    min_input_y_loc = 0.0;
    max_input_y_loc = width;
    input_location = vector<double>(2, 0.0);
    input_location[0] = mesh_x[(int)floor((double)rand()/(double)RAND_MAX * num_vert_length)][0][0];
    input_location[1] = mesh_y[0][(int)floor((double)rand()/(double)RAND_MAX * num_vert_width)][0];
    if (control)
    {
        max_input_mag = 0.0;
        min_input_x_loc = mesh_x[(int)floor(0.5 * num_vert_length)][0][0];
        max_input_x_loc = min_input_x_loc;
        min_input_y_loc = mesh_y[0][(int)floor(0.5 * num_vert_width)][0];
        max_input_y_loc = min_input_y_loc;
        input_location[0] = min_input_x_loc;
        input_location[1] = min_input_y_loc;
    }

    // Initiate input wattage mesh
    input_mesh = vector<vector<double>>(num_vert_length, vector<double>(num_vert_width, 0.0));
    double local_input_power = 0.0;
    for (int i = 0; i < num_vert_length; i++)
    {
        for (int j = 0; j < num_vert_width; j++)
        {
            local_input_power = input_percent * max_input_mag * exp(pow((mesh_x[i][j][0] - input_location[0]), 2.0) * exp_const + pow((mesh_y[i][j][0] - input_location[1]), 2.0) * exp_const);
            if (local_input_power < 0.01 * max_input_mag)
            {
                input_mesh[i][j] = 0.0;
            }
            else
            {
                input_mesh[i][j] = local_input_power;
            }
        }
    }
}
