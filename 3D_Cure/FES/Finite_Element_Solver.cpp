#include "Finite_Element_Solver.h"

using namespace std;

// Constructor
Finite_Element_Solver::Finite_Element_Solver()
{
    // Set randomization seed
    srand(time(NULL));
    RAND_MAX = 1.0;

    //******************** CALCUALTED PARAMETERS ********************//
    // Initiate simulation timeand target velocity index
    current_time = 0.0;
    current_index = 0;

    // Define initial condition deltas
    initial_temp_delta = 0.01 * initial_temperature;
    initial_cure_delta = 0.025 * initial_cure;

    // Define randomizing scaling and problem type
    randomizing_scale = target / 6.0;

    // Calculate the target velocity temporal vector and define the current target
    if (random_target)
    {
        double new_target = target - 2.0 * (rand() - 0.5) * randomizing_scale;
        for (int i = 0; i < target_front_vel.size(); i++)
        {
            target_front_vel[i] = new_target;
        }
    }
    else if (target_switch)
    {
        int switch_location = (int) floor((0.20 * rand() + 0.40) * (double)(target_front_vel.size() - 1));
        double switch_vel = target_front_vel[switch_location] + 2.0 * (rand() - 0.5) * randomizing_scale;
        for (int i = switch_location; i < target_front_vel.size(); i++)
        {
            target_front_vel[i] = switch_vel;
        }
    }
    current_target_front_vel = target_front_vel[current_index];

    // Set trigger conditions
    if (!trigger)
    {
        trigger_flux = 0.0;
        trigger_time = 0.0;
        trigger_duration = 0.0;
    }

    // Create mesh and calculate step size
    double* x_range[num_vert_length];
    double* y_range[num_vert_width];
    double* z_range[num_vert_depth];
    for (int i = 0; i < max(max(num_vert_length, num_vert_width), num_vert_depth); i++)
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

    // Perturb temperature mesh
    temp_mesh = get_perturbation(temp_mesh, initial_temp_delta);

    // Perturb cure mesh
    cure_mesh = get_perturbation(cure_mesh, initial_cure_delta);

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

    // Initiate input
    input_percent = rand();
    input_location[0] = mesh_x[(int)floor(rand() * num_vert_length)][0][0];
    input_location[1] = mesh_y[0][(int)floor(rand() * num_vert_width)][0];
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