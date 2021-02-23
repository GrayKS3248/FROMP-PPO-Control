#include "Finite_Element_Solver.h"

using namespace std;

/**
 * Default constructor
 */
Finite_Element_Solver::Finite_Element_Solver()
{
        // Set randomization seed
        srand(time(NULL));

        // Simulation time and target velocity index
        current_time = 0.0; // Seconds
        current_index = 0; // Unitless

        // Monomer physical parameters
        thermal_diffusivity = thermal_conductivity / (specific_heat * density);

        // Calculate the target velocity temporal vector and define the current target
        int sim_steps = (int)(sim_duration / time_step);
        target_front_vel = vector<double>(sim_steps, 0.0);
        if (random_target)
        {
                double new_target = target - 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
                for (unsigned int i = 0; i < target_front_vel.size(); i++)
                {
                        target_front_vel[i] = new_target;
                }
        }
        else if (target_switch)
        {
                int switch_location = (int) floor((0.20 * (double)rand()/(double)RAND_MAX + 0.40) * (double)(target_front_vel.size() - 1));
                double switch_vel = target_front_vel[switch_location] + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
                for (unsigned int i = switch_location; i < target_front_vel.size(); i++)
                {
                        target_front_vel[i] = switch_vel;
                }
        }
        current_target_front_vel = target_front_vel[current_index];

        // Set trigger conditions
        if (!trigger)
        {
                trigger_flux = 0.0; // Watts / Meter ^ 2
                trigger_time = 0.0; // Seconds
                trigger_duration = 0.0; // Seconds
        }
        else
        {
                trigger_flux = trigger_flux_r; // Watts / Meter ^ 2
                trigger_time = trigger_time_r; // Seconds
                trigger_duration = trigger_duration_r; // Seconds
        }

        // Create mesh and calculate step size
        vector<double> x_range(num_vert_length, 0.0);
        vector<double> y_range(num_vert_width, 0.0);
        vector<double> z_range(num_vert_depth, 0.0);
        mesh_x = vector<vector<vector<double> > >(num_vert_length, vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth)));
        mesh_y = vector<vector<vector<double> > >(num_vert_length, vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth)));
        mesh_z = vector<vector<vector<double> > >(num_vert_length, vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth)));
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
        temp_mesh = vector<vector<vector<double> > >(num_vert_length, vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth, initial_temperature)));
        temp_mesh = get_perturbation(temp_mesh, initial_temp_delta);
        cure_mesh = vector<vector<vector<double> > >(num_vert_length, vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth, initial_cure)));
        cure_mesh = get_perturbation(cure_mesh, initial_cure_delta);

        // Init front mesh and parameters
        front_loc = vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth, 0.0));
        front_vel = vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth, 0.0));
        time_front_last_moved = vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth, 0.0));

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
        input_mesh = vector<vector<double> >(num_vert_length, vector<double>(num_vert_width, 0.0));
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

/**
 * Gets the number of vertices in lengthwise direction
 * @return The number of vertices in the lengthwise direction
 */
int Finite_Element_Solver::get_num_vert_length()
{
        return num_vert_length;
}

/**
 * Gets the number of vertices in widthwise direction
 * @return The number of vertices in the widthwise direction
 */
int Finite_Element_Solver::get_num_vert_width()
{
        return num_vert_width;
}

/**
 * Gets the number of vertices in depthwise direction
 * @return The number of vertices in the depthwise direction
 */
int Finite_Element_Solver::get_num_vert_depth()
{
        return num_vert_depth;
}

/**
 * Gets the duration of the simulation
 * @return The duration of the simulation in seconds
 */
double Finite_Element_Solver::get_sim_duration()
{
        return sim_duration;
}

/**
 * Gets the time step used in the simulation
 * @return The simulation time step in seconds
 */
double Finite_Element_Solver::get_time_step()
{
        return time_step;
}

/**
 * Gets the current input power percent
 * @return The power level of the input in percent
 */
double Finite_Element_Solver::get_input_percent()
{
        return input_percent;
}

/**
 * Gets the location rate scaling factor
 * @return The location rate scaling factor that takes NN raw to FES
 */
double Finite_Element_Solver::get_loc_rate_scale()
{
        return loc_rate_scale;
}

/**
 * Gets the magnitude scaling factor
 * @return The magnitude scaling factor that takes NN raw to FES
 */
double Finite_Element_Solver::get_mag_scale()
{
        return mag_scale;
}

/**
 * Gets the maximum magnitude of the input in W/m^2
 * @return The peak magnitude of the input in W/m^2
 */
double Finite_Element_Solver::get_max_input_mag()
{
        return max_input_mag;
}

/**
 * Gets the exponent constant used to calculate input mesh
 * @return The exponent constant used to calculate input mesh in W/m^2
 */
double Finite_Element_Solver::get_exp_const()
{
        return exp_const;
}

/**
 * Gets the current target velocity
 * @return The current target front velocity in m/s
 */
double Finite_Element_Solver::get_current_target_front_vel()
{
        return current_target_front_vel;
}

/**
 * Gets the current time
 * @return The time in seconds
 */
double Finite_Element_Solver::get_current_time()
{
        return current_time;
}

/**
 * Gets the input location
 * @return The input location as a vector {x,y}
 */
vector<double> Finite_Element_Solver::get_input_location()
{
        return input_location;
}

/**
 * Gets the top layer of the temperature mesh
 * @return The top layer of the temperature mesh as a 2D vector in x,y
 */
vector<vector<double> > Finite_Element_Solver::get_temp_mesh()
{
        vector<vector<double> > ret_val(num_vert_length, vector<double>(num_vert_width, 0.0));
        for (int i = 0; i < num_vert_length; i++)
        {
                for (int j = 0; j < num_vert_width; j++)
                {
                        ret_val[i][j] = temp_mesh[i][j][0];
                }
        }
        return ret_val;
}

/**
 * Gets the top layer of the cure mesh
 * @return The top layer of the cure mesh as a 2D vector in x,y
 */
vector<vector<double> > Finite_Element_Solver::get_cure_mesh()
{
        vector<vector<double> > ret_val(num_vert_length, vector<double>(num_vert_width, 0.0));
        for (int i = 0; i < num_vert_length; i++)
        {
                for (int j = 0; j < num_vert_width; j++)
                {
                        ret_val[i][j] = cure_mesh[i][j][0];
                }
        }
        return ret_val;
}

/**
 * Gets the input mesh
 * @return The input mesh as a 2D vector in x,y of watts/m^2
 */
vector<vector<double> > Finite_Element_Solver::get_input_mesh()
{
        return input_mesh;
}

/**
 * Gets the current front location
 * @return The current front location as a 2D vector in x,y
 */
vector<vector<double> > Finite_Element_Solver::get_front_loc()
{
        return front_loc;
}

/**
 * Gets the current front velocity
 * @return The current front velocity as a 2D vector in x,y
 */
vector<vector<double> > Finite_Element_Solver::get_front_vel()
{
        return front_vel;
}

/**
 * Gets the top layer of the x mesh
 * @return The top layer of the x mesh as a 2D vector in x,y
 */
vector<vector<double> > Finite_Element_Solver::get_mesh_x_z0()
{
        vector<vector<double> > ret_val(num_vert_length, vector<double>(num_vert_width, 0.0));
        for (int i = 0; i < num_vert_length; i++)
        {
                for (int j = 0; j < num_vert_width; j++)
                {
                        ret_val[i][j] = mesh_x[i][j][0];
                }
        }
        return ret_val;
}

/**
 * Gets the top layer of the y mesh
 * @return The top layer of the y mesh as a 2D vector in x,y
 */
vector<vector<double> > Finite_Element_Solver::get_mesh_y_z0()
{
        vector<vector<double> > ret_val(num_vert_length, vector<double>(num_vert_width, 0.0));
        for (int i = 0; i < num_vert_length; i++)
        {
                for (int j = 0; j < num_vert_width; j++)
                {
                        ret_val[i][j] = mesh_y[i][j][0];
                }
        }
        return ret_val;
}

/**
 * Gets the left layer of the y mesh
 * @return The left layer of the y mesh as a 2D vector in y,z
 */
vector<vector<double> > Finite_Element_Solver::get_mesh_y_x0()
{
        vector<vector<double> > ret_val(num_vert_width, vector<double>(num_vert_depth, 0.0));
        for (int i = 0; i < num_vert_width; i++)
        {
                for (int j = 0; j < num_vert_depth; j++)
                {
                        ret_val[i][j] = mesh_y[0][i][j];
                }
        }
        return ret_val;
}

/**
 * Gets the left layer of the z mesh
 * @return The left layer of the z mesh as a 2D vector in y,z
 */
vector<vector<double> > Finite_Element_Solver::get_mesh_z_x0()
{
        vector<vector<double> > ret_val(num_vert_width, vector<double>(num_vert_depth, 0.0));
        for (int i = 0; i < num_vert_width; i++)
        {
                for (int j = 0; j < num_vert_depth; j++)
                {
                        ret_val[i][j] = mesh_z[0][i][j];
                }
        }
        return ret_val;
}

/**
 * Steps the environment forward one time step
 * @param The raw NN x location rate command
 * @param The raw NN y location rate command
 * @param The raw NN magnitude command
 * @return A tuple of the form <state, reward, done>
 */
tuple <vector<double>, double, bool> Finite_Element_Solver::step(double x_loc_rate_action, double y_loc_rate_action, double mag_action)
{
        // Step the input, cure, front, and temperature
        step_input(x_loc_rate_action, y_loc_rate_action, mag_action);
        step_meshes();

        // Get state and Reward
        vector<double> state = get_state();
        double reward = get_reward();

        // Step time_step
        bool done = step_time();

        // Create and return tuple
        tuple<vector<double>, double, bool> ret_val;
        ret_val = make_tuple(state, reward, done);
        return ret_val;
}

/**
 * Resets the environment to initial conditions
 * @return the initial state vector
 */
vector<double> Finite_Element_Solver::reset()
{
        // Simulation time and target velocity index
        current_time = 0.0; // Seconds
        current_index = 0; // Unitless

        // Calculate the target velocity temporal vector and define the current target
        int sim_steps = (int)(sim_duration / time_step);
        target_front_vel = vector<double>(sim_steps, 0.0);
        if (random_target)
        {
                double new_target = target - 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
                for (unsigned int i = 0; i < target_front_vel.size(); i++)
                {
                        target_front_vel[i] = new_target;
                }
        }
        else if (target_switch)
        {
                int switch_location = (int) floor((0.20 * (double)rand()/(double)RAND_MAX + 0.40) * (double)(target_front_vel.size() - 1));
                double switch_vel = target_front_vel[switch_location] + 2.0 * ((double)rand()/(double)RAND_MAX - 0.5) * randomizing_scale;
                for (unsigned int i = switch_location; i < target_front_vel.size(); i++)
                {
                        target_front_vel[i] = switch_vel;
                }
        }
        current_target_front_vel = target_front_vel[current_index];

        // Init and perturb temperature and cure meshes
        temp_mesh = vector<vector<vector<double> > >(num_vert_length, vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth, initial_temperature)));
        temp_mesh = get_perturbation(temp_mesh, initial_temp_delta);
        cure_mesh = vector<vector<vector<double> > >(num_vert_length, vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth, initial_cure)));
        cure_mesh = get_perturbation(cure_mesh, initial_cure_delta);

        // Init front mesh and parameters
        front_loc = vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth, 0.0));
        front_vel = vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth, 0.0));
        time_front_last_moved = vector<vector<double> >(num_vert_width, vector<double>(num_vert_depth, 0.0));

        // Input magnitude parameters
        input_percent = (double)rand()/(double)RAND_MAX;

        // Input location parameters
        input_location = vector<double>(2, 0.0);
        if (control)
        {
                input_location[0] = min_input_x_loc;
                input_location[1] = min_input_y_loc;
        }
        else
        {
                input_location[0] = mesh_x[(int)floor((double)rand()/(double)RAND_MAX * num_vert_length)][0][0];
                input_location[1] = mesh_y[0][(int)floor((double)rand()/(double)RAND_MAX * num_vert_width)][0];
        }

        // Initiate input wattage mesh
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

        // Return the state
        return get_state();

}

/** Get smooth 3D perturbation over input fields
 * @ param array used to determine size of output mesh
 * @ param maximum magnitude of perturbation
 * @ return sum of size_array and smooth continuous perturbation of magnitude delta
 */
vector<vector<vector<double> > > Finite_Element_Solver::get_perturbation(vector<vector<vector<double> > > size_array, double delta)
{
        // Get magnitude and biases
        double mag_1 = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;
        double mag_2 = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;
        double mag_3 = 2.0 * (double)rand()/(double)RAND_MAX - 1.0;
        double bias_1 = 4.0 * M_PI * (double)rand()/(double)RAND_MAX - 2.0 * M_PI;
        double bias_2 = 4.0 * M_PI * (double)rand()/(double)RAND_MAX - 2.0 * M_PI;
        double bias_3 = 4.0 * M_PI * (double)rand()/(double)RAND_MAX - 2.0 * M_PI;
        double min_mag = (double)rand()/(double)RAND_MAX;
        double max_mag = (double)rand()/(double)RAND_MAX;
        double min_x_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;
        double max_x_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;
        double min_y_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;
        double max_y_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;
        double min_z_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;
        double max_z_bias = 2.0*(double)rand()/(double)RAND_MAX-1.0;

        // Get x*y*z over perturbation field
        double x;
        double y;
        double z;
        double xyz;
        double scale = 0.0;
        vector<vector<vector<double> > > perturbation = vector<vector<vector<double> > >(size_array.size(), vector<vector<double> >(size_array[0].size(), vector<double>(size_array[0][0].size(), 0.0)));
        for (unsigned int i = 0; i < size_array.size(); i++)
        {
                x = -2.0*min_mag+min_x_bias + (2.0*max_mag+max_x_bias + 2.0*min_mag+min_x_bias) * ((double)i / (double)size_array.size());
                for (unsigned int j = 0; j < size_array[0].size(); j++)
                {
                        y = -2.0*min_mag+min_y_bias + (2.0*max_mag+max_y_bias + 2.0*min_mag+min_y_bias) * ((double)i / (double)size_array[0].size());
                        for (unsigned int k = 0; k < size_array[0][0].size(); k++)
                        {
                                z =-2.0*min_mag+min_z_bias + (2.0*max_mag+max_z_bias + 2.0*min_mag+min_z_bias) * ((double)i / (double)size_array[0][0].size());
                                xyz = x * y * z;
                                perturbation[i][j][k] = mag_1 * sin(xyz + bias_1) + mag_2 * sin(2.0*xyz + bias_2) + mag_3 * sin(3.0*xyz + bias_3);
                                if (abs(perturbation[i][j][k]) > scale)
                                {
                                        scale = abs(perturbation[i][j][k]);
                                }
                        }
                }
        }

        // Scale the perturbations and sum them to the original array
        for (unsigned int i = 0; i < size_array.size(); i++)
        {
                for (unsigned int j = 0; j < size_array[0].size(); j++)
                {
                        for (unsigned int k = 0; k < size_array[0][0].size(); k++)
                        {
                                perturbation[i][j][k] = size_array[i][j][k] + (delta * perturbation[i][j][k]) / scale;
                        }
                }
        }

        // Return perturbed array
        return perturbation;
}

/** Step the input through time
 * @param The raw NN x location rate command
 * @param The raw NN y location rate command
 * @param The raw NN magnitude command
 */
void Finite_Element_Solver::step_input(double x_loc_rate_action, double y_loc_rate_action, double mag_action)
{
        // Convert the raw PPO x command to usable, clipped x location rate command
        double cmd_x = loc_rate_offset + loc_rate_scale * x_loc_rate_action;
        cmd_x = cmd_x > max_input_loc_rate ? max_input_loc_rate : cmd_x;
        cmd_x = cmd_x < -max_input_loc_rate ? -max_input_loc_rate : cmd_x;

        // Convert the raw PPO y command to usable, clipped y location rate command
        double cmd_y = loc_rate_offset + loc_rate_scale * y_loc_rate_action;
        cmd_y = cmd_y > max_input_loc_rate ? max_input_loc_rate : cmd_y;
        cmd_y = cmd_y < -max_input_loc_rate ? -max_input_loc_rate : cmd_y;

        // Update the input's x location from the converted location rate commands
        input_location[0] = input_location[0] + cmd_x * time_step;
        input_location[0] = input_location[0] > max_input_x_loc ? max_input_x_loc : input_location[0];
        input_location[0] = input_location[0] < min_input_x_loc ? min_input_x_loc : input_location[0];

        // Update the input's y location from the converted location rate commands
        input_location[1] = input_location[1] + cmd_y * time_step;
        input_location[1] = input_location[1] > max_input_y_loc ? max_input_y_loc : input_location[1];
        input_location[1] = input_location[1] < min_input_y_loc ? min_input_y_loc : input_location[1];

        // Convert the raw PPO command to a usable, clipped input percent command
        double input_percent_command = mag_offset + mag_scale * mag_action;
        input_percent_command = input_percent_command > 1.0 ? 1.0 : input_percent_command;
        input_percent_command = input_percent_command < 0.0 ? 0.0 : input_percent_command;

        // Update the input's magnitude from the converted input percent command
        if (input_percent_command > input_percent)
        {
                input_percent = input_percent + input_mag_percent_rate * time_step;
                input_percent = input_percent > input_percent_command ? input_percent_command : input_percent;
        }
        else if (input_percent_command < input_percent)
        {
                input_percent = input_percent - input_mag_percent_rate * time_step;
                input_percent = input_percent < input_percent_command ? input_percent_command : input_percent;
        }

        // Update the input wattage mesh
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < num_vert_length; i++)
                for (int j = 0; j < num_vert_width; j++)
                {
                        double local_input_power = input_percent * max_input_mag * exp(pow((mesh_x[i][j][0] - input_location[0]), 2.0) * exp_const + pow((mesh_y[i][j][0] - input_location[1]), 2.0) * exp_const);
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

/** Calculates the cure rate at every point in the 3D mesh and uses this data to update the cure, temperature, and front meshes
 */
void Finite_Element_Solver::step_meshes()
{
        // Front mesh variables
        const vector<vector<double> > prev_front_loc(front_loc);
        const vector<vector<double> > prev_last_move(time_front_last_moved);

        // Temperature mesh variables
        const vector<vector<vector<double> > > prev_temp(temp_mesh);

        #pragma omp parallel for collapse(3)
        for (unsigned int i = 0; i < mesh_x.size(); i++)
                for (unsigned int j = 0; j < mesh_x[0].size(); j++)
                        for (unsigned int k = 0; k < mesh_x[0][0].size(); k++)
                        {
                                // Calculate the cure rate
                                double cure_rate = pre_exponential * exp(-activiation_energy / (gas_const * prev_temp[i][j][k])) *  pow((1.0 - cure_mesh[i][j][k]), model_fit_order) * (1.0 + autocatalysis_const * cure_mesh[i][j][k]);

                                // Update the cure mesh
                                cure_mesh[i][j][k] = cure_mesh[i][j][k] + cure_rate * time_step;
                                cure_mesh[i][j][k] = cure_mesh[i][j][k] > 1.0 ? 1.0 : cure_mesh[i][j][k];

                                // Update the front location and velocity
                                if ((cure_mesh[i][j][k] >= 0.80) && (front_loc[j][k] <= mesh_x[i][j][k]))
                                {
                                        front_loc[j][k] = mesh_x[i][j][k];
                                        if (prev_last_move[j][k] != 0.0)
                                        {
                                                front_vel[j][k] = (front_loc[j][k] - prev_front_loc[j][k]) / (current_time - prev_last_move[j][k]);
                                        }
                                        time_front_last_moved[j][k] = current_time;
                                }

                                // Temperature variables
                                double dT2_dx2;
                                double dT2_dy2;
                                double dT2_dz2;
                                double left_flux;
                                double right_flux;
                                double front_flux;
                                double back_flux;
                                double top_flux;
                                double bottom_flux;

                                // Calculate the second derivative of temperature wrt x
                                if (i != 0 && i != mesh_x.size()-1)
                                {
                                        dT2_dx2 = (prev_temp[i+1][j][k] - 2.0*prev_temp[i][j][k] + prev_temp[i-1][j][k]) / (x_step*x_step);
                                }
                                else
                                {
                                        if (i == 0)
                                        {
                                                if (current_time >= trigger_time && current_time < trigger_time + trigger_duration)
                                                {
                                                        left_flux = htc*(prev_temp[i][j][k]-ambient_temperature) - trigger_flux;
                                                }
                                                else
                                                {
                                                        left_flux = htc*(prev_temp[i][j][k]-ambient_temperature);
                                                }
                                                dT2_dx2 = 2.0*(prev_temp[i+1][j][k]-prev_temp[i][j][k]-(x_step*left_flux/thermal_conductivity))/(x_step*x_step);
                                        }
                                        if (i == mesh_x.size()-1)
                                        {
                                                right_flux = htc*(prev_temp[i][j][k]-ambient_temperature);
                                                dT2_dx2 = 2.0*(prev_temp[i-1][j][k]-prev_temp[i][j][k]-(x_step*right_flux/thermal_conductivity))/(x_step*x_step);
                                        }
                                }

                                // Calculate the second derivative of temperature wrt y
                                if (j != 0 && j != mesh_x[0].size()-1)
                                {
                                        dT2_dy2 = (prev_temp[i][j+1][k] - 2.0*prev_temp[i][j][k] + prev_temp[i][j-1][k]) / (y_step*y_step);
                                }
                                else
                                {
                                        if (j == 0)
                                        {
                                                front_flux = htc*(prev_temp[i][j][k]-ambient_temperature);
                                                dT2_dy2 = 2.0*(prev_temp[i][j+1][k]-prev_temp[i][j][k]-(y_step*front_flux/thermal_conductivity))/(y_step*y_step);
                                        }
                                        if (j == mesh_x[0].size()-1)
                                        {
                                                back_flux = htc*(prev_temp[i][j][k]-ambient_temperature);
                                                dT2_dy2 = 2.0*(prev_temp[i][j-1][k]-prev_temp[i][j][k]-(y_step*back_flux/thermal_conductivity))/(y_step*y_step);
                                        }
                                }

                                // Calculate the second derivative of temperature wrt z
                                if (k != 0 && k != mesh_x[0][0].size()-1)
                                {
                                        dT2_dz2 = (prev_temp[i][j][k+1] - 2.0*prev_temp[i][j][k] + prev_temp[i][j][k-1]) / (z_step*z_step);
                                }
                                else
                                {
                                        if (k == 0)
                                        {
                                                top_flux = htc*(prev_temp[i][j][k]-ambient_temperature) - input_mesh[i][j];
                                                dT2_dz2 = 2.0*(prev_temp[i][j][k+1]-prev_temp[i][j][k]-(z_step*top_flux/thermal_conductivity))/(z_step*z_step);
                                        }
                                        if (k == mesh_x[0][0].size()-1)
                                        {
                                                bottom_flux = htc*(prev_temp[i][j][k]-ambient_temperature);
                                                dT2_dz2 = 2.0*(prev_temp[i][j][k-1]-prev_temp[i][j][k]-(z_step*bottom_flux/thermal_conductivity))/(z_step*z_step);
                                        }
                                }

                                // Update the temperature field
                                double temp_rate = thermal_diffusivity*(dT2_dx2+dT2_dy2+dT2_dz2)+(enthalpy_of_reaction*cure_rate)/specific_heat;
                                temp_mesh[i][j][k] = temp_mesh[i][j][k] + temp_rate * time_step;
                        }
}

/**
 * Gets the state fed to PPO agent based on temperature, front location, front velocity, and the input
 * @return The normalized state array
 */
vector<double> Finite_Element_Solver::get_state()
{
        vector<double> state(1, 0.0);
        return state;
}

/**
 * Solves for the reward fed to the PPO agent based on the reward function parameters, temperature, and front velocity
 * @return The calculated reward
 */
double Finite_Element_Solver::get_reward()
{
        double reward = 0.0;
        return reward;
}

/**
 * Steps the environments time and updates the target velocity
 * Boolean that determines whether simulation is complete or not
 */
bool Finite_Element_Solver::step_time()
{
        // Update the current time and check for simulation completion
        bool done = (current_index == target_front_vel.size() - 1);
        if (!done)
        {
                current_time = current_time + time_step;
                current_index = current_index + 1;
                current_target_front_vel = target_front_vel[current_index];
        }

        return done;
}
