#define PY_SSIZE_T_CLEAN
#include "Finite_Element_Solver.h"
#include <chrono>
#include <Python.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <queue>

using namespace std;

/**
 * Prints to std out a representation of a 2D vector
 * @param The array to be printed
 */
void print_2D(vector<vector<double> > arr)
{
        // Find the largest order number
        int max_order = 0;
        int curr_order = 0;
        for (unsigned int i = 0; i < arr.size(); i++)
        {
                for (unsigned int j = 0; j < arr[0].size(); j++)
                {
                        curr_order = (int) floor(log10(abs(arr[i][j])));
                        max_order = curr_order > max_order ? curr_order : max_order;
                }
        }

        for (unsigned int i = 0; i < arr.size(); i++)
        {
                for (unsigned int j = 0; j < arr[0].size(); j++)
                {
                        if (arr[i][j] == 0.0)
                        {
                                for (int i = 0; i <= max_order; i++)
                                {
                                        cout << " ";
                                }
                        }
                        else
                        {
                                curr_order = (int) floor(log10(abs(arr[i][j])));
                                curr_order = curr_order < 0 ? 0 : curr_order;
                                for (int i = 0; i < max_order-curr_order; i++)
                                {
                                        cout << " ";
                                }
                        }
                        if (arr[i][j] > 0.0)
                        {
                                printf(" %.2f ", arr[i][j]);
                        }
                        else
                        {
                                printf("%.2f ", arr[i][j]);
                        }
                }
                cout << endl;
        }
}

/**
 * Initializes the python PPO agent
 * @param The length of the state array
 * @param The total number of agent steps taken per trajectory
 * @param The number of trajectories per optimization batch
 * @param The number of (state,reward) pairs per optimization step
 * @param The number of optimization steps taken per optimization batch
 * @param The discount ratio
 * @param GAE lambda
 * @param PPO clipping ratio
 * @param Initial learning rate
 * @param Learning rate exponential decay rate
 * @return PyObject pointer pointing at the initialized PPO agent
 */
PyObject* init_agent(long num_states, long steps_per_trajectory, long trajectories_per_batch, long minibatch_size, long num_epochs, double gamma, double lamb, double epsilon, double alpha, double decay_rate)
{
        // Declare PyObjects
        PyObject *ppo_name, *ppo_module, *ppo_dict, *ppo_init, *ppo_init_args, *ppo_object;

        // Define module name
        ppo_name = PyUnicode_DecodeFSDefault("PPO_Agent_3_Output");

        // Initialize module
        ppo_module = PyImport_Import(ppo_name);
        if (ppo_module == NULL)
        {
                PyErr_Print();
                fprintf(stderr, "Failed to find module\n");
                return NULL;
        }
        Py_DECREF(ppo_name);

        // Load dictionary of module methods and variables
        ppo_dict = PyModule_GetDict(ppo_module);
        if (ppo_dict == NULL)
        {
                PyErr_Print();
                fprintf(stderr, "Failed to load module dict\n");
                return NULL;
        }
        Py_DECREF(ppo_module);

        // Get the initialization function from the module dictionary
        ppo_init = PyDict_GetItemString(ppo_dict, "PPO_Agent");
        if (ppo_init == NULL || !PyCallable_Check(ppo_init))
        {
                PyErr_Print();
                fprintf(stderr, "Failed to find __init__\n");
                return NULL;
        }
        Py_DECREF(ppo_dict);

        // Build the initialization arguments
        ppo_init_args = PyTuple_New(10);
        PyTuple_SetItem(ppo_init_args, 0, PyLong_FromLong(num_states));
        PyTuple_SetItem(ppo_init_args, 1, PyLong_FromLong(steps_per_trajectory));
        PyTuple_SetItem(ppo_init_args, 2, PyLong_FromLong(trajectories_per_batch));
        PyTuple_SetItem(ppo_init_args, 3, PyLong_FromLong(minibatch_size));
        PyTuple_SetItem(ppo_init_args, 4, PyLong_FromLong(num_epochs));
        PyTuple_SetItem(ppo_init_args, 5, PyFloat_FromDouble(gamma));
        PyTuple_SetItem(ppo_init_args, 6, PyFloat_FromDouble(lamb));
        PyTuple_SetItem(ppo_init_args, 7, PyFloat_FromDouble(epsilon));
        PyTuple_SetItem(ppo_init_args, 8, PyFloat_FromDouble(alpha));
        PyTuple_SetItem(ppo_init_args, 9, PyFloat_FromDouble(decay_rate));

        // Initialize ppo object
        ppo_object = PyObject_CallObject(ppo_init, ppo_init_args);
        if (ppo_object == NULL)
        {
                PyErr_Print();
                fprintf(stderr, "Initialization failed\n");
                Py_DECREF(ppo_init);
                Py_DECREF(ppo_init_args);
                return NULL;
        }
        Py_DECREF(ppo_init);
        Py_DECREF(ppo_init_args);

        // return the class
        return ppo_object;
}

/**
 * Converts 1D vector<double> to a PyList
 * @param The vector used to create list
 * @return PyObject pointer pointing at the created list
 */
PyObject* getList(vector<double> state)
{
        PyObject *list = PyList_New(state.size());
        for (unsigned int i = 0; i < state.size(); i++)
        {
                PyList_SetItem(list, i, PyFloat_FromDouble(state[i]));
        }
        return list;
}

/**
 * Converts PyList to a 1D vector<double>
 * @param The list used to create the vector
 * @return 1D vector<double> copy of the PyList
 */
vector<double> getVector(PyObject* list)
{
        if (PyList_Size(list) > 0)
        {
                vector<double> out = vector<double>(PyList_Size(list), 0.0);
                for (unsigned int i = 0; i < PyList_Size(list); i++)
                {
                        out[i] = PyFloat_AsDouble(PyList_GetItem(list, i));
                }
                return out;
        }
        return vector<double>();
}

/**
 * Runs a set of trajectories using the PPO policy, updates the PPO agent, and collects relevant training data
 * @param The finite element solver object used to propogate time
 * @param The ppo agent that defines the policy
 * @param The total number of trajectories to be executed
 * @param The number of simulation steps taken per single agent cycle
 * @param The number of simulation steps taken per single save frame
 * @param The number of agent cycles steps in each trajecotry
 * @return Tuple containing the training data, the updated finite element solver, and the trained PPO agent
 */
auto run(Finite_Element_Solver FES, PyObject* agent, int total_trajectories, int steps_per_agent_cycle, int steps_per_frame, int steps_per_trajectory)
{

        // Determine how many frames there are in each trajectory
        int frames_per_trajectory = 0;
        int frame_index = 0;
        bool frame_done = false;
        while(!frame_done)
        {
                frame_done = (frame_index == (int)FES.get_target_front_vel_arr_size() - 1);
                if ((frame_index % steps_per_frame == 0) || (frame_index==0))
                {
                        frames_per_trajectory++;
                }
                frame_index++;
        }

        // Training data storage
        vector<double> r_per_episode;
        vector<double> x_rate_stdev;
        vector<double> y_rate_stdev;
        vector<double> mag_stdev;
        vector<double> value_error;

        // Best trajecotry data
        vector<double> best_input_location_x;
        vector<double> best_input_location_y;
        vector<double> best_sim_time;
        vector<double> best_target_velocity;
        vector<vector<vector<double> > > best_temperature_field = vector<vector<vector<double> > >(FES.get_num_vert_length(), vector<vector<double> >(FES.get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
        vector<vector<vector<double> > > best_cure_field = vector<vector<vector<double> > >(FES.get_num_vert_length(), vector<vector<double> >(FES.get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
        vector<vector<vector<double> > > best_front_location = vector<vector<vector<double> > >(FES.get_num_vert_width(), vector<vector<double> >(FES.get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));
        vector<vector<vector<double> > > best_front_velocity = vector<vector<vector<double> > >(FES.get_num_vert_width(), vector<vector<double> >(FES.get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));

        // Current trajectory data
        vector<double> curr_input_location_x;
        vector<double> curr_input_location_y;
        vector<double> curr_sim_time;
        vector<double> curr_target_velocity;
        vector<vector<vector<double> > > curr_temperature_field = vector<vector<vector<double> > >(FES.get_num_vert_length(), vector<vector<double> >(FES.get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
        vector<vector<vector<double> > > curr_cure_field = vector<vector<vector<double> > >(FES.get_num_vert_length(), vector<vector<double> >(FES.get_num_vert_width(), vector<double>(frames_per_trajectory, 0.0)));
        vector<vector<vector<double> > > curr_front_location = vector<vector<vector<double> > >(FES.get_num_vert_width(), vector<vector<double> >(FES.get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));
        vector<vector<vector<double> > > curr_front_velocity = vector<vector<vector<double> > >(FES.get_num_vert_width(), vector<vector<double> >(FES.get_num_vert_depth(), vector<double>(frames_per_trajectory, 0.0)));

        // Simulation set variables
        double total_reward = 0.0;
        int curr_step = 0;
        double best_episode = 0.0;
        double percent_complete = 0.0;
        double prev_episode_reward = 0.0;

        // Run a set of episodes
        for (int i = 0; i < total_trajectories; i++)
        {
                // User readout parameters
                percent_complete = total_trajectories>1 ? 100.0 * i / (total_trajectories-1) : 0.0;
                double episode_reward = total_reward;

                // User readout
                stringstream stream;
                stream << std::fixed << std::setprecision(1) << percent_complete;
                string msg1 = stream.str();
                msg1.append("% Complete");
                msg1.append(18 - msg1.length(), ' ');
                string msg2 = "";
                msg2.append("| Traj: " + to_string(i+1) + "/" + to_string(total_trajectories));
                msg2.append(22 - msg2.length(), ' ');
                string msg3 = "| R/Step: ";
                stream.str(std::string());
                stream << std::fixed << std::setprecision(2) << prev_episode_reward/(double)steps_per_trajectory;
                msg3.append(stream.str());
                msg3.append(18 - msg3.length(), ' ');
                string msg4 = "| Avg_R/Step: ";
                string msg6 = "| Avg R: ";
                if (r_per_episode.empty())
                {
                        stream.str(std::string());
                        stream << std::fixed << std::setprecision(2) << 0.0;
                        msg4.append(stream.str());
                        stream.str(std::string());
                        stream << std::fixed << std::setprecision(1) << 0.0;
                        msg6.append(stream.str());
                }
                else
                {
                        int start_index = (int) r_per_episode.size() - 100;
                        start_index = start_index < 0 ? 0 : start_index;
                        double avg_r_per_episode = 0.0;
                        for (int i = start_index; i < (int)r_per_episode.size(); i++)
                        {
                                avg_r_per_episode += r_per_episode[i];
                        }
                        avg_r_per_episode = avg_r_per_episode / (r_per_episode.size() - start_index);
                        stream.str(std::string());
                        stream << std::fixed << std::setprecision(2) << avg_r_per_episode;
                        msg4.append(stream.str());
                        stream.str(std::string());
                        stream << std::fixed << std::setprecision(1) << avg_r_per_episode*steps_per_trajectory;
                        msg6.append(stream.str());
                }
                msg4.append(22 - msg4.length(), ' ');
                msg6.append(18 - msg6.length(), ' ');
                string msg5 = "| Best R: ";
                stream.str(std::string());
                stream << std::fixed << std::setprecision(1) << best_episode;
                msg5.append(stream.str());
                msg5.append(19 - msg5.length(), ' ');
                cout << msg1+msg2+msg3+msg4+msg5+msg6 << "|\r";

                // Reset environment
                vector<double> state = FES.reset();

                // Declare simulation variables
                bool done = false;
                double action_1=0.0, stdev_1, action_2=0.0, stdev_2, action_3=0.0, stdev_3, reward;
                bool run_agent, save_frame;
                int step_in_trajectory = 0;
                PyObject *py_state, *py_result;

                // Simulation for loop
                while (!done)
                {
                        // Determine what to run this simulation step
                        run_agent = (step_in_trajectory % steps_per_agent_cycle == 0) || (step_in_trajectory==0);
                        save_frame = (step_in_trajectory % steps_per_frame == 0) || (step_in_trajectory==0);

                        // Run the agent
                        if (run_agent)
                        {

                                // Get the starting state
                                state = FES.get_state();

                                // Get the agent commanded action
                                py_state = getList(state);
                                py_result = PyObject_CallMethod(agent, "get_action", "O", py_state);
                                action_1 = PyFloat_AsDouble(PyTuple_GetItem(py_result, 0));
                                stdev_1 = PyFloat_AsDouble(PyTuple_GetItem(py_result, 1));
                                action_2 = PyFloat_AsDouble(PyTuple_GetItem(py_result, 2));
                                stdev_2 = PyFloat_AsDouble(PyTuple_GetItem(py_result, 3));
                                action_3 = PyFloat_AsDouble(PyTuple_GetItem(py_result, 4));
                                stdev_3 = PyFloat_AsDouble(PyTuple_GetItem(py_result, 5));

                                // Step the environment
                                done = FES.step(action_1, action_2, action_3);

                                // Update the agent
                                reward = FES.get_reward();
                                py_result = PyObject_CallMethod(agent, "update_agent", "(O,f,f,f,f)", py_state, PyFloat_FromDouble(action_1), PyFloat_FromDouble(action_2), PyFloat_FromDouble(action_3), PyFloat_FromDouble(reward));

                                // Update agent simulation PARAMETERS
                                total_reward = total_reward + reward;
                                curr_step += 1;
                        }
                        else
                        {
                                // Step the environment
                                done = FES.step(action_1, action_2, action_3);
                        }

                        // Update the logs
                        if (save_frame)
                        {
                                vector<double> input_location = FES.get_input_location();
                                vector<vector<double> > temp_mesh = FES.get_temp_mesh();
                                vector<vector<double> > cure_mesh = FES.get_cure_mesh();
                                vector<vector<double> > front_loc = FES.get_front_loc();
                                vector<vector<double> > front_vel = FES.get_front_vel();
                                curr_input_location_x.push_back(input_location[0]);
                                curr_input_location_y.push_back(input_location[1]);
                                curr_sim_time.push_back(FES.get_current_time());
                                curr_target_velocity.push_back(FES.get_current_target_front_vel());
                                for (int i = 0; i < FES.get_num_vert_length(); i++)
                                {
                                        for (int j = 0; j < FES.get_num_vert_width(); j++)
                                        {
                                                curr_temperature_field[i][j][curr_target_velocity.size()-1] = temp_mesh[i][j];
                                                curr_cure_field[i][j][curr_target_velocity.size()-1] = cure_mesh[i][j];
                                        }
                                }
                                for (int j = 0; j < FES.get_num_vert_width(); j++)
                                {
                                        for (int k = 0; k < FES.get_num_vert_depth(); k++)
                                        {
                                                curr_front_location[j][k][curr_target_velocity.size()-1] = front_loc[j][k];
                                                curr_front_velocity[j][k][curr_target_velocity.size()-1] = front_vel[j][k];
                                        }
                                }
                        }

                        // Update the current state and the step in episode
                        step_in_trajectory += 1;

                }

                // Update the episode reward
                episode_reward = total_reward - episode_reward;
                prev_episode_reward = episode_reward;

                // Update the best trajectory memory
                if (episode_reward > best_episode || i == 0)
                {
                        best_episode = episode_reward;
                        best_input_location_x = curr_input_location_x;
                        best_input_location_y = curr_input_location_y;
                        best_sim_time = curr_sim_time;
                        best_target_velocity = curr_target_velocity;
                        best_temperature_field = curr_temperature_field;
                        best_cure_field = curr_cure_field;
                        best_front_location = curr_front_location;
                        best_front_velocity = curr_front_velocity;
                }

                // Reset the current trajectory memory
                curr_input_location_x.clear();
                curr_input_location_y.clear();
                curr_sim_time.clear();
                curr_target_velocity.clear();

                // Update the logs
                r_per_episode.push_back(episode_reward/(double)steps_per_trajectory);
                x_rate_stdev.push_back(stdev_1);
                y_rate_stdev.push_back(stdev_2);
                mag_stdev.push_back(stdev_3);

                // Release the python memory
                Py_DECREF(py_state);
                Py_DECREF(py_result);

                // Final user readout
                if (i == total_trajectories - 1)
                {
                        msg1 = "";
                        msg1.append("100.0% Complete   ");
                        msg2 = "| Traj: ";
                        msg2.append(to_string(i+1) + "/" + to_string(total_trajectories));
                        msg2.append(22 - msg2.length(), ' ');
                        msg3 = "| R/Step: ";
                        stream.str(std::string());
                        stream << std::fixed << std::setprecision(2) << episode_reward/(double)steps_per_trajectory;
                        msg3.append(stream.str());
                        msg3.append(18 - msg3.length(), ' ');
                        msg4 = "| Avg_R/Step: ";
                        msg6 = "| Avg R: ";
                        unsigned int start_index = r_per_episode.size() - 100;
                        start_index = start_index < 0 ? 0 : start_index;
                        double avg_r_per_episode = 0.0;
                        for (unsigned int i = start_index; i < r_per_episode.size(); i++)
                        {
                                avg_r_per_episode += r_per_episode[i];
                        }
                        avg_r_per_episode = avg_r_per_episode / (r_per_episode.size() - start_index);
                        stream.str(std::string());
                        stream << std::fixed << std::setprecision(2) << avg_r_per_episode;
                        msg4.append(stream.str());
                        stream.str(std::string());
                        stream << std::fixed << std::setprecision(1) << avg_r_per_episode*steps_per_trajectory;
                        msg6.append(stream.str());
                        msg4.append(22 - msg4.length(), ' ');
                        msg6.append(18 - msg6.length(), ' ');
                        msg5 = "| Best R: ";
                        stream.str(std::string());
                        stream << std::fixed << std::setprecision(1) << best_episode;
                        msg5.append(stream.str());
                        msg5.append(19 - msg5.length(), ' ');
                        cout << msg1+msg2+msg3+msg4+msg5+msg6 << "|\n";
                }
        }

        // Gather the trajectory value error from the agent
        PyObject* py_value_error = PyObject_GetAttr(agent, PyUnicode_DecodeFSDefault("value_estimation_error"));
        value_error = getVector(py_value_error);
        Py_DECREF(py_value_error);

        // Return agent
        auto out = make_tuple(agent, r_per_episode, x_rate_stdev, y_rate_stdev, mag_stdev, value_error, best_input_location_x,
                              best_input_location_y, best_sim_time, best_target_velocity, best_temperature_field, best_cure_field,
                              best_front_location, best_front_velocity, best_episode);
        return out;
}

int main()
{
        // Agent parameters
        int total_trajectories = 1;
        int steps_per_trajectory = 240;
        int trajectories_per_batch = 10;
        int num_epochs = 10;
        double gamma = 0.99;
        double lamb = 0.95;
        double epsilon = 0.20;
        double start_alpha = 1.0e-3;
        double end_alpha = 5.0e-4;

        // Rendering parameters
        double frame_rate = 10.0;

        // Initialize FES
        Finite_Element_Solver FES = Finite_Element_Solver();

        // Calculated parameters
        double decay_rate = pow(end_alpha/start_alpha, trajectories_per_batch/total_trajectories);
        double execution_period = (FES.get_sim_duration() / (double)steps_per_trajectory);
        int steps_per_agent_cycle = (int) round(execution_period / FES.get_time_step());
        int steps_per_frame = (int) round(1.0 / (FES.get_time_step() * frame_rate));
        steps_per_frame = steps_per_frame <= 0 ? 1 : steps_per_frame;
        int minibatch_size = (int) round(((double)trajectories_per_batch * (double)steps_per_trajectory) / (double)num_epochs);

        // Check inputs
        if ( floor(execution_period / FES.get_time_step()) != (execution_period / FES.get_time_step()) )
        {
                fprintf(stderr, "RuntimeError: Agent execution rate is not multiple of simulation rate\n");
                return 1;
        }

        // Init agent
        Py_Initialize();
        PyObject* agent = init_agent(FES.get_num_state(), steps_per_trajectory, trajectories_per_batch, minibatch_size, num_epochs, gamma, lamb, epsilon, start_alpha, decay_rate);

        // Train agent
        cout << "Training agent..." << endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto [trained_agent, r_per_episode, x_rate_stdev, y_rate_stdev, mag_stdev, value_error, best_input_location_x,
              best_input_location_y, best_sim_time, best_target_velocity, best_temperature_field, best_cure_field,
              best_front_location, best_front_velocity, best_episode] = run(FES, agent, total_trajectories, steps_per_agent_cycle, steps_per_frame, steps_per_trajectory);

        // Stop clock and print duration
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time ).count();
        cout << "Training Took: ";
        printf("%.1f", (double)duration*10e-7);
        cout << " seconds.\n";

        // Save results
        cout << "Saving results..." << endl;
        PyObject* py_r_per_episode = getList(r_per_episode);
        PyObject* py_x_rate_stdev = getList(x_rate_stdev);
        PyObject* py_y_rate_stdev = getList(y_rate_stdev);
        PyObject* py_mag_stdev = getList(mag_stdev);
        PyObject* py_value_error = getList(value_error);
        PyObject* py_best_input_location_x = getList(best_input_location_x);
        PyObject* py_best_input_location_y = getList(best_input_location_y);
        PyObject* py_best_sim_time = getList(best_sim_time);
        PyObject* py_best_target_velocity = getList(best_target_velocity);
        best_temperature_field;
        best_cure_field;
        best_front_location;
        best_front_velocity;
        PyObject* py_best_episode = PyFloat_FromDouble(best_episode);
        Py_DECREF(py_r_per_episode);
        Py_DECREF(py_x_rate_stdev);
        Py_DECREF(py_y_rate_stdev);
        Py_DECREF(py_mag_stdev);
        Py_DECREF(py_value_error);
        Py_DECREF(py_best_input_location_x);
        Py_DECREF(py_best_input_location_y);
        Py_DECREF(py_best_sim_time);
        Py_DECREF(py_best_target_velocity);
        Py_DECREF(py_best_episode);

        // End main
        Py_DECREF(agent);
        Py_DECREF(trained_agent);
        Py_FinalizeEx();
        cout << "Done! Press enter to end...";
        cin.get();
        return 0;
}
