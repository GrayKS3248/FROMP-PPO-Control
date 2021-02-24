#define PY_SSIZE_T_CLEAN
#include "Finite_Element_Solver.h"
#include <chrono>
#include <Python.h>
#include <iostream>

using namespace std;

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

PyObject* init_agent(long num_states, long steps_per_trajectory, long trajectories_per_batch, long minibatch_size, long num_epochs, double gamma, double lamb, double epsilon, double alpha, double decay_rate)
{
        // Declare PyObjects
        PyObject *ppo_name, *ppo_module, *ppo_init, *ppo_init_args, *ppo_object;

        // Load PPO agent module
        ppo_name = PyUnicode_DecodeFSDefault("PPO_Agent_3_Output");
        ppo_module = PyImport_Import(ppo_name);
        Py_DECREF(ppo_name);

        // Initialize module
        if (ppo_module != NULL)
        {
                ppo_init = PyObject_GetAttrString(ppo_module, "__init__");
                if (ppo_init && PyCallable_Check(ppo_init))
                {
                        // Set and pass init args
                        cout << "Initializing PPO Agent...\n";
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
                        Py_DECREF(ppo_init_args);
                }
                // If initialization fails
                else
                {
                        if (PyErr_Occurred())
                                PyErr_Print();
                        fprintf(stderr, "Cannot find __init__\n");
                }
                Py_XDECREF(ppo_init);
                Py_DECREF(ppo_module);
        }
        // If module fails to load:
        else
        {
                PyErr_Print();
                fprintf(stderr, "Failed to load PPO_Agent_3_Output\n");
        }

        return ppo_object;
}

int main()
{
        // Init python interpreter
        Py_Initialize();
        PyObject* agent = init_agent(160, 240, 10, 240, 10, 0.99, 0.95, 0.20, 1.0e-3, 0.0009765625);
        //PyObject_CallMethod(ppo_object, "get_action",const char *arg_format, ...);
        Py_FinalizeEx();

        // Start clock
        auto t1 = std::chrono::high_resolution_clock::now();

        // Initialize FES
        Finite_Element_Solver FES = Finite_Element_Solver();
        cout << "Finite Element Solver" << endl;

        // Reset environment
        vector<double> s = FES.reset();
        bool done = false;
        tuple<vector<double>, double, bool> state_reward_done;

        // Simulation for loop
        while (!done)
        {
                state_reward_done = FES.step(0.0, 0.0, 0.0);
                done = get<2>(state_reward_done);
        }

        // Stop clock and print duration
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        cout << "Execution took: ";
        printf("%.3f", (double)duration*10e-7);
        cout << " seconds." << endl;

        // End main
        cout << endl << "Press enter to end..." << endl;
        cin.get();
        return 0;
}
