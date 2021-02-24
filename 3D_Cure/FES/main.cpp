#include "Finite_Element_Solver.h"
#include <unistd.h>
#include <chrono>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

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

int main(int argc)
{
								// Init python interpreter
								Py_Initialize();
								PyObject *ppoName, *ppoModule;
								ppoName = PyUnicode_DecodeFSDefault("PPO_Agent_3_Output");
								ppoModule = PyImport_Import(ppoName);
								PyRun_SimpleString("print('Importing PPO_Agent_3_Output...')\n");
								PyRun_SimpleString("import PPO_Agent_3_Output as ppo\n");
								PyRun_SimpleString("print('Building agent...')\n");
								PyRun_SimpleString("agent = ppo.PPO_Agent(160, 240, 10, 240, 10, 0.99, 0.95, 0.20, 1.0e-3, 0.0009765625)\n");

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
								Py_FinalizeEx();
								return 0;
}
