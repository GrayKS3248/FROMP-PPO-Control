#include "Finite_Element_Solver.h"
#include <unistd.h>
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

int main()
{
								Finite_Element_Solver FES = Finite_Element_Solver();
								cout << "Finite Element solver" << endl;

								vector<double> s = FES.reset();
								bool done = false;
								tuple<vector<double>, double, bool> state_reward_done;
								while (!done)
								{
																state_reward_done = FES.step(0.0, 0.0, 0.0);
																done = get<2>(state_reward_done);
																/*
																   if ((int)round(FES.get_current_time()/FES.get_time_step()) % 20 == 0)
																   {
																        system("cls");
																        cout << "Current Time: ";
																        printf("%.1f", FES.get_current_time());
																        cout << endl;
																        print_2D(FES.get_cure_mesh());
																        usleep(200000);
																   }
																 */
								}

								cout << endl << "Press enter to end..." << endl;
								cin.get();
								return 0;
}
