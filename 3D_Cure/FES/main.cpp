#include "Finite_Element_Solver.h"

using namespace std;

void print_2D(vector<vector<double>> arr)
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
	
	cout << "Press enter to end..." << endl;
	cin.get();
	return 0;
}
