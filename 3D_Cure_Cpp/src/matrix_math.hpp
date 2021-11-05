#pragma once
#include <vector>
#include <math.h>

using namespace std;

vector<double> mat_vec_mul( vector<vector<double>> input_matrix,  vector<double> input_vector );
vector<vector<double>> get_inv( vector<vector<double>> matrix );
double get_det( vector<vector<double>> matrix );
vector<vector<double>> get_minor_matrix( vector<vector<double>> matrix, unsigned int i, unsigned int j );
vector<vector<double>> get_adj( vector<vector<double>> matrix );
double get_cofactor( vector<vector<double>> matrix, unsigned int i, unsigned int j );