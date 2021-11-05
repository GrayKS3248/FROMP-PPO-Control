#include "matrix_math.hpp"

/** Multiplies a vector by a square matrix
* @param The square matrix
* @param The vector
* @return The matrix product of the two parameters
**/
vector<double> mat_vec_mul( vector<vector<double>> input_matrix,  vector<double> input_vector )
{
	vector<double> product = vector<double>(input_matrix.size(), 0.0);
	for(unsigned int i = 0; i < input_matrix.size(); i++)
	{
		double sum = 0.0;
		for(unsigned int j = 0; j < input_matrix[0].size(); j++)
		{
			sum += input_matrix[i][j] * input_vector[j];
		}
		product[i] = sum;
	}
	
	return product;
}


/** Gets the inverse of a square matrix
* @param The matrix to be inverted
* @return The inverse of the parameter matrix
**/
vector<vector<double>> get_inv( vector<vector<double>> matrix )
{
	vector<vector<double>> inv_matrix = vector<vector<double>>( matrix.size(), vector<double>( matrix[0].size(), 0.0 ) );
	vector<vector<double>> adj_matrix = get_adj( matrix );
	double inv_det = (1.0 / get_det( matrix ));
	for(unsigned int i = 0; i < matrix.size(); i++)
	for(unsigned int j = 0; j < matrix[0].size(); j++)
	{
		inv_matrix[i][j] = inv_det * adj_matrix[i][j];
	}
	return inv_matrix;
}

/** Gets the determinant of a square matrix
* @param The matrix whose determinant is to be calcualted
* @return The determinant of the parameter matrix
**/
double get_det( vector<vector<double>> matrix )
{
	double det = 0.0;
	if ( matrix.size() == 2 && matrix[0].size() == 2 )
	{
		det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
	}
	else
	{
		unsigned int i = 0;
		for( unsigned int j = 0; j < matrix.size(); j++ )
		{
			det+= pow( -1.0, ((double)i+1.0)+((double)j+1.0) ) * matrix[i][j] * get_det( get_minor_matrix( matrix, i, j ) );
		}
	}
	
	return det;
}

/** Gets the square minor of a square matrix at specified masking coordinates
* @param The matrix whose minor matrix is to be calculated
* @param The index of the masked row
* @param The index of the masked column
* @return The minor matrix of the parameter matrix
**/
vector<vector<double>> get_minor_matrix( vector<vector<double>> matrix, unsigned int i, unsigned int j )
{
	vector<vector<double>> minor = vector<vector<double>>( matrix.size()-1, vector<double>( matrix[0].size()-1, 0.0 ) );
	for(unsigned int p = 0; p < matrix.size()-1; p++)
	for(unsigned int q = 0; q < matrix[0].size()-1; q++)
	{
		unsigned int p_access = p;
		if(p>=i)
		{
			p_access = p + 1;
		}
		unsigned int q_access = q;
		if(q>=j)
		{
			q_access = q + 1;
		}
		
		minor[p][q] = matrix[p_access][q_access];
	}
	
	return minor;
}

/** Gets the adjugate matrix of a square matrix
* @param The matrix whose adjugate matrix is to be calculated
* @return The adjugate matrix of the parameter matrix
**/
vector<vector<double>> get_adj( vector<vector<double>> matrix )
{
	vector<vector<double>> adj_matrix = vector<vector<double>>( matrix.size(), vector<double>( matrix[0].size(), 0.0 ) );
	for(unsigned int i = 0; i < matrix.size(); i++)
	for(unsigned int j = 0; j < matrix[0].size(); j++)
	{
		adj_matrix[i][j] = get_cofactor( matrix, j, i );
	}
	return adj_matrix;
}


/** Gets the cofactor of a square matrix at specified coordinates
* @param The matrix whose cofactor is to be calculated
* @param The index of the cofactor row
* @param The index of the cofactor column
* @return The cofactor of the parameter matrix
**/
double get_cofactor( vector<vector<double>> matrix, unsigned int i, unsigned int j )
{
	double minor = get_det( get_minor_matrix( matrix, i, j ) );
	double cofactor = pow( -1.0, ((double)i+1.0)+((double)j+1.0) ) * minor;
	return cofactor;
}