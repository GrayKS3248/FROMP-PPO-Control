/** Multiplies a 3x3 matrix with a 3x1 vector
* @param 3x3 matrix
* @param 3x1 vector
* @return Multiplication of A and B
*/
vector<double> Finite_Element_Solver::mat_mul_3(vector<vector<double>> &A, vector<double> &B)
{
	vector<double> mul = vector<double>(3,0.0);
	
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			mul[i] += A[i][j] * B[j];
		}
	}
	
	return mul;
}

/** Calculates inverse of 3x3 matrix
* @param 3x3 matrix to invert
* @return Inverse of input
*/
vector<vector<double>> Finite_Element_Solver::invert_3x3(vector<vector<double>> &A)
{
	vector<vector<double>> A_minor = vector<vector<double>>(3, vector<double>(3, 0.0));
	vector<vector<double>> A_cofactor = vector<vector<double>>(3, vector<double>(3, 0.0));
	vector<vector<double>> A_adjugate = vector<vector<double>>(3, vector<double>(3, 0.0));
	vector<vector<double>> A_inv = vector<vector<double>>(3, vector<double>(3, 0.0));
	double det_A = 0.0;

	// Populate the A matrix of minors
	A_minor[0][0] = (A[1][1] * A[2][2]) - (A[1][2] * A[2][1]);
	A_minor[1][0] = (A[0][1] * A[2][2]) - (A[0][2] * A[2][1]);
	A_minor[2][0] = (A[0][1] * A[1][2]) - (A[0][2] * A[1][1]);
	A_minor[0][1] = (A[1][0] * A[2][2]) - (A[1][2] * A[2][0]);
	A_minor[1][1] = (A[0][0] * A[2][2]) - (A[0][2] * A[2][0]);
	A_minor[2][1] = (A[0][0] * A[1][2]) - (A[0][2] * A[1][0]);
	A_minor[0][2] = (A[1][0] * A[2][1]) - (A[1][1] * A[2][0]);
	A_minor[1][2] = (A[0][0] * A[2][1]) - (A[0][1] * A[2][0]);
	A_minor[2][2] = (A[0][0] * A[1][1]) - (A[0][1] * A[1][0]);

	// Populate the A matrix of cofactors
	A_cofactor[0][0] = A_minor[0][0];
	A_cofactor[1][0] = -A_minor[1][0];
	A_cofactor[2][0] = A_minor[2][0];
	A_cofactor[0][1] = -A_minor[0][1];
	A_cofactor[1][1] = A_minor[1][1];
	A_cofactor[2][1] = -A_minor[2][1];
	A_cofactor[0][2] = A_minor[0][2];
	A_cofactor[1][2] = -A_minor[1][2];
	A_cofactor[2][2] = A_minor[2][2];

	// Populate the A adjugate matrix
	A_adjugate[0][0] = A_cofactor[0][0];
	A_adjugate[1][0] = A_cofactor[0][1];
	A_adjugate[2][0] = A_cofactor[0][2];
	A_adjugate[0][1] = A_cofactor[1][0];
	A_adjugate[1][1] = A_cofactor[1][1];
	A_adjugate[2][1] = A_cofactor[1][2];
	A_adjugate[0][2] = A_cofactor[2][0];
	A_adjugate[1][2] = A_cofactor[2][1];
	A_adjugate[2][2] = A_cofactor[2][2];

	// Calculate the determinate of A
	det_A = A[0][0] * A_minor[0][0] - A[0][1] * A_minor[0][1] + A[0][2] * A_minor[0][2];

	// Calculate the inverse of A
	A_inv[0][0] = A_adjugate[0][0] / det_A;
	A_inv[1][0] = A_adjugate[1][0] / det_A;
	A_inv[2][0] = A_adjugate[2][0] / det_A;
	A_inv[0][1] = A_adjugate[0][1] / det_A;
	A_inv[1][1] = A_adjugate[1][1] / det_A;
	A_inv[2][1] = A_adjugate[2][1] / det_A;
	A_inv[0][2] = A_adjugate[0][2] / det_A;
	A_inv[1][2] = A_adjugate[1][2] / det_A;
	A_inv[2][2] = A_adjugate[2][2] / det_A;
	
	return A_inv;
}

/** Calculates the Laplacian of the temperature field at a specified location
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @return 7 point least squares 2nd order fitting Lapclacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_72(int i, int j, int k, const vector<vector<vector<double>>> &temperature)
{
	//Flux and Laplace variables
	double dT2_dx2;
	double dT2_dy2;
	double dT2_dz2;
	double left_flux;
	double right_flux;
	double front_flux;
	double back_flux;
	double top_flux;
	double bottom_flux;
	double t_ijk = temperature[i][j][k];
	
	// Fitting variables
	vector<vector<double>> A = vector<vector<double>>(3, vector<double>(3, 0.0));
	vector<vector<double>> A_inv;
	vector<double> B = vector<double>(3, 0.0);
	vector<double> fit_soln;
	double xi = 0.0;
	double yi = 0.0;
	double sum_x_4 = 0.0;
	double sum_x_3 = 0.0;
	double sum_x_2 = 0.0;
	double sum_x = 0.0;
	double sum_x_2_y = 0.0;
	double sum_x_y = 0.0;
	double sum_y = 0.0;

	// Calculate the second derivative of temperature wrt x in interior of mesh
	if (i != 0 && i != num_vert_length-1)
	{
		// (-1, 0, 1, 2, 3, 4, 5) for 2nd order derivative
		if (i==1)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_x[i-1+index][j][k];
				yi = temperature[i-1+index][j][k];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-2, -1, 0, 1, 2, 3, 4) for 2nd order derivative
		else if(i==2)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_x[i-2+index][j][k];
				yi = temperature[i-2+index][j][k];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-4, -3, -2, -1, 0, 1, 2) for 2nd order derivative
		else if(i==num_vert_length-3)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_x[i-4+index][j][k];
				yi = temperature[i-4+index][j][k];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-5, -4, -3, -2, -1, 0, 1) for 2nd order derivative
		else if(i==num_vert_length-2)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_x[i-5+index][j][k];
				yi = temperature[i-5+index][j][k];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-3, -2, -1, 0, 1, 2, 3) for 2nd order derivative
		else
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_x[i-3+index][j][k];
				yi = temperature[i-3+index][j][k];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		
		// Populate B matrix
		B[0] = sum_x_2_y;
		B[1] = sum_x_y;
		B[2] = sum_y;
		
		// Populate A matrix
		A[0][0] = sum_x_4;
		A[1][0] = sum_x_3;
		A[2][0] = sum_x_2;
		A[0][1] = sum_x_3;
		A[1][1] = sum_x_2;
		A[2][1] = sum_x;
		A[0][2] = sum_x_2;
		A[1][2] = sum_x;
		A[2][2] = 7.0;
		
		A_inv = invert_3x3(A);
		fit_soln = mat_mul_3(A_inv, B);
		dT2_dx2 = 2.0 * fit_soln[0];
	}

	// Calculate the second derivative of temperature wrt x at boundaries
	else
	{
		// LHS boundary condition
		if (i == 0)
		{
			// Trigger boundary condition
			if (current_time >= trigger_time && current_time < trigger_time + trigger_duration) { left_flux = htc*(t_ijk-ambient_temperature) - trigger_flux; }
			
			// Non-trigger boundary condition
			else { left_flux = htc*(t_ijk-ambient_temperature); }
			
			dT2_dx2 = 2.0*( temperature[i+1][j][k]-t_ijk-(x_step*left_flux/thermal_conductivity) ) / (x_step*x_step);
		}
		
		// RHS boundary condition
		else if (i == num_vert_length-1)
		{
			right_flux = htc*(t_ijk-ambient_temperature);
			dT2_dx2 = 2.0*( temperature[i-1][j][k] - t_ijk - (x_step*right_flux/thermal_conductivity) ) / (x_step*x_step);
		}
	}

	// Calculate the second derivative of temperature wrt y in interior of mesh
	if (j != 0 && j != num_vert_width-1)
	{
		// (-1, 0, 1, 2, 3, 4, 5) for 2nd order derivative
		if (j==1)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_y[i][j-1+index][k];
				yi = temperature[i][j-1+index][k];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-2, -1, 0, 1, 2, 3, 4) for 2nd order derivative
		else if(j==2)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_y[i][j-2+index][k];
				yi = temperature[i][j-2+index][k];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-4, -3, -2, -1, 0, 1, 2) for 2nd order derivative
		else if(j==num_vert_width-3)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_y[i][j-4+index][k];
				yi = temperature[i][j-4+index][k];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-5, -4, -3, -2, -1, 0, 1) for 2nd order derivative
		else if(j==num_vert_width-2)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_y[i][j-5+index][k];
				yi = temperature[i][j-5+index][k];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-3, -2, -1, 0, 1, 2, 3) for 2nd order derivative
		else
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_y[i][j-3+index][k];
				yi = temperature[i][j-3+index][k];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		
		// Populate B matrix
		B[0] = sum_x_2_y;
		B[1] = sum_x_y;
		B[2] = sum_y;
		
		// Populate A matrix
		A[0][0] = sum_x_4;
		A[1][0] = sum_x_3;
		A[2][0] = sum_x_2;
		A[0][1] = sum_x_3;
		A[1][1] = sum_x_2;
		A[2][1] = sum_x;
		A[0][2] = sum_x_2;
		A[1][2] = sum_x;
		A[2][2] = 7.0;
		
		A_inv = invert_3x3(A);
		fit_soln = mat_mul_3(A_inv, B);
		dT2_dy2 = 2.0 * fit_soln[0];
	}

	// Calculate the second derivative of temperature wrt y at boundaries
	else
	{
		// Front boundary condition
		if (j == 0)
		{
			front_flux = htc*(t_ijk-ambient_temperature);
			dT2_dy2 = 2.0*(temperature[i][j+1][k]-t_ijk-(y_step*front_flux/thermal_conductivity))/(y_step*y_step);
		}
		
		// Back boundary condition
		else if (j == num_vert_width-1)
		{
			back_flux = htc*(t_ijk-ambient_temperature);
			dT2_dy2 = 2.0*(temperature[i][j-1][k]-t_ijk-(y_step*back_flux/thermal_conductivity))/(y_step*y_step);
		}
	}

	// Calculate the second derivative of temperature wrt z in interior of mesh
	if (k != 0 && k != num_vert_depth-1)
	{
		// (-1, 0, 1, 2, 3, 4, 5) for 2nd order derivative
		if (k==1)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_z[i][j][k-1+index];
				yi = temperature[i][j][k-1+index];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-2, -1, 0, 1, 2, 3, 4) for 2nd order derivative
		else if(k==2)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_z[i][j][k-2+index];
				yi = temperature[i][j][k-2+index];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-4, -3, -2, -1, 0, 1, 2) for 2nd order derivative
		else if(k==num_vert_depth-3)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_z[i][j][k-4+index];
				yi = temperature[i][j][k-4+index];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-5, -4, -3, -2, -1, 0, 1) for 2nd order derivative
		else if(k==num_vert_depth-2)
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_z[i][j][k-5+index];
				yi = temperature[i][j][k-5+index];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		// (-3, -2, -1, 0, 1, 2, 3) for 2nd order derivative
		else
		{
			for (int index = 0; index < 7; index++)
			{
				xi = mesh_z[i][j][k-3+index];
				yi = temperature[i][j][k-3+index];
				sum_x_4 += xi*xi*xi*xi;
				sum_x_3 += xi*xi*xi;
				sum_x_2 += xi*xi;
				sum_x += xi;
				sum_x_2_y += xi*xi*yi;
				sum_x_y += xi*yi;
				sum_y += yi;
			}
		}
		
		// Populate B matrix
		B[0] = sum_x_2_y;
		B[1] = sum_x_y;
		B[2] = sum_y;
		
		// Populate A matrix
		A[0][0] = sum_x_4;
		A[1][0] = sum_x_3;
		A[2][0] = sum_x_2;
		A[0][1] = sum_x_3;
		A[1][1] = sum_x_2;
		A[2][1] = sum_x;
		A[0][2] = sum_x_2;
		A[1][2] = sum_x;
		A[2][2] = 7.0;
		
		A_inv = invert_3x3(A);
		fit_soln = mat_mul_3(A_inv, B);
		dT2_dz2 = 2.0 * fit_soln[0];
	}

	// Calculate the second derivative of temperature wrt z at boundaries
	else
	{
		// Top boundary condition
		if (k == 0)
		{
			top_flux = htc*(t_ijk-ambient_temperature) - input_mesh[i][j];
			dT2_dz2 = 2.0*(temperature[i][j][k+1]-t_ijk-(z_step*top_flux/thermal_conductivity))/(z_step*z_step);
		}
		
		// Bottom boundary condition
		else if (k == num_vert_depth-1)
		{
			bottom_flux = htc*(t_ijk-ambient_temperature);
			dT2_dz2 = 2.0*(temperature[i][j][k-1]-t_ijk-(z_step*bottom_flux/thermal_conductivity))/(z_step*z_step);
		}
	}
	
	return dT2_dx2+dT2_dy2+dT2_dz2;
}

/** Calculates the Laplacian of the temperature field at a specified location
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @return 7 stencil Lapclacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_7(int i, int j, int k, const vector<vector<vector<double>>> &temperature)
{
	double dT2_dx2;
	double dT2_dy2;
	double dT2_dz2;
	double left_flux;
	double right_flux;
	double front_flux;
	double back_flux;
	double top_flux;
	double bottom_flux;
	double term_1;
	double term_2;
	double term_3;
	double term_4;
	double term_5;
	double term_6;
	double term_7;
	double t_ijk = temperature[i][j][k];

	// Calculate the second derivative of temperature wrt x in interior of mesh
	if (i != 0 && i != num_vert_length-1)
	{
		// Stencil size 7. Stencil = (-1, 0, 1, 2, 3, 4, 5) for 2nd order derivative
		if (i==1)
		{
			term_1 = 137.0 * temperature[i-1][j][k] / 180.0;
			term_2 = -49.0 * t_ijk / 60.0;
			term_3 = -17.0 * temperature[i+1][j][k] / 12.0;
			term_4 = 47.0 * temperature[i+2][j][k] / 18.0;
			term_5 = -19.0 * temperature[i+3][j][k] / 12.0;
			term_6 = 31.0 * temperature[i+4][j][k] / 60.0;
			term_7 = -13.0 * temperature[i+5][j][k] / 180.0;
		}
		// Stencil size 7. Stencil = (-2, -1, 0, 1, 2, 3, 4) for 2nd order derivative
		else if(i==2)
		{
			term_1 = -13.0 * temperature[i-2][j][k] / 180.0;
			term_2 = 19.0 * temperature[i-1][j][k] / 15.0;
			term_3 = -7.0 * t_ijk / 3.0;
			term_4 = 10.0 * temperature[i+1][j][k] / 9.0;
			term_5 = 1.0 * temperature[i+2][j][k] / 12.0;
			term_6 = -1.0 * temperature[i+3][j][k] / 15.0;
			term_7 = 1.0 * temperature[i+4][j][k] / 90.0;
		}
		// Stencil size 7. Stencil = (-4, -3, -2, -1, 0, 1, 2) for 2nd order derivative
		else if(i==num_vert_length-3)
		{
			term_1 = 1.0 * temperature[i-4][j][k] / 90.0;
			term_2 = -1.0 * temperature[i-3][j][k] / 15.0;
			term_3 = 1.0 * temperature[i-2][j][k] / 12.0;
			term_4 = 10.0 * temperature[i-1][j][k] / 9.0;
			term_5 = -7.0 * t_ijk / 3.0;
			term_6 = 19.0 * temperature[i+1][j][k] / 15.0;
			term_7 = -13.0 * temperature[i+2][j][k] / 180.0;
		}
		// Stencil size 7. Stencil = (-5, -4, -3, -2, -1, 0, 1) for 2nd order derivative
		else if(i==num_vert_length-2)
		{
			term_1 = -13.0 * temperature[i-5][j][k] / 180.0;
			term_2 = 31.0 * temperature[i-4][j][k] / 60.0;
			term_3 = -19.0 * temperature[i-3][j][k] / 12.0;
			term_4 = 47.0 * temperature[i-2][j][k] / 18.0;
			term_5 = -17.0 * temperature[i-1][j][k] / 12.0;
			term_6 = -49.0 * t_ijk / 60.0;
			term_7 = 137.0 * temperature[i+1][j][k] / 180.0;
		}
		// Stencil size 7. Stencil = (-3, -2, -1, 0, 1, 2, 3) for 2nd order derivative
		else
		{
			term_1 = 1.0 * temperature[i-3][j][k] / 90.0;
			term_2 = -3.0 * temperature[i-2][j][k] / 20.0;
			term_3 = 3.0 * temperature[i-1][j][k] / 2.0;
			term_4 = -49.0 * t_ijk / 18.0;
			term_5 = 3.0 * temperature[i+1][j][k] / 2.0;
			term_6 = -3.0 * temperature[i+2][j][k] / 20.0;
			term_7 = 1.0 * temperature[i+3][j][k] / 90.0;
		}
		// Get second derivative based on stencile length 7 finite difference method
		dT2_dx2 = (term_1+term_2+term_3+term_4+term_5+term_6+term_7) / (x_step*x_step);
		
	}

	// Calculate the second derivative of temperature wrt x at boundaries
	else
	{
		// LHS boundary condition
		if (i == 0)
		{
			// Trigger boundary condition
			if (current_time >= trigger_time && current_time < trigger_time + trigger_duration) { left_flux = htc*(t_ijk-ambient_temperature) - trigger_flux; }
			
			// Non-trigger boundary condition
			else { left_flux = htc*(t_ijk-ambient_temperature); }
			
			dT2_dx2 = 2.0*( temperature[i+1][j][k]-t_ijk-(x_step*left_flux/thermal_conductivity) ) / (x_step*x_step);
		}
		
		// RHS boundary condition
		else if (i == num_vert_length-1)
		{
			right_flux = htc*(t_ijk-ambient_temperature);
			dT2_dx2 = 2.0*( temperature[i-1][j][k] - t_ijk - (x_step*right_flux/thermal_conductivity) ) / (x_step*x_step);
		}
	}

	// Calculate the second derivative of temperature wrt y in interior of mesh
	if (j != 0 && j != num_vert_width-1)
	{
		// Stencil size 7. Stencil = (-1, 0, 1, 2, 3, 4, 5) for 2nd order derivative
		if (j==1)
		{
			term_1 = 137.0 * temperature[i][j-1][k] / 180.0;
			term_2 = -49.0 * t_ijk / 60.0;
			term_3 = -17.0 * temperature[i][j+1][k] / 12.0;
			term_4 = 47.0 * temperature[i][j+2][k] / 18.0;
			term_5 = -19.0 * temperature[i][j+3][k] / 12.0;
			term_6 = 31.0 * temperature[i][j+4][k] / 60.0;
			term_7 = -13.0 * temperature[i][j+5][k] / 180.0;
		}
		// Stencil size 7. Stencil = (-2, -1, 0, 1, 2, 3, 4) for 2nd order derivative
		else if(j==2)
		{
			term_1 = -13.0 * temperature[i][j-2][k] / 180.0;
			term_2 = 19.0 * temperature[i][j-1][k] / 15.0;
			term_3 = -7.0 * t_ijk / 3.0;
			term_4 = 10.0 * temperature[i][j+1][k] / 9.0;
			term_5 = 1.0 * temperature[i][j+2][k] / 12.0;
			term_6 = -1.0 * temperature[i][j+3][k] / 15.0;
			term_7 = 1.0 * temperature[i][j+4][k] / 90.0;
		}
		// Stencil size 7. Stencil = (-4, -3, -2, -1, 0, 1, 2) for 2nd order derivative
		else if(j==num_vert_width-3)
		{
			term_1 = 1.0 * temperature[i][j-4][k] / 90.0;
			term_2 = -1.0 * temperature[i][j-3][k] / 15.0;
			term_3 = 1.0 * temperature[i][j-2][k] / 12.0;
			term_4 = 10.0 * temperature[i][j-1][k] / 9.0;
			term_5 = -7.0 * t_ijk / 3.0;
			term_6 = 19.0 * temperature[i][j+1][k] / 15.0;
			term_7 = -13.0 * temperature[i][j+2][k] / 180.0;
		}
		// Stencil size 7. Stencil = (-5, -4, -3, -2, -1, 0, 1) for 2nd order derivative
		else if(j==num_vert_width-2)
		{
			term_1 = -13.0 * temperature[i][j-5][k] / 180.0;
			term_2 = 31.0 * temperature[i][j-4][k] / 60.0;
			term_3 = -19.0 * temperature[i][j-3][k] / 12.0;
			term_4 = 47.0 * temperature[i][j-2][k] / 18.0;
			term_5 = -17.0 * temperature[i][j-1][k] / 12.0;
			term_6 = -49.0 * t_ijk / 60.0;
			term_7 = 137.0 * temperature[i][j+1][k] / 180.0;
		}
		// Stencil size 7. Stencil = (-3, -2, -1, 0, 1, 2, 3) for 2nd order derivative
		else
		{
			term_1 = 1.0 * temperature[i][j-3][k] / 90.0;
			term_2 = -3.0 * temperature[i][j-2][k] / 20.0;
			term_3 = 3.0 * temperature[i][j-1][k] / 2.0;
			term_4 = -49.0 * t_ijk / 18.0;
			term_5 = 3.0 * temperature[i][j+1][k] / 2.0;
			term_6 = -3.0 * temperature[i][j+2][k] / 20.0;
			term_7 = 1.0 * temperature[i][j+3][k] / 90.0;
		}
		// Get second derivative based on stencile length 7 finite difference method
		dT2_dy2 = (term_1+term_2+term_3+term_4+term_5+term_6+term_7) / (y_step*y_step);
	}

	// Calculate the second derivative of temperature wrt y at boundaries
	else
	{
		// Front boundary condition
		if (j == 0)
		{
			front_flux = htc*(t_ijk-ambient_temperature);
			dT2_dy2 = 2.0*(temperature[i][j+1][k]-t_ijk-(y_step*front_flux/thermal_conductivity))/(y_step*y_step);
		}
		
		// Back boundary condition
		else if (j == num_vert_width-1)
		{
			back_flux = htc*(t_ijk-ambient_temperature);
			dT2_dy2 = 2.0*(temperature[i][j-1][k]-t_ijk-(y_step*back_flux/thermal_conductivity))/(y_step*y_step);
		}
	}

	// Calculate the second derivative of temperature wrt z in interior of mesh
	if (k != 0 && k != num_vert_depth-1)
	{
		// Stencil size 7. Stencil = (-1, 0, 1, 2, 3, 4, 5) for 2nd order derivative
		if (k==1)
		{
			term_1 = 137.0 * temperature[i][j][k-1] / 180.0;
			term_2 = -49.0 * t_ijk / 60.0;
			term_3 = -17.0 * temperature[i][j][k+1] / 12.0;
			term_4 = 47.0 * temperature[i][j][k+2] / 18.0;
			term_5 = -19.0 * temperature[i][j][k+3] / 12.0;
			term_6 = 31.0 * temperature[i][j][k+4] / 60.0;
			term_7 = -13.0 * temperature[i][j][k+5] / 180.0;
		}
		// Stencil size 7. Stencil = (-2, -1, 0, 1, 2, 3, 4) for 2nd order derivative
		else if(k==2)
		{
			term_1 = -13.0 * temperature[i][j][k-2] / 180.0;
			term_2 = 19.0 * temperature[i][j][k-1] / 15.0;
			term_3 = -7.0 * t_ijk / 3.0;
			term_4 = 10.0 * temperature[i][j][k+1] / 9.0;
			term_5 = 1.0 * temperature[i][j][k+2] / 12.0;
			term_6 = -1.0 * temperature[i][j][k+3] / 15.0;
			term_7 = 1.0 * temperature[i][j][k+4] / 90.0;
		}
		// Stencil size 7. Stencil = (-4, -3, -2, -1, 0, 1, 2) for 2nd order derivative
		else if(k==num_vert_depth-3)
		{
			term_1 = 1.0 * temperature[i][j][k-4] / 90.0;
			term_2 = -1.0 * temperature[i][j][k-3] / 15.0;
			term_3 = 1.0 * temperature[i][j][k-2] / 12.0;
			term_4 = 10.0 * temperature[i][j][k-1] / 9.0;
			term_5 = -7.0 * t_ijk / 3.0;
			term_6 = 19.0 * temperature[i][j][k+1] / 15.0;
			term_7 = -13.0 * temperature[i][j][k+2] / 180.0;
		}
		// Stencil size 7. Stencil = (-5, -4, -3, -2, -1, 0, 1) for 2nd order derivative
		else if(k==num_vert_depth-2)
		{
			term_1 = -13.0 * temperature[i][j][k-5] / 180.0;
			term_2 = 31.0 * temperature[i][j][k-4] / 60.0;
			term_3 = -19.0 * temperature[i][j][k-3] / 12.0;
			term_4 = 47.0 * temperature[i][j][k-2] / 18.0;
			term_5 = -17.0 * temperature[i][j][k-1] / 12.0;
			term_6 = -49.0 * t_ijk / 60.0;
			term_7 = 137.0 * temperature[i][j][k+1] / 180.0;
		}
		// Stencil size 7. Stencil = (-3, -2, -1, 0, 1, 2, 3) for 2nd order derivative
		else
		{
			term_1 = 1.0 * temperature[i][j][k-3] / 90.0;
			term_2 = -3.0 * temperature[i][j][k-2] / 20.0;
			term_3 = 3.0 * temperature[i][j][k-1] / 2.0;
			term_4 = -49.0 * t_ijk / 18.0;
			term_5 = 3.0 * temperature[i][j][k+1] / 2.0;
			term_6 = -3.0 * temperature[i][j][k+2] / 20.0;
			term_7 = 1.0 * temperature[i][j][k+3] / 90.0;
		}
		// Get second derivative based on stencile length 7 finite difference method
		dT2_dz2 = (term_1+term_2+term_3+term_4+term_5+term_6+term_7) / (z_step*z_step);
	}

	// Calculate the second derivative of temperature wrt z at boundaries
	else
	{
		// Top boundary condition
		if (k == 0)
		{
			top_flux = htc*(t_ijk-ambient_temperature) - input_mesh[i][j];
			dT2_dz2 = 2.0*(temperature[i][j][k+1]-t_ijk-(z_step*top_flux/thermal_conductivity))/(z_step*z_step);
		}
		
		// Bottom boundary condition
		else if (k == num_vert_depth-1)
		{
			bottom_flux = htc*(t_ijk-ambient_temperature);
			dT2_dz2 = 2.0*(temperature[i][j][k-1]-t_ijk-(z_step*bottom_flux/thermal_conductivity))/(z_step*z_step);
		}
	}
	
	return dT2_dx2+dT2_dy2+dT2_dz2;
}

/** Calculates the Laplacian of the temperature field at a specified location
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @return 9 stencil Lapclacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_9(int i, int j, int k, const vector<vector<vector<double>>> &temperature)
{
	double dT2_dx2;
	double dT2_dy2;
	double dT2_dz2;
	double left_flux;
	double right_flux;
	double front_flux;
	double back_flux;
	double top_flux;
	double bottom_flux;
	double term_1;
	double term_2;
	double term_3;
	double term_4;
	double term_5;
	double term_6;
	double term_7;
	double term_8;
	double term_9;
	double t_ijk = temperature[i][j][k];

	// Calculate the second derivative of temperature wrt x in interior of mesh
	if (i != 0 && i != num_vert_length-1)
	{
		// Stencil size 9. Stencil = (-1, 0, 1, 2, 3, 4, 5, 6, 7) for 2nd order derivative
		if (i==1)
		{
			term_1 =  0.648214285714286 * temperature[i-1][j][k];
			term_2 =  0.025396825396220 * t_ijk;
			term_3 =  -4.15 * temperature[i+1][j][k];
			term_4 =  7.65 * temperature[i+2][j][k];
			term_5 =  -7.34722222222351 * temperature[i+3][j][k];
			term_6 =  4.7 * temperature[i+4][j][k];
			term_7 =  -1.95 * temperature[i+5][j][k];
			term_8 =  0.475396825396825 * temperature[i+6][j][k];
			term_9 =  -0.0517857142857143 * temperature[i+7][j][k];
		}
		// Stencil size 9. Stencil = (-2, -1, 0, 1, 2, 3, 4, 5, 6) for 2nd order derivative
		else if(i==2)
		{
			term_1 = -261.0 * temperature[i-2][j][k] / 5040.0;
			term_2 = 5616.0 * temperature[i-1][j][k] / 5040.0;
			term_3 = -9268.0 * t_ijk / 5040.0;
			term_4 = 1008.0 * temperature[i+1][j][k] / 5040.0;
			term_5 = 5670.0 * temperature[i+2][j][k] / 5040.0;
			term_6 = -4144.0 * temperature[i+3][j][k] / 5040.0;
			term_7 = 1764.0 * temperature[i+4][j][k] / 5040.0;
			term_8 = -432.0 * temperature[i+5][j][k] / 5040.0;
			term_9 = 47.0 * temperature[i+6][j][k] / 5040.0;
		}
		// Stencil size 9. Stencil = (-3, -2, -1, 0, 1, 2, 3, 4, 5) for 2nd order derivative
		else if(i==3)
		{
			term_1 = 47.0 * temperature[i-3][j][k] / 5040.0;
			term_2 = -684.0 * temperature[i-2][j][k] / 5040.0;
			term_3 = 7308.0 * temperature[i-1][j][k] / 5040.0;
			term_4 = -13216.0 * t_ijk / 5040.0;
			term_5 = 6930.0 * temperature[i+1][j][k] / 5040.0;
			term_6 = -252.0 * temperature[i+2][j][k] / 5040.0;
			term_7 = -196.0 * temperature[i+3][j][k] / 5040.0;
			term_8 = 72.0 * temperature[i+4][j][k] / 5040.0;
			term_9 = -9.0 * temperature[i+5][j][k] / 5040.0;
		}
		// Stencil size 9. Stencil = (-5, -4, -3, -2, -1, 0, 1, 2, 3) for 2nd order derivative
		else if(i==num_vert_length-4)
		{
			term_1 = -9.0 * temperature[i-5][j][k] / 5040.0;
			term_2 = 72.0 * temperature[i-4][j][k] / 5040.0;
			term_3 = -196.0 * temperature[i-3][j][k] / 5040.0;
			term_4 = -252.0 * temperature[i-2][j][k] / 5040.0;
			term_5 = 6930.0 * temperature[i-1][j][k] / 5040.0;
			term_6 = -13216.0 * t_ijk / 5040.0;
			term_7 = 7308.0 * temperature[i+1][j][k] / 5040.0;
			term_8 = -684.0 * temperature[i+2][j][k] / 5040.0;
			term_9 = 47.0 * temperature[i+3][j][k] / 5040.0;
		}
		// Stencil size 9. Stencil = (-6, -5, -4, -3, -2, -1, 0, 1, 2) for 2nd order derivative
		else if(i==num_vert_length-3)
		{
			term_1 = 47.0 * temperature[i-6][j][k] / 5040.0;
			term_2 = -432.0 * temperature[i-5][j][k] / 5040.0;
			term_3 = 1764.0 * temperature[i-4][j][k] / 5040.0;
			term_4 = -4144.0 * temperature[i-3][j][k] / 5040.0;
			term_5 = 5670.0 * temperature[i-2][j][k] / 5040.0;
			term_6 = 1008.0 * temperature[i-1][j][k] / 5040.0;
			term_7 = -9268.0 * t_ijk / 5040.0;
			term_8 = 5615.0 * temperature[i+1][j][k] / 5040.0;
			term_9 = -261.0 * temperature[i+2][j][k] / 5040.0;
		}
		// Stencil size 9. Stencil = (-7, -6, -5, -4, -3, -2, -1, 0, 1) for 2nd order derivative
		else if(i==num_vert_length-2)
		{
			term_1 = -0.0517857142857143 * temperature[i-7][j][k];
			term_2 = 0.475396825396825 * temperature[i-6][j][k];
			term_3 = -1.95 * temperature[i-5][j][k];
			term_4 = 4.7 * temperature[i-4][j][k];
			term_5 = -7.34722222222351 * temperature[i-3][j][k];
			term_6 = 7.65 * temperature[i-2][j][k];
			term_7 = -4.15 * temperature[i-1][j][k];
			term_8 = 0.025396825396220 * t_ijk;
			term_9 = 0.648214285714286 * temperature[i+1][j][k];
		}
		// Stencil size 9. Stencil = (-4, -3, -2, -1, 0, 1, 2, 3, 4) for 2nd order derivative
		else
		{
			term_1 = -9.0 * temperature[i-4][j][k] / 5040.0;
			term_2 = 128.0 * temperature[i-3][j][k] / 5040.0;
			term_3 = -1008.0 * temperature[i-2][j][k] / 5040.0;
			term_4 = 8064.0 * temperature[i-1][j][k] / 5040.0;
			term_5 = -14350.0 * t_ijk / 5040.0;
			term_6 = 8064.0 * temperature[i+1][j][k] / 5040.0;
			term_7 = -1008.0 * temperature[i+2][j][k] / 5040.0;
			term_8 = 128.0 * temperature[i+3][j][k] / 5040.0;
			term_9 = -9.0 * temperature[i+4][j][k] / 5040.0;
		}
		// Get second derivative based on stencile length 7 finite difference method
		dT2_dx2 = (term_1+term_2+term_3+term_4+term_5+term_6+term_7+term_8+term_9) / (x_step*x_step);
		
	}

	// Calculate the second derivative of temperature wrt x at boundaries
	else
	{
		// LHS boundary condition
		if (i == 0)
		{
			// Trigger boundary condition
			if (current_time >= trigger_time && current_time < trigger_time + trigger_duration) { left_flux = htc*(t_ijk-ambient_temperature) - trigger_flux; }
			
			// Non-trigger boundary condition
			else { left_flux = htc*(t_ijk-ambient_temperature); }
			
			dT2_dx2 = 2.0*( temperature[i+1][j][k]-t_ijk-(x_step*left_flux/thermal_conductivity) ) / (x_step*x_step);
		}
		
		// RHS boundary condition
		else if (i == num_vert_length-1)
		{
			right_flux = htc*(t_ijk-ambient_temperature);
			dT2_dx2 = 2.0*( temperature[i-1][j][k] - t_ijk - (x_step*right_flux/thermal_conductivity) ) / (x_step*x_step);
		}
	}

	// Calculate the second derivative of temperature wrt y in interior of mesh
	if (j != 0 && j != num_vert_width-1)
	{
		// Stencil size 9. Stencil = (-1, 0, 1, 2, 3, 4, 5, 6, 7) for 2nd order derivative
		if (j==1)
		{
			term_1 =  0.648214285714286 * temperature[i][j-1][k];
			term_2 =  0.025396825396220 * t_ijk;
			term_3 =  -4.15 * temperature[i][j+1][k];
			term_4 =  7.65 * temperature[i][j+2][k];
			term_5 =  -7.34722222222351 * temperature[i][j+3][k];
			term_6 =  4.7 * temperature[i][j+4][k];
			term_7 =  -1.95 * temperature[i][j+5][k];
			term_8 =  0.475396825396825 * temperature[i][j+6][k];
			term_9 =  -0.0517857142857143 * temperature[i][j+7][k];
		}
		// Stencil size 9. Stencil = (-2, -1, 0, 1, 2, 3, 4, 5, 6) for 2nd order derivative
		else if(j==2)
		{
			term_1 = -261.0 * temperature[i][j-2][k] / 5040.0;
			term_2 = 5616.0 * temperature[i][j-1][k] / 5040.0;
			term_3 = -9268.0 * t_ijk / 5040.0;
			term_4 = 1008.0 * temperature[i][j+1][k] / 5040.0;
			term_5 = 5670.0 * temperature[i][j+2][k] / 5040.0;
			term_6 = -4144.0 * temperature[i][j+3][k] / 5040.0;
			term_7 = 1764.0 * temperature[i][j+4][k] / 5040.0;
			term_8 = -432.0 * temperature[i][j+5][k] / 5040.0;
			term_9 = 47.0 * temperature[i][j+6][k] / 5040.0;
		}
		// Stencil size 9. Stencil = (-3, -2, -1, 0, 1, 2, 3, 4, 5) for 2nd order derivative
		else if(j==3)
		{
			term_1 = 47.0 * temperature[i][j-3][k] / 5040.0;
			term_2 = -684.0 * temperature[i][j-2][k] / 5040.0;
			term_3 = 7308.0 * temperature[i][j-1][k] / 5040.0;
			term_4 = -13216.0 * t_ijk / 5040.0;
			term_5 = 6930.0 * temperature[i][j+1][k] / 5040.0;
			term_6 = -252.0 * temperature[i][j+2][k] / 5040.0;
			term_7 = -196.0 * temperature[i][j+3][k] / 5040.0;
			term_8 = 72.0 * temperature[i][j+4][k] / 5040.0;
			term_9 = -9.0 * temperature[i][j+5][k] / 5040.0;
		}
		// Stencil size 9. Stencil = (-5, -4, -3, -2, -1, 0, 1, 2, 3) for 2nd order derivative
		else if(j==num_vert_width-4)
		{
			term_1 = -9.0 * temperature[i][j-5][k] / 5040.0;
			term_2 = 72.0 * temperature[i][j-4][k] / 5040.0;
			term_3 = -196.0 * temperature[i][j-3][k] / 5040.0;
			term_4 = -252.0 * temperature[i][j-2][k] / 5040.0;
			term_5 = 6930.0 * temperature[i][j-1][k] / 5040.0;
			term_6 = -13216.0 * t_ijk / 5040.0;
			term_7 = 7308.0 * temperature[i][j+1][k] / 5040.0;
			term_8 = -684.0 * temperature[i][j+2][k] / 5040.0;
			term_9 = 47.0 * temperature[i][j+3][k] / 5040.0;
		}
		// Stencil size 9. Stencil = (-6, -5, -4, -3, -2, -1, 0, 1, 2) for 2nd order derivative
		else if(j==num_vert_width-3)
		{
			term_1 = 47.0 * temperature[i][j-6][k] / 5040.0;
			term_2 = -432.0 * temperature[i][j-5][k] / 5040.0;
			term_3 = 1764.0 * temperature[i][j-4][k] / 5040.0;
			term_4 = -4144.0 * temperature[i][j-3][k] / 5040.0;
			term_5 = 5670.0 * temperature[i][j-2][k] / 5040.0;
			term_6 = 1008.0 * temperature[i][j-1][k] / 5040.0;
			term_7 = -9268.0 * t_ijk / 5040.0;
			term_8 = 5615.0 * temperature[i][j+1][k] / 5040.0;
			term_9 = -261.0 * temperature[i][j+2][k] / 5040.0;
		}
		// Stencil size 9. Stencil = (-7, -6, -5, -4, -3, -2, -1, 0, 1) for 2nd order derivative
		else if(j==num_vert_width-2)
		{
			term_1 = -0.0517857142857143 * temperature[i][j-7][k];
			term_2 = 0.475396825396825 * temperature[i][j-6][k];
			term_3 = -1.95 * temperature[i][j-5][k];
			term_4 = 4.7 * temperature[i][j-4][k];
			term_5 = -7.34722222222351 * temperature[i][j-3][k];
			term_6 = 7.65 * temperature[i][j-2][k];
			term_7 = -4.15 * temperature[i][j-1][k];
			term_8 = 0.025396825396220 * t_ijk;
			term_9 = 0.648214285714286 * temperature[i][j+1][k];
		}
		// Stencil size 9. Stencil = (-4, -3, -2, -1, 0, 1, 2, 3, 4) for 2nd order derivative
		else
		{
			term_1 = -9.0 * temperature[i][j-4][k] / 5040.0;
			term_2 = 128.0 * temperature[i][j-3][k] / 5040.0;
			term_3 = -1008.0 * temperature[i][j-2][k] / 5040.0;
			term_4 = 8064.0 * temperature[i][j-1][k] / 5040.0;
			term_5 = -14350.0 * t_ijk / 5040.0;
			term_6 = 8064.0 * temperature[i][j+1][k] / 5040.0;
			term_7 = -1008.0 * temperature[i][j+2][k] / 5040.0;
			term_8 = 128.0 * temperature[i][j+3][k] / 5040.0;
			term_9 = -9.0 * temperature[i][j+4][k] / 5040.0;
		}
		// Get second derivative based on stencile length 7 finite difference method
		dT2_dy2 = (term_1+term_2+term_3+term_4+term_5+term_6+term_7+term_8+term_9) / (y_step*y_step);
	}

	// Calculate the second derivative of temperature wrt y at boundaries
	else
	{
		// Front boundary condition
		if (j == 0)
		{
			front_flux = htc*(t_ijk-ambient_temperature);
			dT2_dy2 = 2.0*(temperature[i][j+1][k]-t_ijk-(y_step*front_flux/thermal_conductivity))/(y_step*y_step);
		}
		
		// Back boundary condition
		else if (j == num_vert_width-1)
		{
			back_flux = htc*(t_ijk-ambient_temperature);
			dT2_dy2 = 2.0*(temperature[i][j-1][k]-t_ijk-(y_step*back_flux/thermal_conductivity))/(y_step*y_step);
		}
	}

	// Calculate the second derivative of temperature wrt z in interior of mesh
	if (k != 0 && k != num_vert_depth-1)
	{
		// Stencil size 9. Stencil = (-1, 0, 1, 2, 3, 4, 5, 6, 7) for 2nd order derivative
		if (k==1)
		{
			term_1 =  0.648214285714286 * temperature[i][j][k-1];
			term_2 =  0.025396825396220 * t_ijk;
			term_3 =  -4.15 * temperature[i][j][k+1];
			term_4 =  7.65 * temperature[i][j][k+2];
			term_5 =  -7.34722222222351 * temperature[i][j][k+3];
			term_6 =  4.7 * temperature[i][j][k+4];
			term_7 =  -1.95 * temperature[i][j][k+5];
			term_8 =  0.475396825396825 * temperature[i][j][k+6];
			term_9 =  -0.0517857142857143 * temperature[i][j][k+7];
		}
		// Stencil size 9. Stencil = (-2, -1, 0, 1, 2, 3, 4, 5, 6) for 2nd order derivative
		else if(k==2)
		{
			term_1 = -261.0 * temperature[i][j][k-2] / 5040.0;
			term_2 = 5616.0 * temperature[i][j][k-1] / 5040.0;
			term_3 = -9268.0 * t_ijk / 5040.0;
			term_4 = 1008.0 * temperature[i][j][k+1] / 5040.0;
			term_5 = 5670.0 * temperature[i][j][k+2] / 5040.0;
			term_6 = -4144.0 * temperature[i][j][k+3] / 5040.0;
			term_7 = 1764.0 * temperature[i][j][k+4] / 5040.0;
			term_8 = -432.0 * temperature[i][j][k+5] / 5040.0;
			term_9 = 47.0 * temperature[i][j][k+6] / 5040.0;
		}
		// Stencil size 9. Stencil = (-3, -2, -1, 0, 1, 2, 3, 4, 5) for 2nd order derivative
		else if(k==3)
		{
			term_1 = 47.0 * temperature[i][j][k-3] / 5040.0;
			term_2 = -684.0 * temperature[i][j][k-2] / 5040.0;
			term_3 = 7308.0 * temperature[i][j][k-1] / 5040.0;
			term_4 = -13216.0 * t_ijk / 5040.0;
			term_5 = 6930.0 * temperature[i][j][k+1] / 5040.0;
			term_6 = -252.0 * temperature[i][j][k+2] / 5040.0;
			term_7 = -196.0 * temperature[i][j][k+3] / 5040.0;
			term_8 = 72.0 * temperature[i][j][k+4] / 5040.0;
			term_9 = -9.0 * temperature[i][j][k+5] / 5040.0;
		}
		// Stencil size 9. Stencil = (-5, -4, -3, -2, -1, 0, 1, 2, 3) for 2nd order derivative
		else if(k==num_vert_depth-4)
		{
			term_1 = -9.0 * temperature[i][j][k-5] / 5040.0;
			term_2 = 72.0 * temperature[i][j][k-4] / 5040.0;
			term_3 = -196.0 * temperature[i][j][k-3] / 5040.0;
			term_4 = -252.0 * temperature[i][j][k-2] / 5040.0;
			term_5 = 6930.0 * temperature[i][j][k-1] / 5040.0;
			term_6 = -13216.0 * t_ijk / 5040.0;
			term_7 = 7308.0 * temperature[i][j][k+1] / 5040.0;
			term_8 = -684.0 * temperature[i][j][k+2] / 5040.0;
			term_9 = 47.0 * temperature[i][j][k+3] / 5040.0;
		}
		// Stencil size 9. Stencil = (-6, -5, -4, -3, -2, -1, 0, 1, 2) for 2nd order derivative
		else if(k==num_vert_depth-3)
		{
			term_1 = 47.0 * temperature[i][j][k-6] / 5040.0;
			term_2 = -432.0 * temperature[i][j][k-5] / 5040.0;
			term_3 = 1764.0 * temperature[i][j][k-4] / 5040.0;
			term_4 = -4144.0 * temperature[i][j][k-3] / 5040.0;
			term_5 = 5670.0 * temperature[i][j][k-2] / 5040.0;
			term_6 = 1008.0 * temperature[i][j][k-1] / 5040.0;
			term_7 = -9268.0 * t_ijk / 5040.0;
			term_8 = 5615.0 * temperature[i][j][k+1] / 5040.0;
			term_9 = -261.0 * temperature[i][j][k+2] / 5040.0;
		}
		// Stencil size 9. Stencil = (-7, -6, -5, -4, -3, -2, -1, 0, 1) for 2nd order derivative
		else if(k==num_vert_depth-2)
		{
			term_1 = -0.0517857142857143 * temperature[i][j][k-7];
			term_2 = 0.475396825396825 * temperature[i][j][k-6];
			term_3 = -1.95 * temperature[i][j][k-5];
			term_4 = 4.7 * temperature[i][j][k-4];
			term_5 = -7.34722222222351 * temperature[i][j][k-3];
			term_6 = 7.65 * temperature[i][j][k-2];
			term_7 = -4.15 * temperature[i][j][k-1];
			term_8 = 0.025396825396220 * t_ijk;
			term_9 = 0.648214285714286 * temperature[i][j][k+1];
		}
		// Stencil size 9. Stencil = (-4, -3, -2, -1, 0, 1, 2, 3, 4) for 2nd order derivative
		else
		{
			term_1 = -9.0 * temperature[i][j][k-4] / 5040.0;
			term_2 = 128.0 * temperature[i][j][k-3] / 5040.0;
			term_3 = -1008.0 * temperature[i][j][k-2] / 5040.0;
			term_4 = 8064.0 * temperature[i][j][k-1] / 5040.0;
			term_5 = -14350.0 * t_ijk / 5040.0;
			term_6 = 8064.0 * temperature[i][j][k+1] / 5040.0;
			term_7 = -1008.0 * temperature[i][j][k+2] / 5040.0;
			term_8 = 128.0 * temperature[i][j][k+3] / 5040.0;
			term_9 = -9.0 * temperature[i][j][k+4] / 5040.0;
		}
		// Get second derivative based on stencile length 7 finite difference method
		dT2_dz2 = (term_1+term_2+term_3+term_4+term_5+term_6+term_7+term_8+term_9) / (z_step*z_step);
	}

	// Calculate the second derivative of temperature wrt z at boundaries
	else
	{
		// Top boundary condition
		if (k == 0)
		{
			top_flux = htc*(t_ijk-ambient_temperature) - input_mesh[i][j];
			dT2_dz2 = 2.0*(temperature[i][j][k+1]-t_ijk-(z_step*top_flux/thermal_conductivity))/(z_step*z_step);
		}
		
		// Bottom boundary condition
		else if (k == num_vert_depth-1)
		{
			bottom_flux = htc*(t_ijk-ambient_temperature);
			dT2_dz2 = 2.0*(temperature[i][j][k-1]-t_ijk-(z_step*bottom_flux/thermal_conductivity))/(z_step*z_step);
		}
	}
	
	return dT2_dx2+dT2_dy2+dT2_dz2;
}

/** Calculates the Laplacian of the temperature field at a specified location
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @return 5 stencil Lapclacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_5(int i, int j, int k, const vector<vector<vector<double>>> &temperature)
{
	double dT2_dx2;
	double dT2_dy2;
	double dT2_dz2;
	double left_flux;
	double right_flux;
	double front_flux;
	double back_flux;
	double top_flux;
	double bottom_flux;
	double term_1;
	double term_2;
	double term_3;
	double term_4;
	double term_5;
	double t_ijk = temperature[i][j][k];

	// Calculate the second derivative of temperature wrt x in interior of mesh
	if (i != 0 && i != num_vert_length-1)
	{
		// Stencil size 5. Stencil = (-1, 0, 1, 2, 3) for 2nd order derivative
		if (i==1)
		{
			term_1 = 11.0 * temperature[i-1][j][k] / 12.0;
			term_2 = -5.0 * t_ijk / 3.0;
			term_3 = 1.0 * temperature[i+1][j][k] / 2.0;
			term_4 = 1.0 * temperature[i+2][j][k] / 3.0;
			term_5 = -1.0 * temperature[i+3][j][k] / 12.0;
		}
		// Stencil size 5. Stencil = (-3, -2, -1, 0, 1) for 2nd order derivative
		else if(i==num_vert_length-2)
		{
			term_1 = -1.0 * temperature[i-3][j][k] / 12.0;
			term_2 = 1.0 * temperature[i-2][j][k] / 3.0;
			term_3 = 1.0 * temperature[i-1][j][k] / 2.0;
			term_4 = -5.0 * t_ijk / 3.0;
			term_5 = 11.0 * temperature[i+1][j][k] / 12.0;
		}
		// Stencil size 5. Stencil = (-2, -1, 0, 1, 2) for 2nd order derivative
		else
		{
			term_1 = -1.0 * temperature[i-2][j][k] / 12.0;
			term_2 = 4.0 * temperature[i-1][j][k] / 3.0;
			term_3 = -5.0 * t_ijk / 2.0;
			term_4 = 4.0 * temperature[i+1][j][k] / 3.0;
			term_5 = -1.0 * temperature[i+2][j][k] / 12.0;
		}
		// Get second derivative based on stencile length 5 finite difference method
		dT2_dx2 = (term_1+term_2+term_3+term_4+term_5) / (x_step*x_step);
		
	}

	// Calculate the second derivative of temperature wrt x at boundaries
	else
	{
		// LHS boundary condition
		if (i == 0)
		{
			// Trigger boundary condition
			if (current_time >= trigger_time && current_time < trigger_time + trigger_duration) { left_flux = htc*(t_ijk-ambient_temperature) - trigger_flux; }
			
			// Non-trigger boundary condition
			else { left_flux = htc*(t_ijk-ambient_temperature); }
			
			dT2_dx2 = 2.0*( temperature[i+1][j][k]-t_ijk-(x_step*left_flux/thermal_conductivity) ) / (x_step*x_step);
		}
		
		// RHS boundary condition
		else if (i == num_vert_length-1)
		{
			right_flux = htc*(t_ijk-ambient_temperature);
			dT2_dx2 = 2.0*( temperature[i-1][j][k] - t_ijk - (x_step*right_flux/thermal_conductivity) ) / (x_step*x_step);
		}
	}

	// Calculate the second derivative of temperature wrt y in interior of mesh
	if (j != 0 && j != num_vert_width-1)
	{
		// Stencil size 5. Stencil = (-1, 0, 1, 2, 3) for 2nd order derivative
		if (j==1)
		{
			term_1 = 11.0 * temperature[i][j-1][k] / 12.0;
			term_2 = -5.0 * t_ijk / 3.0;
			term_3 = 1.0 * temperature[i][j+1][k] / 2.0;
			term_4 = 1.0 * temperature[i][j+2][k] / 3.0;
			term_5 = -1.0 * temperature[i][j+3][k] / 12.0;
		}
		// Stencil size 5. Stencil = (-3, -2, -1, 0, 1) for 2nd order derivative
		else if(j==num_vert_width-2)
		{
			term_1 = -1.0 * temperature[i][j-3][k] / 12.0;
			term_2 = 1.0 * temperature[i][j-2][k] / 3.0;
			term_3 = 1.0 * temperature[i][j-1][k] / 2.0;
			term_4 = -5.0 * t_ijk / 3.0;
			term_5 = 11.0 * temperature[i][j+1][k] / 12.0;
		}
		// Stencil size 5. Stencil = (-2, -1, 0, 1, 2) for 2nd order derivative
		else
		{
			term_1 = -1.0 * temperature[i][j-2][k] / 12.0;
			term_2 = 4.0 * temperature[i][j-1][k] / 3.0;
			term_3 = -5.0 * t_ijk / 2.0;
			term_4 = 4.0 * temperature[i][j+1][k] / 3.0;
			term_5 = -1.0 * temperature[i][j+2][k] / 12.0;
		}
		// Get second derivative based on stencile length 7 finite difference method
		dT2_dy2 = (term_1+term_2+term_3+term_4+term_5) / (y_step*y_step);
	}

	// Calculate the second derivative of temperature wrt y at boundaries
	else
	{
		// Front boundary condition
		if (j == 0)
		{
			front_flux = htc*(t_ijk-ambient_temperature);
			dT2_dy2 = 2.0*(temperature[i][j+1][k]-t_ijk-(y_step*front_flux/thermal_conductivity))/(y_step*y_step);
		}
		
		// Back boundary condition
		else if (j == num_vert_width-1)
		{
			back_flux = htc*(t_ijk-ambient_temperature);
			dT2_dy2 = 2.0*(temperature[i][j-1][k]-t_ijk-(y_step*back_flux/thermal_conductivity))/(y_step*y_step);
		}
	}

	// Calculate the second derivative of temperature wrt z in interior of mesh
	if (k != 0 && k != num_vert_depth-1)
	{
		// Stencil size 5. Stencil = (-1, 0, 1, 2, 3) for 2nd order derivative
		if (k==1)
		{
			term_1 = 11.0 * temperature[i][j][k-1] / 12.0;
			term_2 = -5.0 * t_ijk / 3.0;
			term_3 = 1.0 * temperature[i][j][k+1] / 2.0;
			term_4 = 1.0 * temperature[i][j][k+2] / 3.0;
			term_5 = -1.0 * temperature[i][j][k+3] / 12.0;
		}
		// Stencil size 5. Stencil = (-3, -2, -1, 0, 1) for 2nd order derivative
		else if(k==num_vert_depth-2)
		{
			term_1 = -1.0 * temperature[i][j][k-3] / 12.0;
			term_2 = 1.0 * temperature[i][j][k-2] / 3.0;
			term_3 = 1.0 * temperature[i][j][k-1] / 2.0;
			term_4 = -5.0 * t_ijk / 3.0;
			term_5 = 11.0 * temperature[i][j][k+1] / 12.0;
		}
		// Stencil size 5. Stencil = (-2, -1, 0, 1, 2) for 2nd order derivative
		else
		{
			term_1 = -1.0 * temperature[i][j][k-2] / 12.0;
			term_2 = 4.0 * temperature[i][j][k-1] / 3.0;
			term_3 = -5.0 * t_ijk / 2.0;
			term_4 = 4.0 * temperature[i][j][k+1] / 3.0;
			term_5 = -1.0 * temperature[i][j][k+2] / 12.0;
		}
		// Get second derivative based on stencile length 7 finite difference method
		dT2_dz2 = (term_1+term_2+term_3+term_4+term_5) / (z_step*z_step);
	}

	// Calculate the second derivative of temperature wrt z at boundaries
	else
	{
		// Top boundary condition
		if (k == 0)
		{
			top_flux = htc*(t_ijk-ambient_temperature) - input_mesh[i][j];
			dT2_dz2 = 2.0*(temperature[i][j][k+1]-t_ijk-(z_step*top_flux/thermal_conductivity))/(z_step*z_step);
		}
		
		// Bottom boundary condition
		else if (k == num_vert_depth-1)
		{
			bottom_flux = htc*(t_ijk-ambient_temperature);
			dT2_dz2 = 2.0*(temperature[i][j][k-1]-t_ijk-(z_step*bottom_flux/thermal_conductivity))/(z_step*z_step);
		}
	}
	
	return dT2_dx2+dT2_dy2+dT2_dz2;
}


/** Calculates the 19-point 3D stencil laplacian
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @param Left and right virtual temperatures from BC
* @param Front and back virtual temperatures from BC
* @param Top and bottom virtual temperatures from BC
* @return 19-point 3D stencil Lapclacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_19(int i, int j, int k, const vector<vector<vector<double>>> &temperature, double*** lr_bc_temps, double*** fb_bc_temps, double*** tb_bc_temps)
{
	double T_000 = temperature[i][j][k];
	double axes[3][3];
	
	bool on_edge = ((i==0) && (j==0));
	on_edge = on_edge || ((i==0) && (j==num_vert_width-1));
	on_edge = on_edge || ((i==num_vert_length-1) && (j==0));
	on_edge = on_edge || ((i==num_vert_length-1) && (j==num_vert_width-1));
	on_edge = on_edge || ((i==0) && (k==0));
	on_edge = on_edge || ((i==0) && (k==num_vert_depth-1));
	on_edge = on_edge || ((i==num_vert_length-1) && (k==0));
	on_edge = on_edge || ((i==num_vert_length-1) && (k==num_vert_depth-1));
	on_edge = on_edge || ((k==0) && (j==0));
	on_edge = on_edge || ((k==0) && (j==num_vert_width-1));
	on_edge = on_edge || ((k==num_vert_depth-1) && (j==0));
	on_edge = on_edge || ((k==num_vert_depth-1) && (j==num_vert_width-1));
	
	// Edges
	if (on_edge)
	{
		return get_laplacian_7(i, j, k, temperature, lr_bc_temps, fb_bc_temps, tb_bc_temps);
	}
	// Left face BC
	else if (i == 0)
	{
		axes[0][0] = (lr_bc_temps[0][j+1][k] - 2.0*T_000 + temperature[i+1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (lr_bc_temps[0][j-1][k] - 2.0*T_000 + temperature[i+1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (lr_bc_temps[0][j][k+1] - 2.0*T_000 + temperature[i+1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (lr_bc_temps[0][j][k-1] - 2.0*T_000 + temperature[i+1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (temperature[i][j-1][k+1] - 2.0*T_000 + temperature[i][j+1][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (temperature[i][j-1][k-1] - 2.0*T_000 + temperature[i][j+1][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (lr_bc_temps[0][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	// Right face BC
	else if (i == num_vert_length-1)
	{		
		axes[0][0] = (temperature[i-1][j+1][k] - 2.0*T_000 + lr_bc_temps[1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (temperature[i-1][j-1][k] - 2.0*T_000 + lr_bc_temps[1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (temperature[i-1][j][k+1] - 2.0*T_000 + lr_bc_temps[1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (temperature[i-1][j][k-1] - 2.0*T_000 + lr_bc_temps[1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (temperature[i][j-1][k+1] - 2.0*T_000 + temperature[i][j+1][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (temperature[i][j-1][k-1] - 2.0*T_000 + temperature[i][j+1][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + lr_bc_temps[1][j][k]) / (x_step*x_step);
	}
	// Front face BC
	else if (j == 0)
	{	
		axes[0][0] = (temperature[i-1][j+1][k] - 2.0*T_000 + fb_bc_temps[0][i+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (fb_bc_temps[0][i-1][k] - 2.0*T_000 + temperature[i+1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (temperature[i-1][j][k+1] - 2.0*T_000 + temperature[i+1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (temperature[i-1][j][k-1] - 2.0*T_000 + temperature[i+1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (fb_bc_temps[0][i][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (fb_bc_temps[0][i][k+1] - 2.0*T_000 + temperature[i][j+1][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (fb_bc_temps[0][i][k-1] - 2.0*T_000 + temperature[i][j+1][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	// Back face BC
	else if (j == num_vert_width-1)
	{	
		axes[0][0] = (fb_bc_temps[1][i-1][k] - 2.0*T_000 + temperature[i+1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (temperature[i-1][j-1][k] - 2.0*T_000 + fb_bc_temps[1][i+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (temperature[i-1][j][k+1] - 2.0*T_000 + temperature[i+1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (temperature[i-1][j][k-1] - 2.0*T_000 + temperature[i+1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + fb_bc_temps[1][i][k]) / (y_step*y_step);
		
		axes[2][0] = (temperature[i][j-1][k+1] - 2.0*T_000 + fb_bc_temps[1][i][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (temperature[i][j-1][k-1] - 2.0*T_000 + fb_bc_temps[1][i][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	// Top face BC
	else if (k == 0)
	{
		axes[0][0] = (temperature[i-1][j+1][k] - 2.0*T_000 + temperature[i+1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (temperature[i-1][j-1][k] - 2.0*T_000 + temperature[i+1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (tb_bc_temps[0][i][j] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (temperature[i-1][j][k+1] - 2.0*T_000 + tb_bc_temps[0][i+1][j]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (tb_bc_temps[0][i-1][j] - 2.0*T_000 + temperature[i+1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (temperature[i][j-1][k+1] - 2.0*T_000 + tb_bc_temps[0][i][j+1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (tb_bc_temps[0][i][j-1] - 2.0*T_000 + temperature[i][j+1][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	// Bottom face BC
	else if (k == num_vert_depth-1)
	{
		axes[0][0] = (temperature[i-1][j+1][k] - 2.0*T_000 + temperature[i+1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (temperature[i-1][j-1][k] - 2.0*T_000 + temperature[i+1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + tb_bc_temps[1][i][j]) / (z_step*z_step);
		
		axes[1][0] = (tb_bc_temps[1][i-1][j] - 2.0*T_000 + temperature[i+1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (temperature[i-1][j][k-1] - 2.0*T_000 + tb_bc_temps[1][i+1][j]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (tb_bc_temps[1][i][j-1] - 2.0*T_000 + temperature[i][j+1][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (temperature[i][j-1][k-1] - 2.0*T_000 + tb_bc_temps[1][i][j+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	// Bulk material
	else
	{
		axes[0][0] = (temperature[i-1][j+1][k] - 2.0*T_000 + temperature[i+1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (temperature[i-1][j-1][k] - 2.0*T_000 + temperature[i+1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (temperature[i-1][j][k+1] - 2.0*T_000 + temperature[i+1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (temperature[i-1][j][k-1] - 2.0*T_000 + temperature[i+1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (temperature[i][j-1][k+1] - 2.0*T_000 + temperature[i][j+1][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (temperature[i][j-1][k-1] - 2.0*T_000 + temperature[i][j+1][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	
	// Calculate laplacians for all 3 bases
	double laplacian_1 = axes[0][0] + axes[0][1] + axes[0][2];
	double laplacian_2 = axes[1][0] + axes[1][1] + axes[1][2];
	double laplacian_3 = axes[2][0] + axes[2][1] + axes[2][2];
	
	return ((laplacian_1 + laplacian_2 + laplacian_3) / (3.0));
}

/** Calculates the 19-point 3D stencil laplacian
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @param Left and right virtual temperatures from BC
* @param Front and back virtual temperatures from BC
* @param Top and bottom virtual temperatures from BC
* @return 19-point 3D stencil Lapclacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_19_norm(int i, int j, int k, const vector<vector<vector<double>>> &temperature, double*** lr_bc_temps, double*** fb_bc_temps, double*** tb_bc_temps)
{
	double T_000 = temperature[i][j][k];
	double axes[3][3];
	
	bool on_edge = ((i==0) && (j==0));
	on_edge = on_edge || ((i==0) && (j==num_vert_width-1));
	on_edge = on_edge || ((i==num_vert_length-1) && (j==0));
	on_edge = on_edge || ((i==num_vert_length-1) && (j==num_vert_width-1));
	on_edge = on_edge || ((i==0) && (k==0));
	on_edge = on_edge || ((i==0) && (k==num_vert_depth-1));
	on_edge = on_edge || ((i==num_vert_length-1) && (k==0));
	on_edge = on_edge || ((i==num_vert_length-1) && (k==num_vert_depth-1));
	on_edge = on_edge || ((k==0) && (j==0));
	on_edge = on_edge || ((k==0) && (j==num_vert_width-1));
	on_edge = on_edge || ((k==num_vert_depth-1) && (j==0));
	on_edge = on_edge || ((k==num_vert_depth-1) && (j==num_vert_width-1));
	
	// Edges
	if (on_edge)
	{
		return get_laplacian_7(i, j, k, temperature, lr_bc_temps, fb_bc_temps, tb_bc_temps);
	}
	// Left face BC
	else if (i == 0)
	{
		axes[0][0] = (lr_bc_temps[0][j+1][k] - 2.0*T_000 + temperature[i+1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (lr_bc_temps[0][j-1][k] - 2.0*T_000 + temperature[i+1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (lr_bc_temps[0][j][k+1] - 2.0*T_000 + temperature[i+1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (lr_bc_temps[0][j][k-1] - 2.0*T_000 + temperature[i+1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (temperature[i][j-1][k+1] - 2.0*T_000 + temperature[i][j+1][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (temperature[i][j-1][k-1] - 2.0*T_000 + temperature[i][j+1][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (lr_bc_temps[0][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	// Right face BC
	else if (i == num_vert_length-1)
	{		
		axes[0][0] = (temperature[i-1][j+1][k] - 2.0*T_000 + lr_bc_temps[1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (temperature[i-1][j-1][k] - 2.0*T_000 + lr_bc_temps[1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (temperature[i-1][j][k+1] - 2.0*T_000 + lr_bc_temps[1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (temperature[i-1][j][k-1] - 2.0*T_000 + lr_bc_temps[1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (temperature[i][j-1][k+1] - 2.0*T_000 + temperature[i][j+1][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (temperature[i][j-1][k-1] - 2.0*T_000 + temperature[i][j+1][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + lr_bc_temps[1][j][k]) / (x_step*x_step);
	}
	// Front face BC
	else if (j == 0)
	{	
		axes[0][0] = (temperature[i-1][j+1][k] - 2.0*T_000 + fb_bc_temps[0][i+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (fb_bc_temps[0][i-1][k] - 2.0*T_000 + temperature[i+1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (temperature[i-1][j][k+1] - 2.0*T_000 + temperature[i+1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (temperature[i-1][j][k-1] - 2.0*T_000 + temperature[i+1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (fb_bc_temps[0][i][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (fb_bc_temps[0][i][k+1] - 2.0*T_000 + temperature[i][j+1][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (fb_bc_temps[0][i][k-1] - 2.0*T_000 + temperature[i][j+1][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	// Back face BC
	else if (j == num_vert_width-1)
	{	
		axes[0][0] = (fb_bc_temps[1][i-1][k] - 2.0*T_000 + temperature[i+1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (temperature[i-1][j-1][k] - 2.0*T_000 + fb_bc_temps[1][i+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (temperature[i-1][j][k+1] - 2.0*T_000 + temperature[i+1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (temperature[i-1][j][k-1] - 2.0*T_000 + temperature[i+1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + fb_bc_temps[1][i][k]) / (y_step*y_step);
		
		axes[2][0] = (temperature[i][j-1][k+1] - 2.0*T_000 + fb_bc_temps[1][i][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (temperature[i][j-1][k-1] - 2.0*T_000 + fb_bc_temps[1][i][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	// Top face BC
	else if (k == 0)
	{
		axes[0][0] = (temperature[i-1][j+1][k] - 2.0*T_000 + temperature[i+1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (temperature[i-1][j-1][k] - 2.0*T_000 + temperature[i+1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (tb_bc_temps[0][i][j] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (temperature[i-1][j][k+1] - 2.0*T_000 + tb_bc_temps[0][i+1][j]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (tb_bc_temps[0][i-1][j] - 2.0*T_000 + temperature[i+1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (temperature[i][j-1][k+1] - 2.0*T_000 + tb_bc_temps[0][i][j+1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (tb_bc_temps[0][i][j-1] - 2.0*T_000 + temperature[i][j+1][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	// Bottom face BC
	else if (k == num_vert_depth-1)
	{
		axes[0][0] = (temperature[i-1][j+1][k] - 2.0*T_000 + temperature[i+1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (temperature[i-1][j-1][k] - 2.0*T_000 + temperature[i+1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + tb_bc_temps[1][i][j]) / (z_step*z_step);
		
		axes[1][0] = (tb_bc_temps[1][i-1][j] - 2.0*T_000 + temperature[i+1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (temperature[i-1][j][k-1] - 2.0*T_000 + tb_bc_temps[1][i+1][j]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (tb_bc_temps[1][i][j-1] - 2.0*T_000 + temperature[i][j+1][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (temperature[i][j-1][k-1] - 2.0*T_000 + tb_bc_temps[1][i][j+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	// Bulk material
	else
	{
		axes[0][0] = (temperature[i-1][j+1][k] - 2.0*T_000 + temperature[i+1][j-1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][1] = (temperature[i-1][j-1][k] - 2.0*T_000 + temperature[i+1][j+1][k]) / (x_step*x_step + y_step*y_step);
		axes[0][2] = (temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1]) / (z_step*z_step);
		
		axes[1][0] = (temperature[i-1][j][k+1] - 2.0*T_000 + temperature[i+1][j][k-1]) / (x_step*x_step + z_step*z_step);
		axes[1][1] = (temperature[i-1][j][k-1] - 2.0*T_000 + temperature[i+1][j][k+1]) / (x_step*x_step + z_step*z_step);
		axes[1][2] = (temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k]) / (y_step*y_step);
		
		axes[2][0] = (temperature[i][j-1][k+1] - 2.0*T_000 + temperature[i][j+1][k-1]) / (y_step*y_step + z_step*z_step);
		axes[2][1] = (temperature[i][j-1][k-1] - 2.0*T_000 + temperature[i][j+1][k+1]) / (y_step*y_step + z_step*z_step);
		axes[2][2] = (temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k]) / (x_step*x_step);
	}
	
	// Calculate laplacians for all 3 bases
	double laplacian_1 = axes[0][0] + axes[0][1] + axes[0][2];
	double laplacian_2 = axes[1][0] + axes[1][1] + axes[1][2];
	double laplacian_3 = axes[2][0] + axes[2][1] + axes[2][2];
	
	return (((x_step / z_step) * laplacian_1 + (x_step / y_step) * laplacian_2 + laplacian_3) / (1.0 + (x_step / z_step) + (x_step / y_step)));
}

/** Calculates the 27-point 3D stencil laplacian
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @param Left and right virtual temperatures from BC
* @param Front and back virtual temperatures from BC
* @param Top and bottom virtual temperatures from BC
* @return 27-point 3D stencil Lapclacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_27(int i, int j, int k, const vector<vector<vector<double>>> &temperature, double*** lr_bc_temps, double*** fb_bc_temps, double*** tb_bc_temps)
{
	double T_000 = temperature[i][j][k];
	double laplacian = 0.0;
	
	bool on_edge = ((i==0) && (j==0));
	on_edge = on_edge || ((i==0) && (j==num_vert_width-1));
	on_edge = on_edge || ((i==num_vert_length-1) && (j==0));
	on_edge = on_edge || ((i==num_vert_length-1) && (j==num_vert_width-1));
	on_edge = on_edge || ((i==0) && (k==0));
	on_edge = on_edge || ((i==0) && (k==num_vert_depth-1));
	on_edge = on_edge || ((i==num_vert_length-1) && (k==0));
	on_edge = on_edge || ((i==num_vert_length-1) && (k==num_vert_depth-1));
	on_edge = on_edge || ((k==0) && (j==0));
	on_edge = on_edge || ((k==0) && (j==num_vert_width-1));
	on_edge = on_edge || ((k==num_vert_depth-1) && (j==0));
	on_edge = on_edge || ((k==num_vert_depth-1) && (j==num_vert_width-1));
	
	// Edges
	if (on_edge)
	{
		return get_laplacian_7(i, j, k, temperature, lr_bc_temps, fb_bc_temps, tb_bc_temps);
	}
	// Left face BC
	else if (i == 0)
	{
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (p == -1)
				{
					laplacian += (lr_bc_temps[0][j+q][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
			}
		}
	}
	// Right face BC
	else if (i == num_vert_length-1)
	{	
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (p == 1)
				{
					laplacian += (lr_bc_temps[1][j+q][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
			}
		}
	}
	// Front face BC
	else if (j == 0)
	{
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (q == -1)
				{
					laplacian += (fb_bc_temps[0][i+p][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
			}
		}		
	}
	// Back face BC
	else if (j == num_vert_width-1)
	{	
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (q == 1)
				{
					laplacian += (fb_bc_temps[1][i+p][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
			}
		}
	}
	// Top face BC
	else if (k == 0)
	{
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (r == -1)
				{
					laplacian += (tb_bc_temps[0][i+p][j+q] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
			}
		}
	}
	// Bottom face BC
	else if (k == num_vert_depth-1)
	{
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (r == 1)
				{
					laplacian += (tb_bc_temps[1][i+p][j+q] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
				}
			}
		}
	}
	// Bulk material
	else
	{
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p)*x_step*x_step + abs((double)q)*y_step*y_step + abs((double)r)*z_step*z_step);
			}
		}
	}
	
	// Normalize the calculated laplacian sum
	laplacian = 3.0 * laplacian / 13.0;
	
	return laplacian;
}

/** Calculates the 27-point 3D stencil laplacian
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @param Left and right virtual temperatures from BC
* @param Front and back virtual temperatures from BC
* @param Top and bottom virtual temperatures from BC
* @return 27-point 3D stencil Lapclacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_27_norm(int i, int j, int k, const vector<vector<vector<double>>> &temperature, double*** lr_bc_temps, double*** fb_bc_temps, double*** tb_bc_temps)
{
	double T_000 = temperature[i][j][k];
	double laplacian = 0.0;
	
	bool on_edge = ((i==0) && (j==0));
	on_edge = on_edge || ((i==0) && (j==num_vert_width-1));
	on_edge = on_edge || ((i==num_vert_length-1) && (j==0));
	on_edge = on_edge || ((i==num_vert_length-1) && (j==num_vert_width-1));
	on_edge = on_edge || ((i==0) && (k==0));
	on_edge = on_edge || ((i==0) && (k==num_vert_depth-1));
	on_edge = on_edge || ((i==num_vert_length-1) && (k==0));
	on_edge = on_edge || ((i==num_vert_length-1) && (k==num_vert_depth-1));
	on_edge = on_edge || ((k==0) && (j==0));
	on_edge = on_edge || ((k==0) && (j==num_vert_width-1));
	on_edge = on_edge || ((k==num_vert_depth-1) && (j==0));
	on_edge = on_edge || ((k==num_vert_depth-1) && (j==num_vert_width-1));
	
	// Edges
	if (on_edge)
	{
		return get_laplacian_7(i, j, k, temperature, lr_bc_temps, fb_bc_temps, tb_bc_temps);
	}
	// Left face BC
	else if (i == 0)
	{
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (p == -1)
				{
					laplacian += (lr_bc_temps[0][j+q][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
			}
		}
	}
	// Right face BC
	else if (i == num_vert_length-1)
	{	
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (p == 1)
				{
					laplacian += (lr_bc_temps[1][j+q][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
			}
		}
	}
	// Front face BC
	else if (j == 0)
	{
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (q == -1)
				{
					laplacian += (fb_bc_temps[0][i+p][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
			}
		}		
	}
	// Back face BC
	else if (j == num_vert_width-1)
	{	
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (q == 1)
				{
					laplacian += (fb_bc_temps[1][i+p][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
			}
		}
	}
	// Top face BC
	else if (k == 0)
	{
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (r == -1)
				{
					laplacian += (tb_bc_temps[0][i+p][j+q] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
			}
		}
	}
	// Bottom face BC
	else if (k == num_vert_depth-1)
	{
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				if (r == 1)
				{
					laplacian += (tb_bc_temps[1][i+p][j+q] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
				else
				{
					laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
				}
			}
		}
	}
	// Bulk material
	else
	{
		for(int p = -1; p <= 1; p++)
		for(int q = -1; q <= 1; q++)
		for(int r = -1; r <= 1; r++)
		{
			if ( (p != 0) && (q != 0) && (r != 0) )
			{
				laplacian += (temperature[i+p][j+q][k+r] - T_000) / (abs((double)p) + abs((double)q) + abs((double)r));
			}
		}
	}
	
	// Normalize the calculated laplacian sum
	double avg_step = (x_step + y_step + z_step) / 3.0;
	laplacian = 3.0 * laplacian / (13.0 * avg_step*avg_step);
	
	return laplacian;
}