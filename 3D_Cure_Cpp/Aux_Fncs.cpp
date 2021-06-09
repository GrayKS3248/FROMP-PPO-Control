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

// Laplacian calculation consts for 4rd order 7-stencil
double laplacian_consts_4th[7][9] = { { 363.0/560.0, 8.0/315.0, -83.0/20.0, 153.0/20.0, -529.0/72.0, 47.0/10.0, -39.0/20.0, 599.0/1260.0, -29.0/560.0 }, 
				      { -29.0/560.0, 39.0/35.0, -331.0/180.0, 1.0/5.0, 9.0/8.0, -37.0/45.0, 7.0/20.0, -3.0/35.0, 47.0/5040.0 }, 
				      { 47.0/5040.0, -19.0/140.0, 29.0/20.0, -118.0/45.0, 11.0/8.0, -1.0/20.0, -7.0/180.0, 1.0/70.0, -1.0/560.0 }, 
				      { -1.0/560.0, 8.0/315.0, -1.0/5.0, 8.0/5.0, -205.0/72.0, 8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0 }, 
				      { -1.0/560.0, 1.0/70.0, -7.0/180.0, -1.0/20.0, 11.0/8.0, -118.0/45.0, 29.0/20.0, -19.0/140.0, 47.0/5040.0 }, 
				      { 47.0/5040.0, -3.0/35.0, 7.0/20.0, -37.0/45.0, 9.0/8.0, 1.0/5.0, -331.0/180.0, 39.0/35.0, -29.0/560.0 }, 
				      { -29.0/560.0, 599.0/1260.0, -39.0/20.0, 47.0/10.0, -529.0/72.0, 153.0/20.0, -83.0/20.0, 8.0/315.0, 363.0/560.0 } };
				  
// Laplacian calculation consts for 3rd order 7-stencil
double laplacian_consts_3rd[5][7] = { { 137.0/180.0, -49.0/60.0, -17.0/12.0, 47.0/18.0, -19.0/12.0,  31.0/60.0, -13.0/180.0 }, 
				  { -13.0/180.0,  19.0/15.0,  -7.0/3.0,  10.0/9.0,    1.0/12.0,  -1.0/15.0,   1.0/90.0 }, 
				  {   1.0/90.0,   -3.0/20.0,   3.0/2.0, -49.0/18.0,   3.0/2.0,   -3.0/20.0,   1.0/90.0 },
				  {   1.0/90.0,   -1.0/15.0,   1.0/12.0, 10.0/9.0,   -7.0/3.0,   19.0/15.0, -13.0/180.0 }, 
				  { -13.0/180.0,  31.0/60.0, -19.0/12.0, 47.0/18.0, -17.0/12.0, -49.0/60.0, 137.0/180.0 } };
				  
// Laplacian calculation consts for 2nd order 7-stencil
double laplacian_consts_2nd[3][5] = { { 11.0/12.0, -5.0/3.0,  1.0/2.0,  1.0/3.0, -1.0/12.0 }, 
				      { -1.0/12.0,  4.0/3.0, -5.0/2.0,  4.0/3.0, -1.0/12.0 }, 
				      { -1.0/12.0,  1.0/3.0,  1.0/2.0, -5.0/3.0, 11.0/12.0 } };

/** Calculates the 7-point stencil, 1st order, 3D laplacian
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @return 7-point stencil, 1st order, 3D laplacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_7_1st(int i, int j, int k, const vector<vector<vector<double>>> &temperature)
{
	double T_000 = temperature[i][j][k];
	double d2t_dx2 = 0.0;
	double d2t_dy2 = 0.0;
	double d2t_dz2 = 0.0;
	
	// Right face BC
	if (i==0)
	{
		d2t_dx2 = lr_bc_temps[0][j][k] - 2.0*T_000 + temperature[i+1][j][k];
	}
	// Left face BC
	else if(i==num_vert_length-1)
	{
		d2t_dx2 = temperature[i-1][j][k] - 2.0*T_000 + lr_bc_temps[1][j][k];
	}
	// Bulk material
	else
	{
		d2t_dx2 = temperature[i-1][j][k] - 2.0*T_000 + temperature[i+1][j][k];
	}
	d2t_dx2 = d2t_dx2 / (x_step*x_step);
	
	
	// Front face BC
	if (j==0)
	{
		d2t_dy2 = fb_bc_temps[0][i][k] - 2.0*T_000 + temperature[i][j+1][k];
	}
	// Back face BC
	else if(j==num_vert_width-1)
	{
		d2t_dy2 = temperature[i][j-1][k] - 2.0*T_000 + fb_bc_temps[1][i][k];
	}
	// Bulk material
	else
	{
		d2t_dy2 = temperature[i][j-1][k] - 2.0*T_000 + temperature[i][j+1][k];
	}
	d2t_dy2 = d2t_dy2 / (y_step*y_step);
	
	
	// Top face BC
	if (k==0)
	{
		d2t_dz2 = tb_bc_temps[0][i][j] - 2.0*T_000 + temperature[i][j][k+1];
	}
	// Bottom face BC
	else if(k==num_vert_depth-1)
	{
		d2t_dz2 = temperature[i][j][k-1] - 2.0*T_000 + tb_bc_temps[1][i][j];
	}
	// Bulk material
	else
	{
		d2t_dz2 = temperature[i][j][k-1] - 2.0*T_000 + temperature[i][j][k+1];
	}
	d2t_dz2 = d2t_dz2 / (z_step*z_step);
	
	return d2t_dx2 + d2t_dy2 + d2t_dz2;
}

/** Calculates the 7-point stencil, 2nd order, 3D laplacian
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @return 7-point stencil, 2nd order, 3D laplacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_7_2nd(int i, int j, int k, const vector<vector<vector<double>>> &temperature)
{
	double T_000 = temperature[i][j][k];
	double d2t_dx2 = 0.0;
	double d2t_dy2 = 0.0;
	double d2t_dz2 = 0.0;
	
	// Right face BC
	if (i==0)
	{
		d2t_dx2 = lr_bc_temps[0][j][k] - 2.0*T_000 + temperature[i+1][j][k];
	}
	// Left face BC
	else if(i==num_vert_length-1)
	{
		d2t_dx2 = temperature[i-1][j][k] - 2.0*T_000 + lr_bc_temps[1][j][k];
	}
	// Bulk material
	else
	{
		int start_p = -2;
		start_p = (i==1) ? -1 : start_p;
		start_p = (i==num_vert_length-2) ? -3 : start_p;
		for (int p = start_p; p < start_p + 5; p++)
		{
			d2t_dx2 += laplacian_consts_2nd[abs(start_p)-1][p-start_p] * temperature[i+p][j][k];
		}
	}
	d2t_dx2 = d2t_dx2 / (x_step*x_step);
	
	
	// Front face BC
	if (j==0)
	{
		d2t_dy2 = fb_bc_temps[0][i][k] - 2.0*T_000 + temperature[i][j+1][k];
	}
	// Back face BC
	else if(j==num_vert_width-1)
	{
		d2t_dy2 = temperature[i][j-1][k] - 2.0*T_000 + fb_bc_temps[1][i][k];
	}
	// Bulk material
	else
	{
		int start_q = -2;
		start_q = (j==1) ? -1 : start_q;
		start_q = (j==num_vert_width-2) ? -3 : start_q;
		for (int q = start_q; q < start_q + 5; q++)
		{
			d2t_dy2 += laplacian_consts_2nd[abs(start_q)-1][q-start_q] * temperature[i][j+q][k];
		}
	}
	d2t_dy2 = d2t_dy2 / (y_step*y_step);
	
	
	// Top face BC
	if (k==0)
	{
		d2t_dz2 = tb_bc_temps[0][i][j] - 2.0*T_000 + temperature[i][j][k+1];
	}
	// Bottom face BC
	else if(k==num_vert_depth-1)
	{
		d2t_dz2 = temperature[i][j][k-1] - 2.0*T_000 + tb_bc_temps[1][i][j];
	}
	// Bulk material
	else
	{
		int start_r = -2;
		start_r = (k==1) ? -1 : start_r;
		start_r = (k==num_vert_depth-2) ? -3 : start_r;
		for (int r = start_r; r < start_r + 5; r++)
		{
			d2t_dz2 += laplacian_consts_2nd[abs(start_r)-1][r-start_r] * temperature[i][j][k+r];
		}
	}
	d2t_dz2 = d2t_dz2 / (z_step*z_step);
	
	return d2t_dx2 + d2t_dy2 + d2t_dz2;
}

/** Calculates the 7-point stencil, 3rd order, 3D laplacian
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @return 7-point stencil, 3rd order, 3D laplacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_7_3rd(int i, int j, int k, const vector<vector<vector<double>>> &temperature)
{
	double T_000 = temperature[i][j][k];
	double d2t_dx2 = 0.0;
	double d2t_dy2 = 0.0;
	double d2t_dz2 = 0.0;
	
	// Right face BC
	if (i==0)
	{
		d2t_dx2 = lr_bc_temps[0][j][k] - 2.0*T_000 + temperature[i+1][j][k];
	}
	// Left face BC
	else if(i==num_vert_length-1)
	{
		d2t_dx2 = temperature[i-1][j][k] - 2.0*T_000 + lr_bc_temps[1][j][k];
	}
	// Bulk material
	else
	{
		int start_p = -3;
		start_p = (i==1) ? -1 : start_p;
		start_p = (i==2) ? -2 : start_p;
		start_p = (i==num_vert_length-3) ? -4 : start_p;
		start_p = (i==num_vert_length-2) ? -5 : start_p;
		for (int p = start_p; p < start_p + 7; p++)
		{
			d2t_dx2 += laplacian_consts_3rd[abs(start_p)-1][p-start_p] * temperature[i+p][j][k];
		}
	}
	d2t_dx2 = d2t_dx2 / (x_step*x_step);
	
	
	// Front face BC
	if (j==0)
	{
		d2t_dy2 = fb_bc_temps[0][i][k] - 2.0*T_000 + temperature[i][j+1][k];
	}
	// Back face BC
	else if(j==num_vert_width-1)
	{
		d2t_dy2 = temperature[i][j-1][k] - 2.0*T_000 + fb_bc_temps[1][i][k];
	}
	// Bulk material
	else
	{
		int start_q = -3;
		start_q = (j==1) ? -1 : start_q;
		start_q = (j==2) ? -2 : start_q;
		start_q = (j==num_vert_width-3) ? -4 : start_q;
		start_q = (j==num_vert_width-2) ? -5 : start_q;
		for (int q = start_q; q < start_q + 7; q++)
		{
			d2t_dy2 += laplacian_consts_3rd[abs(start_q)-1][q-start_q] * temperature[i][j+q][k];
		}
	}
	d2t_dy2 = d2t_dy2 / (y_step*y_step);
	
	
	// Top face BC
	if (k==0)
	{
		d2t_dz2 = tb_bc_temps[0][i][j] - 2.0*T_000 + temperature[i][j][k+1];
	}
	// Bottom face BC
	else if(k==num_vert_depth-1)
	{
		d2t_dz2 = temperature[i][j][k-1] - 2.0*T_000 + tb_bc_temps[1][i][j];
	}
	// Bulk material
	else
	{
		int start_r = -3;
		start_r = (k==1) ? -1 : start_r;
		start_r = (k==2) ? -2 : start_r;
		start_r = (k==num_vert_depth-3) ? -4 : start_r;
		start_r = (k==num_vert_depth-2) ? -5 : start_r;
		for (int r = start_r; r < start_r + 7; r++)
		{
			d2t_dz2 += laplacian_consts_3rd[abs(start_r)-1][r-start_r] * temperature[i][j][k+r];
		}
	}
	d2t_dz2 = d2t_dz2 / (z_step*z_step);
	
	return d2t_dx2 + d2t_dy2 + d2t_dz2;
}

/** Calculates the 7-point stencil, 4th order, 3D laplacian
* @param i index at which the Laplacian is calculated
* @param j index at which the Laplacian is calculated
* @param k index at which the Laplacian is calculated
* @param Temperature field
* @return 7-point stencil, 4th order, 3D laplacian at (i,j,k)
*/
double Finite_Element_Solver::get_laplacian_7_4th(int i, int j, int k, const vector<vector<vector<double>>> &temperature)
{
	double T_000 = temperature[i][j][k];
	double d2t_dx2 = 0.0;
	double d2t_dy2 = 0.0;
	double d2t_dz2 = 0.0;
	
	// Right face BC
	if (i==0)
	{
		d2t_dx2 = lr_bc_temps[0][j][k] - 2.0*T_000 + temperature[i+1][j][k];
	}
	// Left face BC
	else if(i==num_vert_length-1)
	{
		d2t_dx2 = temperature[i-1][j][k] - 2.0*T_000 + lr_bc_temps[1][j][k];
	}
	// Bulk material
	else
	{
		int start_p = -4;
		start_p = (i==1) ? -1 : start_p;
		start_p = (i==2) ? -2 : start_p;
		start_p = (i==3) ? -3 : start_p;
		start_p = (i==num_vert_length-4) ? -5 : start_p;
		start_p = (i==num_vert_length-3) ? -6 : start_p;
		start_p = (i==num_vert_length-2) ? -7 : start_p;
		for (int p = start_p; p < start_p + 9; p++)
		{
			d2t_dx2 += laplacian_consts_4th[abs(start_p)-1][p-start_p] * temperature[i+p][j][k];
		}
	}
	d2t_dx2 = d2t_dx2 / (x_step*x_step);
	
	
	// Front face BC
	if (j==0)
	{
		d2t_dy2 = fb_bc_temps[0][i][k] - 2.0*T_000 + temperature[i][j+1][k];
	}
	// Back face BC
	else if(j==num_vert_width-1)
	{
		d2t_dy2 = temperature[i][j-1][k] - 2.0*T_000 + fb_bc_temps[1][i][k];
	}
	// Bulk material
	else
	{
		int start_q = -4;
		start_q = (j==1) ? -1 : start_q;
		start_q = (j==2) ? -2 : start_q;
		start_q = (j==3) ? -3 : start_q;
		start_q = (j==num_vert_width-4) ? -5 : start_q;
		start_q = (j==num_vert_width-3) ? -6 : start_q;
		start_q = (j==num_vert_width-2) ? -7 : start_q;
		for (int q = start_q; q < start_q + 9; q++)
		{
			d2t_dy2 += laplacian_consts_4th[abs(start_q)-1][q-start_q] * temperature[i][j+q][k];
		}
	}
	d2t_dy2 = d2t_dy2 / (y_step*y_step);
	
	
	// Top face BC
	if (k==0)
	{
		d2t_dz2 = tb_bc_temps[0][i][j] - 2.0*T_000 + temperature[i][j][k+1];
	}
	// Bottom face BC
	else if(k==num_vert_depth-1)
	{
		d2t_dz2 = temperature[i][j][k-1] - 2.0*T_000 + tb_bc_temps[1][i][j];
	}
	// Bulk material
	else
	{
		int start_r = -4;
		start_r = (k==1) ? -1 : start_r;
		start_r = (k==2) ? -2 : start_r;
		start_r = (k==3) ? -3 : start_r;
		start_r = (k==num_vert_depth-4) ? -5 : start_r;
		start_r = (k==num_vert_depth-3) ? -6 : start_r;
		start_r = (k==num_vert_depth-2) ? -7 : start_r;
		for (int r = start_r; r < start_r + 9; r++)
		{
			d2t_dz2 += laplacian_consts_4th[abs(start_r)-1][r-start_r] * temperature[i][j][k+r];
		}
	}
	d2t_dz2 = d2t_dz2 / (z_step*z_step);
	
	return d2t_dx2 + d2t_dy2 + d2t_dz2;
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