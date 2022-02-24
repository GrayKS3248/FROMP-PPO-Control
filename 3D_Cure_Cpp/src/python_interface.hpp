#pragma once
#include <vector>
#include <Python.h>
#include <math.h>
#include "Config_Handler.hpp"

using namespace std;

//******************************************************************** PYTHON API CONVERSION FUNCTIONS ********************************************************************//
template <typename T>
PyObject* get_1D_list(T arr);
template <typename T>
PyObject* get_2D_list(T arr);
template <typename T>
PyObject* get_3D_list(T arr);
vector<double> get_1D_vector(PyObject* list);
vector<vector<double>> get_2D_vector(PyObject* list);

//******************************************************************** PYTHON API INITIALIZATION FUNCTIONS ********************************************************************//
PyObject* init_agent(int num_addtional_inputs, int num_outputs, string load_path);
PyObject* init_save_render_plot();


//******************************************************************** PYTHON API CALL METHOD FUNCTIONS ********************************************************************//
int store_training_curves(PyObject* save_render_plot, vector<double> r_per_episode, vector<double> value_error);
int store_lr_curves(PyObject* save_render_plot, vector<double> actor_lr, vector<double> critic_lr);
int store_stdev_history(PyObject* save_render_plot, vector<double> x_stdev, vector<double> y_stdev, vector<double> mag_stdev);
int store_input_history(PyObject* save_render_plot, vector<double> input_location_x, vector<double> input_location_y, vector<double> input_percent, vector<double> trigger_power, vector<double> source_power);
int store_field_history(PyObject* save_render_plot, vector<vector<vector<double>>> temperature_field, vector<vector<vector<double>>> cure_field, vector<vector<vector<double>>> fine_temperature_field, vector<vector<vector<double>>> fine_cure_field, vector<vector<double>> fine_mesh_loc);
int store_front_history(PyObject* save_render_plot, vector<vector<vector<double>>> front_curve, vector<vector<double>> front_fit, vector<double> front_velocity, vector<double> front_temperature, vector<double> front_shape_param);
int store_target_and_time(PyObject* save_render_plot, vector<double> target, vector<double> time, vector<vector<double>> reward);
int store_top_mesh(PyObject* save_render_plot, vector<vector<double>> mesh_x_z0, vector<vector<double>> mesh_y_z0);
int store_input_params(PyObject* save_render_plot, double max_input_mag, double exp_const);
int store_options(PyObject* save_render_plot, bool control_speed, string configs_string);
int store_monomer_properties(PyObject* save_render_plot, double specific_heat, double density, double adiabatic_rxn_temp);
int store_domain_properties(PyObject* save_render_plot, double volume, double surface_area);
int store_boundary_conditions(PyObject* save_render_plot, double heat_transfer_coeff, double ambient_temp, double initial_temperature);


//******************************************************************** PYTHON API SAVE, PLOT, RENDER FUNCTIONS ********************************************************************//
int save_agent_results(PyObject* save_render_plot, PyObject* agent, bool render);
int save_results(PyObject* save_render_plot, bool render);