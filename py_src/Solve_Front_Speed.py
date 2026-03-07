# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:12:49 2022

@author: GKSch
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Front speed solution tolerance
tol = 0.001
f_start = 1.0
f_end = 1.25

# Initial conditions
T0 = 24.0       # Celcius
a0 = 0.07    # Decimal Percent

# Boundary conditions
dTf = -1e-5
Tf = 1.0
af = 1.0

# Options
graph = True

# DCPD Monomer with GC2 physical parameters
R = 8.314       # Joules / Mol * Kelvin
k = 0.152;      # Watts / Meter * Kelvin
p = 980.0;      # Kilograms / Meter ^ 3
hr = 350000.0;  # Joules / Kilogram
cp = 1600.0;    # Joules / Kilogram * Kelvin
A = 8.55e15;    # 1 / Seconds
E = 110750.0;   # Joules / Mol
n = 1.72;       # Unitless
m = 0.77;       # Unitless
C = 14.48;      # Unitless
ac = 0.41;      # Decimal Percent


# Define temperature dependent reation dynamics
def f(temp):
    return A * np.exp(-E/(R*temp))

# Define cure dependent reaction dynamics
def g(cure):
    
    if cure > 0.0 and cure < 1.0:
        return (cure**m*(1-cure)**n) / (1.0 + np.exp(C*(cure-ac)))
    else:
        return 0.0

# Define system of ODEs for traveling wave solution to reaction-diffusion equation
def ode(x,phi,speed):
    temp_derivative, temp, cure = phi
    
    temp_second_derivative = -(speed*p*cp*temp_derivative/k) - (hr*p/k)*f(temp)*g(cure)
    cure_derivative = -(1.0/speed)*f(temp)*g(cure)
    
    return [temp_second_derivative, temp_derivative, cure_derivative]

# Solves the ODE around the front given a front speed guess
def solve_ode(speed):
    # Calculate the boundary temperature
    T_max = T0+273.15 + (1.0 - a0) * hr / cp

    # Solve the IVP
    sol = solve_ivp(ode, [0.0, 100.0], [dTf, 0.99*T_max*Tf, 0.99*af], args=[speed/1000.0], dense_output=True, method='Radau')
    
    # Determine front location from global solution
    space = np.linspace(sol.sol.t_min,sol.sol.t_max,int(1e6))
    front_center = space[int(np.round(0.50*(np.argmax(abs(np.diff(sol.sol(space)[2,:])))+0.5 + np.argmax(abs(sol.sol(space)[0,:])))))]
    
    # Determine front width defined by the radius of front center to location of 0.99 cure
    front_width = front_center - space[np.argmin(abs(sol.sol(space)[2,:]-0.99))]
    
    # Calculate solution around front
    front_space = np.linspace(front_center-front_width*0.5,front_center+front_width*0.5,1000)
    front_sol = sol.sol(front_space)
    return front_sol, front_space, T_max

# Define function that calculates terminal cure and front error given initial conditions, system constants, and front speed guess
def get_err(vel):

    # Get the solution curves    
    front_sol, _, T_max = solve_ode(vel)
    
    # Calculate error
    temp_err = 100.0*(((front_sol[1,:]-(T0+273.15))/(T_max-(T0+273.15)))[-1])
    cure_err = 100.0*(front_sol[2,-1]-a0)
    err = (abs(temp_err) + abs(cure_err))
    
    # Return calculated error
    return err

# Solve the minimization problem to get estimated front speed
front_speed = minimize_scalar(get_err, bounds=(f_start, f_end), method='bounded', options={'xatol':tol}).x

# Get the solution with the optimal front speed
front_sol, front_space, T_max = solve_ode(front_speed)
soln_str = "Solution found: " + str(np.round(front_speed,int(np.log10(1/tol)))) + " mm/s"
print(soln_str)

# Plot solution
if graph:
    plt.clf()
    plt.gcf().set_size_inches(8.5, 5.5)
    title_str = "ODE Front Profile: T0 = " + str(T0) + " °C, α0 = " + str(a0)
    plt.title(title_str,fontsize='xx-large')
    plt.xlabel("Position [mm]",fontsize='large')
    plt.ylabel("θ, α [-]",fontsize='large')
    plt.plot(1000.0*(front_space-front_space[0]),(front_sol[1,:]-(T0+273.15))/(T_max-(T0+273.15)), color='red', label='θ', lw=2.5)
    plt.plot(1000.0*(front_space-front_space[0]),front_sol[2,:], color='blue', label='α', lw=2.5)
    text_str = str(np.round(front_speed,int(np.log10(1/tol)))) + " mm/s"
    plt.text(1000.0*(np.mean(front_space-front_space[0]) + 0.05*(np.max(front_space-front_space[0])-np.min(front_space-front_space[0]))), 0.5, text_str, fontsize='x-large')
    plt.arrow(1000.0*(np.mean(front_space-front_space[0]) + 0.05*(np.max(front_space-front_space[0])-np.min(front_space-front_space[0]))), 0.45, 1, 0, width=0.01, head_length=0.125, fc='k')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.legend(loc='upper right',fontsize='large')
    plt.tight_layout()
    plt.savefig("../results/ODE_Profiles.png", dpi = 500)