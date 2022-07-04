# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:54:30 2022

@author: GKSch
"""

from Environment import Env
from Controller import Con
import time
import numpy as np
import multiprocessing as mp

#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#

def run_sim(verbose=False, render=False, controlled=True, **kwargs):

    # Initialize environment
    env = Env(verbose=verbose,render=render,total_time=10.0,heat_transfer_coeff=20.0,max_deviation_enthalpy_rxn=3500.0,max_deviation_init_temp=0.2,max_deviation_init_cure=0.005)
    
    # Initialize controller
    if controlled:
        con = Con(env,**kwargs)
        move_input_frames=env.times_C[env.times_C%(1./5.)<env.dt_C]
    
    done = False
    if verbose:
        print("Starting simulation...")
    t0 = time.time()
    while not done:
        if controlled:
            heat_CL, heat_F, heat_CR = con.get_opt_loc_input(env, con.get_tag_temp(env.mean_a0, 1.0), update_input_loc=(env.times_C[env.time_ind_C] in move_input_frames))
            done = env.step(heat_CL=heat_CL, heat_F=heat_F, heat_CR=heat_CR)
        else:
            done = env.step()
        if verbose and (env.times_C[env.time_ind_C] in env.frames):
            print("\tt={:.2f}".format(np.round(env.times_C[env.time_ind_C],3)),end='\r')

    # Post process front data
    if verbose:
        print("Post process...")
    front_x_loc, front_time, front_speed, ss_speed, ss_std = env.process_front()
    
    # Print time
    tf = time.time()
    if verbose:
        print("Simulation took: " + str(np.round(tf-t0,1)) + " seconds\n\t(" + str(np.round((tf-t0)/env.tot_t,2)) + " CPU seconds per sim second)")
        print("Steady state speed: " + str(np.round(ss_speed,4)) + " mm/s")
    
    # Plot and save results
    if verbose:
        print("Plot and render...")
    env.plot(front_x_loc, front_time, front_speed, ss_speed, ss_std, name="res")
    if render:
        env.render('res')
    if verbose:
        print("Done!")
        
#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
#__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__@__#
if __name__ == "__main__":
    
    # multi = False
    
    # if multi:
    #     # Define number of processes per batch
    #     num_processes_per_batch = 9
        
    #     # Setup kwargs for processing
        
    #     # Setup a list of processes that we want to run
    #     qr_rat = 10**np.linspace(0,8,9)
    #     processes = [mp.Process(target=run_sim, kwargs={'QR_ratio':qr_rat[x]}) for x in range(9)]
        
    #     # Determine process batches
    #     process_num = np.arange(len(processes))
    #     if num_processes_per_batch > len(processes):
    #         num_processes_per_batch = len(processes)
    #     batch_num = np.arange(len(processes))//num_processes_per_batch
        
    #     # Run processes
    #     for batch in range(np.max(batch_num)+1):
    #         print("Batch #" + str(batch+1) + "\tRunning processes: " + str(np.min(process_num[batch_num==batch])+1) + "-" + str(np.max(process_num[batch_num==batch])+1))
    #         batch_processes = [processes[x] for x in process_num[batch_num==batch]]
    #         for p in batch_processes:
    #             p.start()
    #         for p in batch_processes:
    #             p.join()
    #     print("Done!")
    # else:
    #     run_sim(verbose=True, render=True, controlled=True)
    
    verbose=True
    render=True
    controlled=True
    
    # Initialize environment
    env = Env(verbose=verbose,render=render,total_time=50.0,heat_transfer_coeff=20.0,max_deviation_enthalpy_rxn=3500.0,max_deviation_init_temp=0.2,max_deviation_init_cure=0.005)
    
    # Initialize controller
    if controlled:
        con = Con(env)
        move_input_frames=env.times_C[env.times_C%(1./5.)<env.dt_C]
    
    done = False
    if verbose:
        print("Starting simulation...")
    t0 = time.time()
    while not done:
        if controlled:
            heat_CL, heat_F, heat_CR = con.get_opt_loc_input(env, con.get_tag_temp(env.mean_a0, 1.0), update_input_loc=(env.times_C[env.time_ind_C] in move_input_frames))
            done = env.step(heat_CL=heat_CL, heat_F=heat_F, heat_CR=heat_CR)
        else:
            done = env.step()
        if verbose and (env.times_C[env.time_ind_C] in env.frames):
            print("\tt={:.2f}".format(np.round(env.times_C[env.time_ind_C],3)),end='\r')

    # Post process front data
    if verbose:
        print("Post process...")
    front_x_loc, front_time, front_speed, ss_speed, ss_std = env.process_front()
    
    # Print time
    tf = time.time()
    if verbose:
        print("Simulation took: " + str(np.round(tf-t0,1)) + " seconds\n\t(" + str(np.round((tf-t0)/env.tot_t,2)) + " CPU seconds per sim second)")
        print("Steady state speed: " + str(np.round(ss_speed,4)) + " mm/s")
    
    # Plot and save results
    if verbose:
        print("Plot and render...")
    env.plot(front_x_loc, front_time, front_speed, ss_speed, ss_std, name="res")
    if render:
        env.render('res')
    if verbose:
        print("Done!")