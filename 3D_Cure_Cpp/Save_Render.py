import PPO_Agent_3_Output as ppo
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy import signal
from scipy.stats import multivariate_normal

class Run:

    def __init__(self, agent, r_per_episode, x_rate_stdev, y_rate_stdev, mag_stdev, 
                 value_error, input_location_x, input_location_y, input_percent, time, 
                 target, temperature_field, cure_field, front_location, front_velocity, 
                 front_temperature, best_reward, mesh_x_z0, mesh_y_z0, max_input_mag,
                 exp_const, mesh_y_x0, mesh_z_x0, control_speed, render):

        print("Saving results...")

        data = {
            'r_per_episode' : np.array(r_per_episode),
            'value_error' : np.array(value_error),
            'x_rate_stdev': np.array(x_rate_stdev),
            'y_rate_stdev': np.array(y_rate_stdev),
            'mag_stdev': np.array(mag_stdev),
            'input_location_x': np.array(input_location_x),
            'input_location_y': np.array(input_location_y),
            'input_percent': np.array(input_percent),
            'temperature_field': np.array(temperature_field),
            'cure_field': np.array(cure_field),
            'front_location': np.array(front_location),
            'front_velocity': np.array(front_velocity),
            'front_temperature': np.array(front_temperature),
            'target': np.array(target),
            'time': np.array(time),
            'best_reward': best_reward,
            'mesh_x_z0' : np.array(mesh_x_z0),
            'mesh_y_z0' : np.array(mesh_y_z0),
            'max_input_mag' : max_input_mag,
            'exp_const' : exp_const,
            'mesh_y_x0' : np.array(mesh_y_x0),
            'mesh_z_x0' : np.array(mesh_z_x0),
            'control_speed' : control_speed,
			'render': render,
        }

        # Find save paths
        done = False
        curr_folder = 1
        while not done:
            path = "results/PPO_"+str(curr_folder)
            video_path = "results/PPO_"+str(curr_folder)+"/video/"
            if not os.path.isdir(path):
                os.mkdir(path)
                os.mkdir(video_path)
                done = True
            else:
                curr_folder = curr_folder + 1

        # Pickle all important outputs
        output = { 'data':data, 'agent':agent }
        save_file = path + "/output"
        with open(save_file, 'wb') as file:
            pickle.dump(output, file)

        # Plot the trajectory
        print("Plotting...")
        if data['control_speed']==1:
            # Plot speed trajectory
            plt.clf()
            plt.title("Front Velocity",fontsize='xx-large')
            plt.xlabel("Simulation Time [s]",fontsize='large')
            plt.ylabel("Front Velocity [mm/s]",fontsize='large')
            plt.plot(data['time'], 1000.0*(np.mean(data['front_velocity'],axis=(0,1))),c='r',lw=2.5)
            plt.plot(data['time'], 1000.0*(data['target']),c='b',ls='--',lw=2.5)
            plt.legend(('Actual','Target'),loc='best',fontsize='large')
            plt.ylim(0.0, 1500.0*np.max(data['target']))
            plt.xlim(0.0, np.round(data['time'][-1]))
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            save_file = path + "/trajectory.png"
            plt.savefig(save_file, dpi = 500)
            plt.close()
            
            # Plot front temperature trajectory
            plt.clf()
            plt.title("Front Temperature",fontsize='xx-large')
            plt.xlabel("Location [mm]",fontsize='large')
            plt.ylabel("Front Temperature [C]",fontsize='large')
            plt.plot(np.mean(1000.0*np.array(data['front_location']),axis=(0,1)), (np.mean(data['front_temperature'],axis=(0,1))-273.15),c='r',lw=2.5)
            plt.ylim(0.0, np.max(1.025*(np.mean(data['front_temperature'],axis=(0,1))-273.15)))
            plt.xlim(0.0, 1000.0*data['mesh_x_z0'][-1,0])
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            save_file = path + "/temp.png"
            plt.savefig(save_file, dpi = 500)
            plt.close()
            
        else:
            # Plot speed trajectory
            plt.clf()
            plt.title("Front Velocity",fontsize='xx-large')
            plt.xlabel("Simulation Time [s]",fontsize='large')
            plt.ylabel("Front Velocity [mm/s]",fontsize='large')
            plt.plot(data['time'], 1000.0*(np.mean(data['front_velocity'],axis=(0,1))),c='r',lw=2.5)
            plt.ylim(0.0, np.max(1025.0*np.array(np.mean(data['front_velocity'],axis=(0,1)))))
            plt.xlim(0.0, np.round(data['time'][-1]))
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            save_file = path + "/speed.png"
            plt.savefig(save_file, dpi = 500)
            plt.close()
            
            # Plot front temperature trajectory
            plt.clf()
            plt.title("Front Temperature",fontsize='xx-large')
            plt.xlabel("Location [mm]",fontsize='large')
            plt.ylabel("Front Temperature [C]",fontsize='large')
            plt.plot(np.mean(1000.0*np.array(data['front_location']),axis=(0,1)), (np.mean(data['front_temperature'],axis=(0,1))-273.15),c='r',lw=2.5)
            plt.plot(np.mean(1000.0*np.array(data['front_location']),axis=(0,1)), (data['target']-273.15),c='b',ls='--',lw=2.5)
            plt.legend(('Actual','Target'),loc='best',fontsize='large')
            plt.ylim(0.0, 1.5*(np.max(data['target'])-273.15))
            plt.xlim(0.0, 1000.0*data['mesh_x_z0'][-1,0])
            plt.xticks(fontsize='large')
            plt.yticks(fontsize='large')
            plt.gcf().set_size_inches(8.5, 5.5)
            save_file = path + "/trajectory.png"
            plt.savefig(save_file, dpi = 500)
            plt.close()

        #Plot actor learning curve
        plt.clf()
        plt.title("Actor Learning Curve, Episode-Wise",fontsize='xx-large')
        plt.xlabel("Episode",fontsize='large')
        plt.ylabel("Average Reward per Simulation Step",fontsize='large')
        plt.plot([*range(len(data['r_per_episode']))],data['r_per_episode'],lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/actor_learning.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()

        # Plot value learning curve
        plt.clf()
        title_str = "Critic Learning Curve"
        plt.title(title_str,fontsize='xx-large')
        plt.xlabel("Optimization Step",fontsize='large')
        plt.ylabel("MSE Loss",fontsize='large')
        plt.plot([*range(len(data['value_error']))],data['value_error'],lw=2.5,c='r')
        plt.yscale("log")
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/critic_learning.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()

        # Plot x rate stdev curve
        plt.clf()
        plt.title("Laser X Position Rate Stdev",fontsize='xx-large')
        plt.xlabel("Episode",fontsize='large')
        plt.ylabel("Laser X Position Rate Stdev [m/s]",fontsize='large')
        plt.plot([*range(len(data['x_rate_stdev']))],np.array(data['x_rate_stdev']),lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/x_rate_stdev.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()

        # Plot y rate stdev curve
        plt.clf()
        plt.title("Laser Y Position Rate Stdev",fontsize='xx-large')
        plt.xlabel("Episode",fontsize='large')
        plt.ylabel("Laser Y Position Rate Stdev [m/s]",fontsize='large')
        plt.plot([*range(len(data['y_rate_stdev']))],np.array(data['y_rate_stdev']),lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/y_rate_stdev.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()

        # Plot magnitude stdev curve
        plt.clf()
        plt.title("Laser Magnitude Stdev",fontsize='xx-large')
        plt.xlabel("Episode",fontsize='large')
        plt.ylabel('Laser Magnitude Stdev [K/s]',fontsize='large')
        plt.plot([*range(len(data['mag_stdev']))],np.array(data['mag_stdev']),lw=2.5,c='r')
        plt.xticks(fontsize='large')
        plt.yticks(fontsize='large')
        plt.gcf().set_size_inches(8.5, 5.5)
        save_file = path + "/mag_stdev.png"
        plt.savefig(save_file, dpi = 500)
        plt.close()

        # Make videos of the best temperature field trajecotry and cure field trajectories as function of time
        if data['render'] == 1:
            print("Rendering...")
            min_temp = 10.0*np.floor((np.min(data['temperature_field'])-273.15)/10.0)
            max_temp = 10.0*np.ceil((np.max(data['temperature_field'])-273.15)/10.0)
            
            # Determine front shape deltas
            front_mean_loc = np.mean(1000.0*np.array(data['front_location']),axis=(0,1))
            min_loc = 0.5*np.floor((np.min(np.min(1000.0*np.array(data['front_location']),axis=(0,1)) - front_mean_loc))/0.5)
            max_loc = 0.5*np.ceil((np.max(np.max(1000.0*np.array(data['front_location']),axis=(0,1)) - front_mean_loc))/0.5)
            
            # Determine front speed deltas
            max_vel = 0.5*np.ceil((np.max(1000.0*data['front_velocity']))/0.5)
            
            # Determine radius of convolution
            radius_of_conv = int(np.round(len(data['mesh_y_x0'])*len(data['mesh_y_x0'][0])/100)*2.0-1.0)
            
            for curr_step in range(len(data['time'])):
            
            	# Calculate input field
            	input_percent = data['input_percent'][curr_step]
            	input_location_x = data['input_location_x'][curr_step]
            	input_location_y = data['input_location_y'][curr_step]
            	input_mesh = input_percent*data['max_input_mag']*np.exp(((data['mesh_x_z0']-input_location_x)**2*data['exp_const']) +
            														   (data['mesh_y_z0']-input_location_y)**2*data['exp_const'])
            	input_mesh[input_mesh<0.01*data['max_input_mag']] = 0.0
            
            	# Make fig for temperature, cure, and input
            	plt.cla()
            	plt.clf()
            	fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
            	fig.set_size_inches(11,8.5)
            
            	# Plot temperature
            	c0 = ax0.pcolormesh(1000.0*data['mesh_x_z0'], 1000.0*data['mesh_y_z0'], data['temperature_field'][:,:,curr_step]-273.15, shading='gouraud', cmap='jet', vmin=min_temp, vmax=max_temp)
            	cbar0 = fig.colorbar(c0, ax=ax0)
            	cbar0.set_label('Temperature [C]',labelpad=20,fontsize='large')
            	cbar0.ax.tick_params(labelsize=12)
            	ax0.set_xlabel('X Position [mm]',fontsize='large')
            	ax0.set_ylabel('Y Position [mm]',fontsize='large')
            	ax0.tick_params(axis='x',labelsize=12)
            	ax0.tick_params(axis='y',labelsize=12)
            	ax0.set_aspect('equal', adjustable='box')
            	ax0.set_title('Max Temperature = '+'{:.2f}'.format(np.max(data['temperature_field'][:,:,curr_step]-273.15))+' C',fontsize='large')
            
            	# Plot cure
            	c1 = ax1.pcolormesh(1000.0*data['mesh_x_z0'], 1000.0*data['mesh_y_z0'], data['cure_field'][:,:,curr_step], shading='gouraud', cmap='YlOrBr', vmin=0.0, vmax=1.0)
            	cbar1 = fig.colorbar(c1, ax=ax1)
            	cbar1.set_label('Degree Cure [-]', labelpad=20,fontsize='large')
            	cbar1.ax.tick_params(labelsize=12)
            	ax1.set_xlabel('X Position [mm]',fontsize='large')
            	ax1.set_ylabel('Y Position [mm]',fontsize='large')
            	ax1.tick_params(axis='x',labelsize=12)
            	ax1.tick_params(axis='y',labelsize=12)
            	ax1.set_aspect('equal', adjustable='box')
            
            	# Plot input
            	c2 = ax2.pcolormesh(1000.0*data['mesh_x_z0'], 1000.0*data['mesh_y_z0'], 1.0e-3*input_mesh, shading='gouraud', cmap='coolwarm', vmin=0.0, vmax=1.0e-3*data['max_input_mag'])
            	ax2.plot(1000.0*data['front_location'][:,0,curr_step].squeeze(), 1000.0*data['mesh_y_z0'][0,:], 'k-', lw=1.5)
            	cbar2 = fig.colorbar(c2, ax=ax2)
            	cbar2.set_label('Input Heat [KW/m^2]',labelpad=20,fontsize='large')
            	cbar2.ax.tick_params(labelsize=12)
            	ax2.set_xlabel('X Position [mm]',fontsize='large')
            	ax2.set_ylabel('Y Position [mm]',fontsize='large')
            	ax2.tick_params(axis='x',labelsize=12)
            	ax2.tick_params(axis='y',labelsize=12)
            	ax2.set_aspect('equal', adjustable='box')
            
            	# Set title and save
            	title_str = "Time From Trigger: "+'{:.2f}'.format(data['time'][curr_step])+'s'
            	fig.suptitle(title_str,fontsize='xx-large')
            	plt.savefig(video_path+str(curr_step).zfill(4)+'.png', dpi=100)
            	plt.close()
            
            	# Make fig for front location and velocity
            	plt.cla()
            	plt.clf()
            	fig, (ax0, ax1) = plt.subplots(1,2,subplot_kw={"projection": "3d"})
            	fig.set_size_inches(14.0,8.0)
            
            	# Convolve front location data
            	back_msaa_index = np.clip(curr_step-5,0,len(data['time'])-1)
            	front_msaa_index = np.clip(curr_step+5,0,len(data['time'])-1)
            	front_delta_loc = np.mean(1000.0*np.array(data['front_location'][:,:,back_msaa_index:front_msaa_index]),axis=2) - np.mean(front_mean_loc[back_msaa_index:front_msaa_index])
            	front_delta_min = np.min(front_delta_loc)
            	front_delta_max = np.max(front_delta_loc)
            	if not ((front_delta_loc<=1.0e-4).all() and (front_delta_loc>=-1.0e-4).all()):
            		x,y=np.meshgrid(np.linspace(-1,1,radius_of_conv),np.linspace(-1,1,radius_of_conv))
            		win=multivariate_normal.pdf(np.dstack((x,y)),mean=[0,0],cov=[[1.0,0.0],[0.0,1.0]])
            		padded = front_delta_loc
            		for i in range(int((radius_of_conv+1)/2)-1):
            			padded = np.append(padded[:,0].reshape(len(padded[:,0]),1),padded,axis=1)
            			padded = np.append(padded[0,:].reshape(1,len(padded[0,:])),padded,axis=0)
            			padded = np.append(padded,padded[:,-1].reshape(len(padded[:,-1]),1),axis=1)
            			padded = np.append(padded,padded[-1,:].reshape(1,len(padded[-1,:])),axis=0)
            		out = signal.convolve2d(padded,win,mode='valid')
            		out=out*((front_delta_max-front_delta_min)/(np.max(out)-np.min(out)))
            		out=out-np.mean(out)
            	else:
            		out = front_delta_loc
            
            	# Plot front location
            	ax0.plot_surface(1000.0*data['mesh_y_x0'], 1000.0*data['mesh_z_x0'],out,cmap='coolwarm',vmin=min_loc,vmax=max_loc,alpha=1.0)
            	ax0.set_xlabel('Y Position [mm]',fontsize='large',labelpad=15)
            	ax0.set_ylabel('Z Position [mm]',fontsize='large',labelpad=15)
            	ax0.set_zlabel('Lengthwise Delta [mm]',fontsize='large',labelpad=20)
            	ax0.tick_params(axis='x',labelsize=12,pad=10)
            	ax0.tick_params(axis='y',labelsize=12,pad=10)
            	ax0.tick_params(axis='z',labelsize=12,pad=10)
            	ax0.set_zlim(min_loc,max_loc)
            	ax0.set_title("Front Shape",fontsize='xx-large')
            
            	# Covolve front speed data
            	back_msaa_index = np.clip(curr_step-5,0,len(data['time'])-1)
            	front_msaa_index = np.clip(curr_step+5,0,len(data['time'])-1)
            	curr_front_vel = np.mean(1000.0*np.array(data['front_velocity'][:,:,back_msaa_index:front_msaa_index]),axis=2)
            	front_vel_min = np.min(curr_front_vel)
            	front_vel_max = np.max(curr_front_vel)
            	if not ((curr_front_vel<=1.0e-4).all() and (curr_front_vel>=-1.0e-4).all()):
            		x,y=np.meshgrid(np.linspace(-1,1,radius_of_conv),np.linspace(-1,1,radius_of_conv))
            		win=multivariate_normal.pdf(np.dstack((x,y)),mean=[0,0],cov=[[1.0,0.0],[0.0,1.0]])
            		padded = curr_front_vel
            		for i in range(int((radius_of_conv+1)/2)-1):
            			padded = np.append(padded[:,0].reshape(len(padded[:,0]),1),padded,axis=1)
            			padded = np.append(padded[0,:].reshape(1,len(padded[0,:])),padded,axis=0)
            			padded = np.append(padded,padded[:,-1].reshape(len(padded[:,-1]),1),axis=1)
            			padded = np.append(padded,padded[-1,:].reshape(1,len(padded[-1,:])),axis=0)
            		out = signal.convolve2d(padded,win,mode='valid')
            		out=out*((front_vel_max-front_vel_min)/(np.max(out)-np.min(out)))
            		out=out-np.mean(out)+np.mean(curr_front_vel)
            	else:
            		out = curr_front_vel
            
            	# Plot front speed
            	ax1.plot_surface(1000.0*data['mesh_y_x0'],1000.0*data['mesh_z_x0'],out,cmap='coolwarm',vmin=0.0,vmax=max_vel,alpha=1.0)
            	ax1.set_xlabel('Y Position [mm]',fontsize='large',labelpad=15)
            	ax1.set_ylabel('Z Position [mm]',fontsize='large',labelpad=15)
            	ax1.set_zlabel('Front Speed [mm/s]',fontsize='large',labelpad=20)
            	ax1.tick_params(axis='x',labelsize=12,pad=10)
            	ax1.tick_params(axis='y',labelsize=12,pad=10)
            	ax1.tick_params(axis='z',labelsize=12,pad=10)
            	ax1.set_zlim(0.0,max_vel)
            	ax1.set_title("Front Speed",fontsize='xx-large')
            
            	# Set title and save
            	title_str = "Time From Trigger: "+'{:.2f}'.format(data['time'][curr_step])+'s'
            	fig.suptitle(title_str,fontsize='xx-large')
            	plt.savefig(video_path+"f_"+str(curr_step).zfill(4)+'.png', dpi=100)
            	plt.close()