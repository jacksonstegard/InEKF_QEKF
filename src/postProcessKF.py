# ===================================================================
#   Post Processing Class for Invariant/Quaternion EKFs
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   This class is used for plotting results from Invariant/Quaternion
#   EKFs. Additionally for performance statistics for the state errors
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------
# Local Imports
# ---------------------------------
from utils import quat2euler, rot2Euler

# ---------------------------------
# Post Process Kalman Filter
# ---------------------------------
class postProcessKF:
    def __init__(self, x, tIMU, tGPS, imu, gps, truth, trajName, filterName, numMCs, P):
        # Save input 
        self.x = x
        self.tIMU = tIMU
        self.tGPS = tGPS
        self.imu = imu
        self.gps  = gps
        self.truth = truth
        self.trajName = trajName
        self.filterName = filterName
        self.numMCs = numMCs
        self.P = P
        
        # Process data for plotting depending on filter name
        if filterName == 'QEKF':
            # Consider randomized Monte Carlos
            if self.numMCs > 0:
                # Loop over Monte Carlos and get euler angles
                self.eulerAngles = np.zeros((self.x.shape[0], 3, self.x.shape[2]))
                for i in range(self.P.shape[0]):
                    self.eulerAngles[i,:,:] = quat2euler(self.x[i,6,:],self.x[i,7,:],self.x[i,8,:],self.x[i,9,:]).T * 180/np.pi
                
                # Get the reset of the states
                self.pos = self.x[:,0:3,:]
                self.vel = self.x[:,3:6,:]
                self.b_a = self.x[:,10:13,:]
                self.b_omega  = self.x[:,13:16,:]
                
            else:
                # Get states for a nonrandomized monte carlo
                self.eulerAngles = quat2euler(self.x[6,:],self.x[7,:],self.x[8,:],self.x[9,:]).T * 180/np.pi
                self.pos = self.x[0:3,:]
                self.vel = self.x[3:6,:]
                self.biasAccel = self.x[10:13,:]
                self.biasOmega  = self.x[13:16,:]
                
                # Get sigmas if just one monte carlo
                diagonals = np.sqrt(np.diagonal(self.P, axis1=1, axis2=2).T) # Convert to sigma
                self.sigmaPos = diagonals[0:3,:]
                self.sigmaVel = diagonals[3:6,:]
                self.sigmaEulerAngle = diagonals[6:9,:] * 180/np.pi
                self.sigmaBiasAccel = diagonals[9:12,:]
                self.sigmaBiasOmega = diagonals[12:15,:]
                
        elif filterName == 'InEKF':
            # Consider randomized Monte Carlos
            if self.numMCs > 0:
                # Initialize states
                self.eulerAngles = np.zeros((x.size,3,P.shape[1])) 
                self.vel = np.zeros((x.size,3,P.shape[1]))
                self.pos = np.zeros((x.size,3,P.shape[1]))
                self.biasOmega = np.zeros((x.size,3,P.shape[1]))
                self.biasAccel = np.zeros((x.size,3,P.shape[1]))
                
                # Get states for multiple monte carlo
                for i in range(x.size):
                    # Grab state tuple
                    x_i = x[i]
                    xState = x_i['xState']
                    thetaState = x_i['thetaState']
                    
                    # Define states
                    self.eulerAngles[i,:,:] = rot2Euler(xState[:,0:3,0:3]) * 180/np.pi
                    self.vel[i,:,:] = xState[:,0:3,3].T
                    self.pos[i,:,:] = xState[:,0:3,4].T
                    self.biasOmega[i,:,:]  = thetaState[0:3,:]
                    self.biasAccel[i,:,:] = thetaState[3:6,:]
            
            else:
                # Get states for a nonrandomized monte carlo
                xState = x['xState']
                thetaState = x['thetaState']
                self.eulerAngles = rot2Euler(xState[:,0:3,0:3]) * 180/np.pi
                self.vel = xState[:,0:3,3].T
                self.pos = xState[:,0:3,4].T
                self.biasOmega  = thetaState[0:3,:]
                self.biasAccel = thetaState[3:6,:]
                
                # Get sigmas if just one monte carlo
                diagonals = np.sqrt(np.diagonal(self.P, axis1=1, axis2=2).T) # Convert to sigma
                self.sigmaEulerAngle = diagonals[0:3,:] * 180/np.pi
                self.sigmaVel = diagonals[3:6,:]
                self.sigmaPos = diagonals[6:9,:]
                self.sigmaBiasOmega = diagonals[9:12,:]
                self.sigmaBiasAccel = diagonals[12:15,:]
                pass
    
    def plotTrajStates(self, plotCov = False, sigmaMult = 1):
        # Setup saving of figures
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "figs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply sigma multiplier if only one monte carlo is run
        if self.P.ndim < 4:
            self.sigmaPos *= sigmaMult
            self.sigmaVel *= sigmaMult
            self.sigmaEulerAngle *= sigmaMult
            self.sigmaBiasAccel *= sigmaMult
            self.sigmaBiasOmega *= sigmaMult
        
        # Plot positions
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        
        # - Plot x position
        if self.numMCs > 0:
            for i in range(self.P.shape[0]):
                ax[0].plot(self.tIMU, self.pos[i,0,:])
        else:
            ax[0].plot(self.tIMU, self.pos[0,:], color='blue', label=r'$\hat{p}_x$')
            if plotCov:
                ax[0].fill_between(self.tIMU, -self.sigmaPos[0,:]+self.pos[0,:], self.sigmaPos[0,:]+self.pos[0,:],
                                   color='blue', alpha=0.2) # label=r'$\pm 3\sigma$'
                ax[0].set_ylim(np.min([self.pos[0,:],self.truth['position'][:,0]])-1, np.max([self.pos[0,:],self.truth['position'][:,0]])+1)
            
        ax[0].plot(self.tIMU, self.truth['position'][:,0], color='black', label='$p_x$')
        ax[0].set_xlabel('Time [sec]')
        ax[0].set_ylabel('Position X [m]')
        # ax[0].grid(True)
        ax[0].legend(loc='upper right')
        ax[0].set_title(f'{self.filterName} - Positions - {self.trajName}')
        
        # - Plot y position
        if self.numMCs > 0:
            for i in range(self.P.shape[0]):
                ax[1].plot(self.tIMU, self.pos[i,1,:])
        else:
            ax[1].plot(self.tIMU, self.pos[1,:], color='blue', label=r'$\hat{p}_y$')
            if plotCov:
                ax[1].fill_between(self.tIMU, -self.sigmaPos[1,:]+self.pos[1,:], self.sigmaPos[1,:]+self.pos[1,:],
                                   color='blue', alpha=0.2) 
                ax[1].set_ylim(np.min([self.pos[1,:],self.truth['position'][:,1]])-1, np.max([self.pos[1,:],self.truth['position'][:,1]])+1)
            
        ax[1].plot(self.tIMU, self.truth['position'][:,1], color='black', label='$p_y$')
        ax[1].set_xlabel('Time [sec]')
        ax[1].set_ylabel('Position Y [m]')
        # ax[1].grid(True)
        ax[1].legend(loc='upper right')
        
        # - Plot z position
        if self.numMCs > 0:
            for i in range(self.P.shape[0]):
                ax[2].plot(self.tIMU, self.pos[i,2,:])
        else:
            ax[2].plot(self.tIMU, self.pos[2,:], color='blue', label=r'$\hat{p}_z$')
            if plotCov:
                ax[2].fill_between(self.tIMU, -self.sigmaPos[2,:]+self.pos[2,:], self.sigmaPos[2,:]+self.pos[2,:],
                                   color='blue', alpha=0.2) 
                ax[2].set_ylim(np.min([self.pos[2,:],self.truth['position'][:,2]])-1, np.max([self.pos[2,:],self.truth['position'][:,2]])+1)
            
        ax[2].plot(self.tIMU, self.truth['position'][:,2], color='black', label='$p_z$')
        ax[2].set_xlabel('Time [sec]')
        ax[2].set_ylabel('Position Z [m]')
        # ax[2].grid(True)
        ax[2].legend(loc='upper right')
        
        # - Show positions
        plt.tight_layout()
        plt.show()
        
        # - Save plot
        plot_filename = f"{self.filterName}_{self.trajName}_positions.pdf"
        plot_filepath = os.path.join(output_dir, plot_filename)
        fig.savefig(plot_filepath)
        
        # Plot Velocities
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        
        # - Plot x velocity
        if self.numMCs > 0:
            for i in range(self.P.shape[0]):
                ax[0].plot(self.tIMU, self.vel[i,0,:])
        else:
            ax[0].plot(self.tIMU, self.vel[0,:], color='blue', label=r'$\hat{v}_x$')
            if plotCov:
                ax[0].fill_between(self.tIMU, -self.sigmaVel[0,:]+self.vel[0,:], self.sigmaVel[0,:]+self.vel[0,:],
                                   color='blue', alpha=0.2) # label=r'$\pm 3\sigma$'
                ax[0].set_ylim(np.min([self.vel[0,:],self.truth['velocity'][:,0]])-1, np.max([self.vel[0,:],self.truth['velocity'][:,0]])+1)
            
        ax[0].plot(self.tIMU, self.truth['velocity'][:,0], color='black', label='$v_x$')
        ax[0].set_xlabel('Time [sec]')
        ax[0].set_ylabel('Velocity X [m/s]')
        # ax[0].grid(True)
        ax[0].legend(loc='upper right')
        ax[0].set_title(f'{self.filterName} - Velocities - {self.trajName}')
        
        # - Plot y velocity
        if self.numMCs > 0:
            for i in range(self.P.shape[0]):
                ax[1].plot(self.tIMU, self.vel[i,1,:])
        else:
            ax[1].plot(self.tIMU, self.vel[1,:], color='blue', label=r'$\hat{v}_y$')
            if plotCov:
                ax[1].fill_between(self.tIMU, -self.sigmaVel[1,:]+self.vel[1,:], self.sigmaVel[1,:]+self.vel[1,:],
                                   color='blue', alpha=0.2) 
                ax[1].set_ylim(np.min([self.vel[1,:],self.truth['velocity'][:,1]])-1, np.max([self.vel[1,:],self.truth['velocity'][:,1]])+1)
            
        ax[1].plot(self.tIMU, self.truth['velocity'][:,1], color='black', label='$v_y$')
        ax[1].set_xlabel('Time [sec]')
        ax[1].set_ylabel('Velocity Y [m/s]')
        # ax[1].grid(True)
        ax[1].legend(loc='upper right')
        
        # - Plot z velocity
        if self.numMCs > 0:
            for i in range(self.P.shape[0]):
                ax[2].plot(self.tIMU, self.vel[i,2,:])
        else:
            ax[2].plot(self.tIMU, self.vel[2,:], color='blue', label=r'$\hat{v}_z$')
            if plotCov:
                ax[2].fill_between(self.tIMU, -self.sigmaVel[2,:]+self.vel[2,:], self.sigmaVel[2,:]+self.vel[2,:],
                                   color='blue', alpha=0.2) 
                ax[2].set_ylim(np.min([self.vel[2,:],self.truth['velocity'][:,2]])-1, np.max([self.vel[2,:],self.truth['velocity'][:,2]])+1)
            
        ax[2].plot(self.tIMU, self.truth['velocity'][:,2], color='black', label='$v_z$')
        ax[2].set_xlabel('Time [sec]')
        ax[2].set_ylabel('Velocity Z [m/s]')
        # ax[2].grid(True)
        ax[2].legend(loc='upper right')
        
        # - Show Velocities
        plt.tight_layout()
        plt.show()
        
        # - Save plot
        plot_filename = f"{self.filterName}_{self.trajName}_velocities.pdf"
        plot_filepath = os.path.join(output_dir, plot_filename)
        fig.savefig(plot_filepath)
        
        # Plot attitudes        
        # - Setup figure
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        
        # - Plot x attitude
        if self.numMCs > 0:
            for i in range(self.P.shape[0]):
                ax[0].plot(self.tIMU, self.eulerAngles[i,0,:])
        else:
            ax[0].plot(self.tIMU, self.eulerAngles[0,:], color='blue', label=r'$\hat{\theta}_x$')
            if plotCov:
                ax[0].fill_between(self.tIMU, -self.sigmaEulerAngle[0,:]+self.eulerAngles[0,:], self.sigmaEulerAngle[0,:]+self.eulerAngles[0,:],
                                    color='blue', alpha=0.2)
                ax[0].set_ylim(np.min([self.eulerAngles[0,:],self.truth['theta'][:,0]])-0.1, np.max([self.eulerAngles[0,:],self.truth['theta'][:,0]])+0.1)
            
        ax[0].plot(self.tIMU, self.truth['theta_deg'][:,0], color='black', label=r'$\theta_x$')
        ax[0].set_ylim([np.min(self.truth['theta_deg'][:,0]) - 20, np.max(self.truth['theta_deg'][:,0]) + 20]) 
        ax[0].set_xlabel('Time [sec]')
        ax[0].set_ylabel('Attitude X [deg]')
        # ax[0].grid(True)
        ax[0].legend(loc='upper right')
        ax[0].set_title(f'{self.filterName} - Attitude - {self.trajName}')
        
        # - Plot y attitude
        if self.numMCs > 0:
            for i in range(self.P.shape[0]):
                ax[1].plot(self.tIMU, self.eulerAngles[i,1,:])
        else:
            ax[1].plot(self.tIMU, self.eulerAngles[1,:], color='blue', label=r'$\hat{\theta}_y$')
            if plotCov:
                ax[1].fill_between(self.tIMU, -self.sigmaEulerAngle[1,:]+self.eulerAngles[1,:], self.sigmaEulerAngle[1,:]+self.eulerAngles[1,:],
                                    color='blue', alpha=0.2)
                ax[1].set_ylim(np.min([self.eulerAngles[1,:],self.truth['theta'][:,1]])-0.1, np.max([self.eulerAngles[1,:],self.truth['theta'][:,1]])+0.1)
        
        # ax[1].set_ylim([np.min(self.truth['theta_deg'][:,1]) - 20, np.max(self.truth['theta_deg'][:,1]) + 20]) 
        ax[1].plot(self.tIMU, self.truth['theta_deg'][:,1], color='black', label=r'$\theta_y$')
        ax[1].set_xlabel('Time [sec]')
        ax[1].set_ylabel('Attitude Y [deg]')
        # ax[1].grid(True)
        ax[1].legend(loc='upper right')
        
        # - Plot z attitudes
        if self.numMCs > 0:
            for i in range(self.P.shape[0]):
                ax[2].plot(self.tIMU, self.eulerAngles[i,2,:])
        else:
            ax[2].plot(self.tIMU, self.eulerAngles[2,:], color='blue', label=r'$\hat{\theta}_z$')
            if plotCov:
                ax[2].fill_between(self.tIMU, -self.sigmaEulerAngle[2,:]+self.eulerAngles[2,:], self.sigmaEulerAngle[2,:]+self.eulerAngles[2,:],
                                    color='blue', alpha=0.2)
                ax[2].set_ylim(np.min([self.eulerAngles[2,:],self.truth['theta'][:,2]])-0.1, np.max([self.eulerAngles[2,:],self.truth['theta'][:,2]])+0.1)
            
        # ax[2].set_ylim([np.min(self.truth['theta_deg'][:,2]) - 20, np.max(self.truth['theta_deg'][:,2]) + 20]) 
        ax[2].plot(self.tIMU, self.truth['theta_deg'][:,2], color='black', label=r'$\theta_z$')
        ax[2].set_xlabel('Time [sec]')
        ax[2].set_ylabel('Attitude Z [deg]')
        # ax[2].grid(True)
        ax[2].legend(loc='upper right')
        
        # - Show attitudes
        plt.tight_layout()
        plt.show()
        
        # - Save plot
        plot_filename = f"{self.filterName}_{self.trajName}_attitudes.pdf"
        plot_filepath = os.path.join(output_dir, plot_filename)
        fig.savefig(plot_filepath)
        
    def plotDebug(self, x = None):
        # If x is not provided use class attribute
        if x is None:
            x = self.x
            
        # Debug
        # - Plot Y position truth and gps
        # plt.figure()
        # plt.plot(self.tGPS, self.gps['position'][:,1], label='gps y')
        # plt.plot(self.tIMU, self.truth['position'][:,1], label='truth y')
        # plt.legend()
        
        # # # - Plot Z position truth and gps
        # plt.figure()
        # plt.plot(self.tGPS, self.gps['position'][:,2],marker='o', label='gps z')
        # plt.plot(self.tIMU, self.truth['position'][:,2],marker='o', label='truth z')
        # plt.legend()
        
        # - Plot accel bias
        plt.figure()
        plt.plot(self.tIMU, self.biasAccel[0,:], color='blue', linestyle='--', label=r'$\hat{b}_{a,x}$')
        plt.plot(self.tIMU, self.imu['accelBias'][:,0], color='blue', label=r'$b_{a,x}$')
        plt.plot(self.tIMU, self.biasAccel[1,:], color='red', linestyle='--', label=r'$\hat{b}_{a,y}$')
        plt.plot(self.tIMU, self.imu['accelBias'][:,1], color='red', label=r'$b_{a,y}$')
        plt.plot(self.tIMU, self.biasAccel[2,:], color='green', linestyle='--', label=r'$\hat{b}_{a,z}$')
        plt.plot(self.tIMU, self.imu['accelBias'][:,2], color='green', label=r'$b_{a,z}$')
        plt.legend()
        
        # Plot gyro bias
        plt.figure()
        plt.plot(self.tIMU, self.biasOmega[0,:],color='blue', linestyle='--', label=r'$\hat{b}_{g,x}$')
        plt.plot(self.tIMU, self.imu['gyroBias'][:,0], color='blue', label=r'$b_{g,x}$')
        plt.plot(self.tIMU, self.biasOmega[1,:],color='red', linestyle='--', label=r'$\hat{b}_{g,y}$')
        plt.plot(self.tIMU, self.imu['gyroBias'][:,1], color='red', label=r'$b_{g,y}$')
        plt.plot(self.tIMU, self.biasOmega[2,:],color='green', linestyle='--', label=r'$\hat{b}_{g,z}$')
        plt.plot(self.tIMU, self.imu['gyroBias'][:,2], color='green', label=r'$b_{g,z}$')
        plt.legend()
        
        
    def printResults(self):
        # Calculate error
        # If Monte Carlos exist compare mean timeseries to truth
        if self.numMCs > 0:
            numMCs = self.P.shape[0]
            posError = np.abs(np.mean(self.pos[:,0:3,],axis=0) - self.truth['position'].T)
            velError = np.abs(np.mean(self.vel[:,0:3,],axis=0) - self.truth['velocity'].T)
            attError = np.abs(np.mean(self.eulerAngles[:,0:3,],axis=0) - self.truth['theta_deg'].T)
        else:
            numMCs = 1
            posError = np.abs(self.pos[0:3, :] - self.truth['position'].T)
            velError = np.abs(self.vel[0:3, :] - self.truth['velocity'].T)
            attError = np.abs(self.eulerAngles - self.truth['theta_deg'].T)
        
        
        # Print title
        print("------------------------------------------")
        print(f"Filter: {self.filterName}")
        print(f"Trajectory: {self.trajName}")
        print(f"Monte Carlos: {numMCs}\n")
        
        # Print position errors
        print("Position Errors [m]:")
        print(f"X Error: Mean = {np.mean(posError[0, :]):.4f}, Max = {np.max(posError[0, :]):.4f}")
        print(f"Y Error: Mean = {np.mean(posError[1, :]):.4f}, Max = {np.max(posError[1, :]):.4f}")
        print(f"Z Error: Mean = {np.mean(posError[2, :]):.4f}, Max = {np.max(posError[2, :]):.4f}")
        
        # Print velocity errors
        print("\nVelocity Errors [m/s]:")
        print(f"X Error: Mean = {np.mean(velError[0, :]):.4f}, Max = {np.max(velError[0, :]):.4f}")
        print(f"Y Error: Mean = {np.mean(velError[1, :]):.4f}, Max = {np.max(velError[1, :]):.4f}")
        print(f"Z Error: Mean = {np.mean(velError[2, :]):.4f}, Max = {np.max(velError[2, :]):.4f}")
        
        # Print attitude errors
        print("\nAttitude Errors [deg]:")
        print(f"X Error: Mean = {np.mean(attError[0, :]):.4f}, Max = {np.max(attError[0, :]):.4f}")
        print(f"Y Error: Mean = {np.mean(attError[1, :]):.4f}, Max = {np.max(attError[1, :]):.4f}")
        print(f"Z Error: Mean = {np.mean(attError[2, :]):.4f}, Max = {np.max(attError[2, :]):.4f}")
        print("------------------------------------------")
