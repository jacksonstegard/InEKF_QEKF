# ===================================================================
#   Read the Mid Air Dataset and Generate Data
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   This class reads data from the Mid Air Dataset. Additional 
#   options are given to generate IMU and GPS data.
# -------------------------------------------------------------------
#   Link:
#   https://midair.ulg.ac.be/data_organization.html
# -------------------------------------------------------------------
# Data format
# tajectory_i
#   ├ camera_data
#    ├ color_left
#    ├ color_right
#    ├ color_down
#    ├ depth
#    ├ normals
#    ├ segmentation
#    ├ stereo_disparity
#    └ stereo_occlusion
#   ├ gps
#    ├ GDOP
#    ├ HDOP
#    ├ PDOP
#    ├ VDOP
#    ├ no_vis_sats
#    ├ position
#    └ velocity
#   ├ groundtruth
#    ├ attitude
#    ├ angular_velocity
#    ├ position
#    ├ velocity
#    └ acceleration
#   └ imu
#    ├ accelerometer
#    └ gyroscope
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import h5py
import numpy as np
from collections import defaultdict

# ---------------------------------
# Local Modules
# ---------------------------------
from utils.quat2euler import quat2euler
from utils.quat2rot import quat2rot

# ---------------------------------
# MAD Dataset Loader
# ---------------------------------
class dataLoader:
    def __init__(self, filename, trajectoryNames, groups):
        # Initialize MADLoader object
        self.filename = filename
        self.trajectoryNames = trajectoryNames
        self.groups = groups
        self.data = defaultdict(lambda: defaultdict(dict)) # Avoids needing to check key existence
    
    # Load all data
    def loadData(self):
        # Load file and get data for desired trajectory
        with h5py.File(self.filename, "r") as f:
            # Loop over trajectories
            for trajectoryName in self.trajectoryNames:
                
                # Loop over groups
                for group in self.groups:
                    # Get data sets in group
                    dataSets = list(f[trajectoryName + '/' + group].keys())
                    
                    # Get data in data set
                    for data in dataSets:
                        # Path to data
                        dataPath = f[trajectoryName + '/' + group + '/' + data]
                        
                        # Add data to the appropriate place in the dictionary
                        self.data[trajectoryName][group][data] = {
                            'data' : dataPath[()],
                            'metaData' : dict(dataPath.attrs)
                            }
                        
    # Get specific data                 
    def getData(self, trajectoryName, group, data):
        # Return specific data, if dictionary or data doesn't exist return {} or None
        return self.data.get(trajectoryName, {}).get(group, {}).get(data, {}).get('data',None)
          
    # Get specific meta data      
    def getMetaData(self, trajectoryName, group, data, metaDataVar):
        # Return specific data, if dictionary or data doesn't exist return {} or None
        return self.data.get(trajectoryName, {}).get(group, {}).get(data, None).get('metaData', {}).get(metaDataVar,None)
    
    # General case for downloading measurement and truth data
    def getGeneralData(self, trajectoryName):
        # Setup dictionaries
        gps, imu, truth = {}, {}, {}
        
        # GPS data
        gps['position'] = self.getData(trajectoryName, 'gps', 'position') # world frame  [m]
        gps['velocity'] = self.getData(trajectoryName,'gps', 'velocity') # world frame [m/s]
        
        # IMU data
        imu['accel'] = self.getData(trajectoryName,'imu', 'accelerometer') # reference local [m/s^2]
        imu['accelInitBias'] = self.getMetaData(trajectoryName,'imu', 'accelerometer','init_bias_est').flatten() # reference local [m/s^2]
        imu['gyro'] = self.getData(trajectoryName,'imu', 'gyroscope') # reference local [rad/s]
        imu['gyroInitBias'] = self.getMetaData(trajectoryName,'imu', 'gyroscope','init_bias_est').flatten() # reference local [rad/s]
        
        # Ground truth
        truth['accel'] = self.getData(trajectoryName,'groundtruth', 'acceleration')     # world frame [m/s^2]
        truth['omega'] = self.getData(trajectoryName,'groundtruth', 'angular_velocity') # body frame [rad/s]
        truth['quaternion'] = self.getData(trajectoryName,'groundtruth', 'attitude')    # world frame (w,x,y,z)  
        truth['theta'] = quat2euler(truth['quaternion'][:,0],truth['quaternion'][:,1],\
                                    truth['quaternion'][:,2],truth['quaternion'][:,3]) # world frame (x,y,x)     
        truth['theta_deg'] = truth['theta'].copy() * 180 / np.pi
        truth['position'] = self.getData(trajectoryName,'groundtruth', 'position') # world frame [m]
        truth['velocity'] = self.getData(trajectoryName,'groundtruth', 'velocity') # world frame [m/s]
        
        return gps, imu, truth 
            
    # Generate GPS data
    def generateGPS(self, trajectoryName, posSigma = 5):
        # Get groundtruth position and velocity
        pos = self.getData(trajectoryName,'groundtruth', 'position').copy() # world frame [m]
        # vel = self.getData(trajectoryName,'groundtruth', 'velocity').copy() # world frame [m/s]
        
        # Get groundtruth samples at GPS sampling rate
        idxGT2GPS = int(self.data[trajectoryName]["imu"]["accelerometer"]["metaData"]["sampling_frequency"] \
                        / self.data[trajectoryName]["gps"]["position"]["metaData"]["sampling_frequency"])
        
        pos = pos[0:-1:idxGT2GPS]
        # vel = vel[0:-1:idxGT2GPS]
        
        # Add noise to ground truth
        pos += np.random.randn(3, len(pos)).T * posSigma
        # vel += np.random.randn(3, len(vel)).T * velSigma
        
        # Define output
        gps = {}
        gps['position'] = pos
        # gps['velocity'] = vel

        return gps
    
    # Generate Magnetometer Data
    def generateMag(self, trajectoryName, magSampleRate, magSample, magSigma):
        # Assume a constant local tri-axial magnetometer sample in [nT]

        # Get groundtruth samples at Magnetometer sampling rate
        idxGT2Mag = int(self.data[trajectoryName]["imu"]["accelerometer"]["metaData"]["sampling_frequency"] / magSampleRate)
         
        # Get quaternion data
        q = self.getData(trajectoryName,'groundtruth', 'attitude').copy() # world frame (w,x,y,z) 
        q = q[0:-1:idxGT2Mag]
        
        # Predefine magneometer data
        mag = {}
        magMeasOut = np.zeros((q.shape[0],3))
        
        # Loop over trajectory to define measurements
        for i in range(q.shape[0]):
            # Get rotation matrix
            R = quat2rot(q[i,0], q[i,1], q[i,2], q[i,3])
            
            magMeasOut[i,:] = R.T @ magSample + np.random.normal([0., 0., 0.], magSigma)
            
        mag['intensity'] = magMeasOut
        
        return mag
    
    # Generate IMU data
    def generateIMU(self, trajectoryName, noiseAccel=None, noiseGyro=None, noiseBiasAcc=None, noiseBiasGyro=None):
        # Reference code here:
        # https://github.com/montefiore-institute/midair-dataset/blob/master/tools/IMU-data_generator.py#L8
        imu = {}
        
        # Get required groundtruths
        trueAccel = self.getData(trajectoryName,'groundtruth', 'acceleration').copy()     # world frame [m/s^2]
        trueOmega = self.getData(trajectoryName,'groundtruth', 'angular_velocity').copy() # body frame [rad/s]
        trueQuat = self.getData(trajectoryName,'groundtruth', 'attitude').copy()    # world frame (w,x,y,z)
        
        # IMU noise parameters chosen randomly in a range of values encountered in real devices
        if noiseAccel is None:
            noiseAccel = 5 * np.power(10., -np.random.uniform(low=2., high=4., size=(1, 3)))
            
        if noiseGyro is None:
            noiseGyro = 8 * np.power(10., -np.random.uniform(low=3., high=5., size=(1, 3)))
            
        if noiseBiasAcc is None:
            noiseBiasAcc = 2 * np.power(10., -np.random.uniform(low=3., high=5., size=(1, 3)))
            
        if noiseBiasGyro is None:
            noiseBiasGyro = np.power(10., -np.random.uniform(low=4., high=6., size=(1, 3)))
        
        # Get initial parameters
        accelMeasOut = np.zeros(trueAccel.shape)
        gyroMeasOut = np.zeros(trueOmega.shape)
        imuBiasAccelOut = np.zeros(trueAccel.shape)
        imuBiasGyroOut = np.zeros(trueOmega.shape)
        
        imuBiasAccel = np.random.normal([0., 0., 0.], noiseBiasAcc)
        imuBiasGyro = np.random.normal([0., 0., 0.], noiseBiasGyro)
        
        # imu['accelInitBias'] = imuBiasAccel + np.random.normal([0., 0., 0.], noiseAccel / 50)
        # imu['gyroInitBias'] = imuBiasGyro + np.random.normal([0., 0., 0.], noiseGyro / 50)
        
        imu['accelInitBias'] = imuBiasAccel.copy()
        imu['gyroInitBias'] = imuBiasGyro.copy()
        
        # Loop over trajectory to define measurements
        for i in range(trueQuat.shape[0]):
            # Get rotation matrix
            R = quat2rot(trueQuat[i,0], trueQuat[i,1], trueQuat[i,2], trueQuat[i,3])
            
            # Define accelerometer measurement
            accelMeasOut[i,:] = R.T @ (trueAccel[i,:] + np.array([0., 0., -9.81])) + \
                                      imuBiasAccel + np.random.normal([0., 0., 0.], noiseAccel)
            
            # Define gyro measurement
            gyroMeasOut[i,:] = trueOmega[i,:] + imuBiasGyro + np.random.normal([0., 0., 0.], noiseGyro)

            # Update bias for measurements
            imuBiasAccelOut[i,:] = imuBiasAccel
            imuBiasGyroOut[i,:] = imuBiasGyro
            imuBiasAccel += np.random.normal([0., 0., 0.], noiseBiasAcc)
            imuBiasGyro += np.random.normal([0., 0., 0.], noiseBiasGyro)
         
        imu['accel'] = accelMeasOut
        imu['gyro'] = gyroMeasOut
        imu['accelBias'] = imuBiasAccelOut
        imu['gyroBias'] = imuBiasGyroOut
        
        return imu
        
        

# Example
if __name__ == "__main__":
    filename = "sensor_records_sunny.hdf5"
    trajectoryNames = ["trajectory_0000","trajectory_0001"]
    groups = ["gps", "groundtruth", "imu"] 
    specificData = 'accelerometer'
    specificMetaData = 'init_bias_est'
    
    madData = MADLoader(filename, trajectoryNames, groups)
    madData.loadData()
    GDOP = madData.getData("trajectory_0000", "imu", specificData)
    GDOP_init_bias_est = madData.getMetaData("trajectory_0000", "imu", specificData, specificMetaData)
    
    gps, imu, truth = madData.getGeneralData("trajectory_0000")
    
    # Proved my rotation matrix works the same as the following:
    # from pyquaternion import Quaternion
    # attitude = Quaternion(trueQuat[i, :])
    # attitude.conjugate.rotate(trueAccel[i, :] + np.array([0., 0., -9.81])) \
    #                                                             + imuBiasAccel
