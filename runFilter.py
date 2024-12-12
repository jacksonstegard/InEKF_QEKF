# ===================================================================
#   Invariant/Quaternion Extended Kalman Filter Comparison Simulation 
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   This script runs the QEKF and/or InEKF filter(s) on sensor data 
#   emulator sensor data and truth data from the MAD dataset. Monte
#   Carlo simulations are performed to compare filters
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# ---------------------------------
# Local Imports
# ---------------------------------
from data import dataLoader
from src import InEKF, QEKF, postProcessKF
from utils import quat2euler, euler2quat, quat2Tuple

# ---------------------------------
# Main Execution Block
# ---------------------------------
if __name__ == "__main__":
    # ---------------------------------
    # Simulation Setup
    # ---------------------------------
    # Filter selection
    runFilters = {
        'QEKF' : True,
        'InEKF': True
    }
    
    # GPS setup
    generateGps = 1
    gpsPosSigma = 1
    
    # Magnetometer setup
    generateMag = 1
    magSample = np.array([18323, 4947, 49346]) # Redmond Washington sample location
    magSample = magSample / np.linalg.norm(magSample) # Normalize mag sample
    magSigma = 0.05 # nT (normalized)
    magSampleRate = 10 # Hz
    
    # IMU setup
    generateIMU = 1
    sigmaAccel = 5e-4
    sigmaOmega = 1e-4
    sigmaAccelBias = 6e-4
    sigmaOmegaBias = 3e-5
    
    # Monte Carlo setup
    numMCs = 0 # Zero Monte Carlo will create no randomization of initial conditions
    mcsInitPosBounds = 15
    mcsInitVelBounds = 5
    mcsInitAngBounds = 15
    mcsInitAccelBiasBounds = 5e-5
    mcsInitGyroBiasBounds = 5e-6
    
    # Plotting setup
    plotCovaraince = False
    initTime = 10 # Initial time to plot up to for convergence plots
    
    # Initial constants
    np.random.seed(1)
    deltaT = 0.01
    gravity = np.array([0,0,9.8])
    
    # ---------------------------------
    # Load Data
    # ---------------------------------
    # Setup data extraction
    filename = "data/sensor_records_sunny.hdf5"
    trajectoryNames = ["trajectory_0002"]
    # trajectoryNames = [f"trajectory_{i:04d}" for i in range(30)]
    groups = ["gps", "groundtruth", "imu"] 
    
    # Get data
    madData = dataLoader(filename, trajectoryNames, groups)
    madData.loadData()
    data = madData.data
    
    # ---------------------------------
    # Loop Over Trajectories
    # ---------------------------------
    for trajectoryName in trajectoryNames:   
        
        # ---------------------------------
        # Get Truth and Sensor Data
        # ---------------------------------
        # Load measurement and truth data
        gps, imu, truth = madData.getGeneralData(trajectoryName)
        
        # Optionally generate gps data
        if generateGps:
            gps = madData.generateGPS(trajectoryName, posSigma = gpsPosSigma)
        
        # Optionally generate imu data
        if generateIMU:
            oneArray = np.ones(3)
            imu = madData.generateIMU(trajectoryName, noiseAccel=sigmaAccel*oneArray, noiseGyro=sigmaOmega*oneArray,
                                      noiseBiasAcc=sigmaAccelBias*oneArray, noiseBiasGyro=sigmaOmegaBias*oneArray)
        
        # Optionally generate mag data
        if generateMag:
            # Get mag data
            mag = madData.generateMag(trajectoryName, magSampleRate, magSample, magSigma)
            
            # Collect mag sampling data
            deltaTMag = 1 / magSampleRate # (1/Hz) -> sec
            numSamplesMag = len(mag['intensity'])
            tMag = np.arange(0, numSamplesMag * deltaTMag, deltaTMag)
            
        # Get sampling data for imu and gps
        deltaT = 1 / data[trajectoryName]["imu"]["accelerometer"]["metaData"]["sampling_frequency"] # (1/Hz) -> sec
        deltaTGPS = 1 / data[trajectoryName]["gps"]["position"]["metaData"]["sampling_frequency"] # (1/Hz) -> sec
        numSamplesIMU = len(imu['accel'])
        numSamplesGPS = len(gps['position'])
        tIMU = np.arange(0, numSamplesIMU * deltaT, deltaT)
        tGPS = np.arange(0, numSamplesGPS * deltaTGPS, deltaTGPS)
                
        # ---------------------------------
        # Initialize States and Covariances
        # ---------------------------------
        # Initialize covariance
        P0Noise = np.array([5, 5, 5, 
                      2, 2, 2, 
                      0.7, 0.7, 0.7, 
                      1e-3, 1e-3, 1e-3, 
                      1e-3, 1e-3, 1e-3])**2
        P0 = np.diag(P0Noise)
        
        # Define process noise
        w = np.array([0.0, 0.0, 0.0, 
                     sigmaAccel**2, sigmaAccel**2, sigmaAccel**2,
                     sigmaOmega**2, sigmaOmega**2, sigmaOmega**2,
                     sigmaAccelBias**2, sigmaAccelBias**2, sigmaAccelBias**2,
                     sigmaOmegaBias**2, sigmaOmegaBias**2, sigmaOmegaBias**2])
        
        # Define measurement noise
        vGPS = np.full(3, gpsPosSigma**2)
        vMag = np.full(3, magSigma**2)
        
        # Get true initial conditions
        p0True = truth['position'][0]
        v0True = truth['velocity'][0]
        q0True = truth['quaternion'][0]
        # R0True = quat2rot(q0True[0], q0True[1], q0True[2], q0True[3])
        ba0True = imu['accelInitBias']
        bg0True = imu['gyroInitBias']
        
        # Initialize arrays to hold randomized initial conditions for each Monte Carlo simulation
        if numMCs > 0:
            p0 = np.zeros((numMCs, 3))
            v0 = np.zeros((numMCs, 3))
            q0 = np.zeros((numMCs, 4))
            # R0 = np.zeros(((numMCs, 3,3)))
            ba0 = np.zeros((numMCs, 3))
            bg0 = np.zeros((numMCs, 3))
        else:
            p0 = p0True
            v0 = v0True
            q0 = q0True
            # R0 = R0True
            ba0 = ba0True
            bg0 = bg0True
            
        # Define Monte Carlo initial conditions if randomized Monte Carlos are used
        for i in range(numMCs):
            # Randomize initial position, velocity, quaternion and biases relative to true initial condition
            p0[i] = p0True * np.ones(3) + np.random.uniform(-mcsInitPosBounds,mcsInitPosBounds,3)
            v0[i] = v0True * np.ones(3) + np.random.uniform(-mcsInitVelBounds,mcsInitVelBounds,3)
            theta0 = quat2euler(q0True[0], q0True[1], q0True[2], q0True[3])
            q0[i] = euler2quat(np.random.uniform(-mcsInitAngBounds, mcsInitAngBounds, 3) \
                            * np.pi / 180 + theta0.ravel()).ravel()
            # R0[i] = quat2rot(q0[i,0], q0[i,1], q0[i,2], q0[i,3])
            ba0[i] = ba0True * np.ones(3) + np.random.uniform(-mcsInitAccelBiasBounds,mcsInitAccelBiasBounds,3)
            bg0[i] = bg0True * np.ones(3) + np.random.uniform(-mcsInitGyroBiasBounds,mcsInitGyroBiasBounds,3)
                
        # ---------------------------------
        # Loop Over Filters and Monte Carlos
        # ---------------------------------
        for filterName, runFilter in runFilters.items():
            
            if runFilter:
                # Loop over Monte Carlos
                allX, allTheta, allP = [], [], []

                # Loop over Monte Carlo
                for numMonte in range(max(numMCs, 1)):
                    
                    # Initialize state and input using truth
                    if numMCs > 0:
                        x0 = np.concatenate((p0[numMonte], v0[numMonte], q0[numMonte], ba0[numMonte], bg0[numMonte]))
                    else:
                        x0 = np.concatenate((np.array([0,0,0]), v0, q0, ba0, bg0))
                    
                    # Define initial inputs
                    initialInputs = {'x0': x0, 'P0Noise': P0Noise, 'w': w,'v_gps': vGPS, \
                                     'v_mag': vMag, 'm_w': magSample, 'deltaT': deltaT, 'g': gravity}
                    
                    # Pre-allocate state
                    if filterName == 'QEKF':
                        # Initiallize state vector and add initial values
                        x = np.zeros((16, len(tIMU) - 1))
                        x = np.hstack([x0.reshape(-1, 1), x])
                        
                    elif filterName == 'InEKF':
                        # Get initial conditions for state tuple
                        x0InEKF, theta0InEKF = quat2Tuple(initialInputs['x0'])
                        
                        # Initialize state tuple and add initial values
                        xState = np.zeros((len(tIMU) - 1, 5, 5))
                        xState = np.concatenate([x0InEKF.reshape(-1,5,5), xState], axis=0)
                        thetaState = np.zeros((6, len(tIMU) - 1))
                        thetaState = np.hstack([theta0InEKF.reshape(-1, 1), thetaState])
                        x = {'xState': xState, 'thetaState': thetaState}
                    
                    # Pre-allocate covaraince
                    P = np.zeros((len(tIMU) - 1, 15, 15))
                    P = np.concatenate([P0.reshape(-1,15,15), P], axis=0)
            
                    # Define filter object with initial inputs
                    if filterName == 'QEKF':
                        # Initiallize quaternion kalman filter
                        filt = QEKF(initialInputs)
                        
                    elif filterName == 'InEKF':
                        # Initialize invarient Kalman filter
                        filt = InEKF(initialInputs)
                    
                    else:
                        raise ValueError("Error: Unknown FilterName Entered!")
                        
                    # ---------------------------------
                    # Run Filter On Sensor Data
                    # ---------------------------------
                    for i in range(len(tIMU)-1):
                        
                        # Get input from imu
                        u = np.block([imu['accel'][i], imu['gyro'][i]])
                        
                        # Propagate state estimate
                        filt.propagation(u)
                        
                        # Check if Magneometer measurement is received and is generated (only check if generated for mag)
                        if generateMag:
                            iMag = int(np.floor(i*deltaT / deltaTMag))
                            if np.isclose(tIMU[i], tMag[iMag]):
                                # Get measurement from gps
                                y = mag['intensity'][iMag]
                                
                                # Correct state estimate
                                filt.correction(y, 'mag')
                        
                        # Check if GPS measurement is received
                        iGPS = int(np.floor(i*deltaT / deltaTGPS))
                        if np.isclose(tIMU[i], tGPS[iGPS]):
                            # Get measurement from gps
                            y = gps['position'][iGPS]
                            
                            # Correct state estimate
                            filt.correction(y, 'gps')
                        
                        # Collect state and covaraince
                        if filterName == 'QEKF':
                            x[:,i + 1] = filt.x
                            P[i + 1,:,:] = filt.P
                            
                        elif filterName == 'InEKF':
                            x['xState'][i + 1,:,:] = filt.x
                            x['thetaState'][:,i + 1] = filt.theta.ravel()
                            P[i + 1,:,:] = filt.P_r
            
                    # Append state and covaraince for Monte Carlo runs
                    allX.append(x)
                    allP.append(P)

                # ---------------------------------
                # Post Process Filter Results
                # ---------------------------------
                if numMCs == 0:
                    # Post process results for one nonrandomized Monte Carlo
                    # Define post processing object
                    ppKF = postProcessKF(x, tIMU, tGPS, imu, gps, truth, trajectoryName, filterName, numMCs, P = P)
                    
                    # Plot states and debug plots over entire trajectory
                    ppKF.plotTrajStates(plotCov=plotCovaraince)
                    ppKF.plotDebug()
                    
                    # Print errors
                    ppKF.printResults()
                    
                else:
                    # Convert list to numpy array
                    allX = np.array(allX)
                    allP = np.array(allP)
                    
                    # Initialize post processing object
                    ppKF = postProcessKF(allX, tIMU, tGPS, imu, gps, truth, trajectoryName, filterName, numMCs, allP)
                    
                    # Plots states over entire trajectory
                    ppKF.plotTrajStates()
            
                    # Plot states over initial time interval
                    ppKF.plotInitialTime(initTime)
            
                    # Print errors
                    ppKF.printResults()
