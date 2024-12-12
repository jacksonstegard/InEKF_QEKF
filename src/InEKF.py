# ===================================================================
#   Invariant Extended Kalman Filter Class
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   This class defines the required methods for the Invarient Extended 
#   Kalman Filter. A Right InEKF (World Error) model is used in 
#   propergation of states and correction of magneometer measurements.
#   The use of a Left InEKF (Body error) is used during correction 
#   steps for GPS measurements which is switched back to the Right InEKF
#   once updates are applied. Note that this class assumes the use of 
#   IMU measurements from a accelerometer and gyroscope as well as
#   potential other measurements that may or may not be included 
#   for the correction step in the Kalman filter from a GPS reciever 
#   and/or a Magneometer
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np
from scipy.linalg import expm
from scipy.linalg import sqrtm

# ---------------------------------
# Local Imports
# ---------------------------------
from utils import quat2Tuple, getSkew, adjoint, adjointInv

# ---------------------------------
# InEKF Class
# ---------------------------------
class InEKF:
    def __init__(self, initial_input):
       # Initialize state tuple
        self.x, self.theta = quat2Tuple(initial_input['x0'])
        
        # Swap states from quaternion to InEKF
        P0_Noise = self.swapCovStates(initial_input['P0Noise'])
        
        # Convert quaternion covaraince to InEKF covariance
        P_v =  np.diag(P0_Noise[3:6]) + getSkew(self.x[0:3,3]) @ np.diag(P0_Noise[0:3]) @ getSkew(self.x[0:3,3]).T # Convert quaternion vel error to invarient vel error
        P_p = np.diag(P0_Noise[6:9]) + getSkew(self.x[0:3,4]) @ np.diag(P0_Noise[0:3]) @ getSkew(self.x[0:3,4]).T # Convert quaternion pos error to invarient pos error
        
        # Organize sublocks of covariance
        P_theta = np.block([np.diag(P0_Noise[0:3]), np.zeros((3,12))])
        P_v = np.block([np.zeros((3,3)), P_v, np.zeros((3,9))])
        P_p = np.block([np.zeros((3,6)), P_p, np.zeros((3,6))])
        P_bias = np.block([np.zeros((6,9)), np.diag(P0_Noise[9:15])])
        
        # Define right invariant covariance
        self.P_r = np.block([[P_theta],[P_v],[P_p],[P_bias]])
        
        # Initialize process covariance
        self.Q_in = np.diag(self.swapCovStates(initial_input['w']))
        self.Q = self.Q_in.copy()
        
        # Initialize measurement covariance
        self.N_in_gps = np.diag(initial_input['v_gps'])
        self.N_gps = self.N_in_gps.copy()
        
        self.N_in_mag = np.diag(initial_input['v_mag'])
        self.N_mag = self.N_in_mag.copy()
        
        # Initialize mag measurement (assumed constant over flight and already normalized)
        self.m_w = initial_input['m_w']
        
        # Initialize time step, and gravity
        self.delta_t = initial_input['deltaT']
        self.gravity = initial_input['g']
        
        # Initialize initial measurment 
        self.u = np.zeros(6)
        self.y_gps = np.zeros(5)
        self.y_mag = np.zeros(5)
        
        # Initialize useful constants
        self.Z = np.zeros((3,3))
        self.I = np.eye(3)
        self.I_15 = np.eye(15)
        self.PI_Matrix = np.block([np.eye(3), np.zeros((3,2))]) # Define "PI" matrix to reduce demensions
        
    def swapCovStates(self, noise_in):
        # Convert array indices for noise vector (QEKF -> InEKF)
        # Make a copy of input noise
        noise = noise_in.copy()
        
        # Swap noise values
        noise[0:3] = noise_in[6:9]    # Swap in delta theta
        noise[6:9] = noise_in[0:3]    # Swap in position
        noise[9:12] = noise_in[12:15] # Swap in bias accel
        noise[12:15] = noise_in[9:12] # Swap in bias omega
        
        return noise
        
    def updateF(self):
        # Define constants
        I = self.I
        Z = self.Z
        delta_t = self.delta_t
        g = self.gravity
        
        # Get current states for linearization (requred for bias estimation)
        R = self.x[0:3,0:3]
        v = self.x[0:3,3]
        p = self.x[0:3,4]
        
        # Define euler approximated right invarient state transition matrix
        self.F = np.block([
                [I, Z, Z, -R * delta_t, Z],
                [getSkew(g) * delta_t, I, Z, -getSkew(v) @ R * delta_t, -R * delta_t],
                [Z, I * delta_t, I, -getSkew(p) @ R * delta_t, Z],
                [Z, Z, Z, I, Z],
                [Z, Z, Z ,Z, I]
                ])
    
    def updateQ(self):
        # Define right invarient process noise matrix
        adj = adjoint(self.x) # Don't need theta states
        # self.Q = self.F @ adj @ self.Q_in * self.delta_t @ adj.T @ self.F.T # Continous noise case
        self.Q = adj @ self.Q_in @ adj.T
        
    def getProcessModel(self):
        # Update nominal equations using euler intergration
        # Get prior (k-1) states
        R_prior = self.x[0:3,0:3]
        v_prior = self.x[0:3,3]
        p_prior = self.x[0:3,4]
        b_omega_prior = self.theta[0:3].ravel()
        b_a_prior = self.theta[3:6].ravel()
        
        # Get measurements (use quaternion order convention for input)
        a_m = self.u[0:3]
        omega_m = self.u[3:6]
        
        # Propagate discrete nominal states
        # R = R_prior @ (self.I + (omega_m - b_omega_prior) * self.delta_t)
        R = R_prior @  expm(getSkew(omega_m - b_omega_prior) * self.delta_t)
        v = v_prior + self.delta_t * (R_prior @ (self.I + 0.5 * self.delta_t * \
            getSkew(omega_m - b_omega_prior)) @ (a_m - b_a_prior) + self.gravity)
        p = p_prior + self.delta_t * v_prior + 0.5 * self.delta_t * self.delta_t * (R_prior @ \
            (self.I + (1/3) * self.delta_t * getSkew(omega_m - b_omega_prior)) @ (a_m - b_a_prior) + self.gravity)
        b_omega = b_omega_prior
        b_a = b_a_prior
        
        # Check R norm
        self.checkRNorm()
        
        # Update propagated state tuple
        self.x[0:3,0:3] = R
        self.x[0:3,3] = v
        self.x[0:3,4] = p
        self.theta[0:3] = b_omega.reshape(3,1)
        self.theta[3:6] = b_a.reshape(3,1)
        
    def checkRNorm(self):
        # Check the constaint det(R * R^T ) = 1
        # Get R
        R = self.x[0:3,0:3]
        
        # Get norm
        R_norm = (R @ R.T)
        
        # Check norm is approximately one
        if not(np.isclose(np.linalg.det(R_norm), 1, 1e-5)):
            # Update R
            self.x[0:3,0:3] = np.linalg.inv(sqrtm(R_norm)) @ R
        
    def propagation(self, u):  
        # Update input to process model
        self.u = u
        
        # Update F matrix linearized at x_{k-1} and u_{k-1}
        self.updateF()
        
        # Propagate nominal state equations with x_{k-1} and u_{k-1}
        self.getProcessModel()
        
        # Update process noise matrix
        self.updateQ()
        
        # Propagate covariance matrix
        self.P_r = self.F @ self.P_r @ self.F.T + self.Q
    
    def updateH(self, meas_type):
        if meas_type == 'gps':
            # Define reduced left invarient H matrix
            self.H = np.block([np.zeros((3,6)), np.eye(3), np.zeros((3,6))])
            
        elif meas_type == 'mag':
            self.H = np.block([getSkew(self.m_w), np.zeros((3,12))])
    
    def updateN(self, meas_type):
        # Get rotation matrix
        R = self.x[0:3,0:3]
        
        # Update depending on measurement being gps or mag
        if meas_type == 'gps':
            # Update reduced left invarient measurement noise matrix
            self.N_gps = R.T @ self.N_in_gps @ R
            
        elif meas_type == 'mag':
            # Update reduced right invarient measurement noise matrix
            self.N_mag = R @ self.N_in_mag @ R.T
    
    def correction(self, y, meas_type):        
        # Update measurment matrix
        self.updateH(meas_type)
        
        # Update measurement noise
        self.updateN(meas_type)
        
        if meas_type == 'gps':
            # Update the gps measurement
            self.y_gps = np.block([y,0,1]) # First three elements are position, then zero and 1
            
            # Convert right invarient covariance to left invarient covariance 
            adj_inv = adjointInv(self.x)
            P_l = adj_inv @ self.P_r @ adj_inv.T
            
            # Get S matrix for Kalman gain
            S = self.H @ P_l @ self.H.T + self.N_gps
            detS = np.linalg.det(S)
            
            # Check for singularity in S calculation and compute kalman gain K
            if np.isclose(detS, 0, 1e-5):
                # - Regularization ensures inverse exists
                K = P_l @ self.H.T @ np.linalg.inv(S + np.eye(S.shape[0]) * 1e-7)
            else:
                # - Traditional K calculation
                K = P_l @ self.H.T @ np.linalg.inv(S)
                
            # Split kalman gains
            K_x = K[0:9,:]
            K_theta = K[9:,:]
            
            # Get x inverse
            x_inv = np.linalg.inv(self.x)
            # x_inv = np.linalg.solve(self.x, np.eye(self.x.shape[0]))
            
            # Define x innovation and take skew of nine element vector
            x_vec_innovation = K_x @ self.PI_Matrix @ x_inv @ self.y_gps
            x_innovation = np.zeros((5,5))
            x_innovation[0:3,0:3] = getSkew(x_vec_innovation[0:3])
            x_innovation[0:3,3] = x_vec_innovation[3:6]
            x_innovation[0:3,4] = x_vec_innovation[6:9]
            
            # Define theta innovation
            theta_vec_innovation = K_theta @ self.PI_Matrix @ x_inv @ self.y_gps
            
            # Update state tuple
            self.x = self.x @ expm(x_innovation)
            self.theta = self.theta + theta_vec_innovation.reshape(6, 1)
            
            # Update left covaraince
            P_l = (self.I_15  - K @ self.H) @ P_l @ (self.I_15  - K @ self.H).T + K @ self.N_gps @ K.T
            
            # Convert left covaraince back to right covarince
            adj = adjoint(self.x)
            self.P_r = adj @ P_l @ adj.T
            
        elif meas_type == 'mag':
            # Normalize mag measurement
            y = y / np.linalg.norm(y)
            
            # Update the mag measurement
            self.y_mag = np.block([y,0,0])
            
            # Get S matrix for Kalman gain
            S = self.H @ self.P_r @ self.H.T + self.N_mag
            detS = np.linalg.det(S)
            
            # Check for singularity in S calculation and compute kalman gain K
            if np.isclose(detS, 0, 1e-5):
                # - Regularization ensures inverse exists
                K = self.P_r @ self.H.T @ np.linalg.inv(S + np.eye(S.shape[0]) * 1e-5)
            else:
                # - Traditional K calculation
                K = self.P_r @ self.H.T @ np.linalg.inv(S)
                
            # Split kalman gains
            K_x = K[0:9,:]
            K_theta = K[9:,:]
            
            # Define x innovation and take skew of nine element vector
            x_vec_innovation = K_x @ (self.PI_Matrix @ self.x @ self.y_mag - self.m_w)
            x_innovation = np.zeros((5,5))
            x_innovation[0:3,0:3] = getSkew(x_vec_innovation[0:3])
            x_innovation[0:3,3] = x_vec_innovation[3:6]
            x_innovation[0:3,4] = x_vec_innovation[6:9]
            
            # Define theta innovation
            theta_vec_innovation = K_theta @ (self.PI_Matrix @ self.x @ self.y_mag - self.m_w)
            
            # Update state tuple
            self.x = expm(x_innovation) @ self.x
            self.theta = self.theta + theta_vec_innovation.reshape(6, 1)
            
            # Update right covaraince
            self.P_r = (self.I_15  - K @ self.H) @ self.P_r @ (self.I_15  - K @ self.H).T + K @ self.N_mag @ K.T

        else:
            raise ValueError(f"Error: Unknown Measurement type named: {meas_type}!")
