# ===================================================================
#   Quaternion Extended Kalman Filter Class
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   This class defines the required methods for the Quaternion Extended 
#   Kalman Filter. Note that the attitude error is represented in the
#   world frame. Note that this class assumes the use of 
#   IMU measurements from a accelerometer and gyroscope as well as
#   potential other measurements that may or may not be included 
#   for the correction step in the Kalman filter from a GPS reciever 
#   and/or a Magneometer
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# ---------------------------------
# Local Imports
# ---------------------------------
from utils import quat2rot, quatProduct, euler2quat, getSkew

# ---------------------------------
# QEKF Class
# ---------------------------------
class QEKF:
    def __init__(self, initial_input):
       # Initialize state input 
        self.x = initial_input['x0']
        
        # Initialize state, process and measurement covariance
        self.P = np.diag(initial_input['P0Noise'])
        self.Q = np.diag(initial_input['w'])
        self.N_gps = np.diag(initial_input['v_gps'])
        self.N_mag = np.diag(initial_input['v_mag'])
        
        # Initialize time step and gravity
        self.delta_t = initial_input['deltaT']
        self.gravity = initial_input['g']
        
        # Initialize mag measurement (assumed constant over flight and already normalized)
        self.m_w = initial_input['m_w']
        
        # Initialize initial measurment 
        self.u = np.zeros(6)
        self.y_gps = np.zeros(15)
        self.y_mag = np.zeros(3)
        
        # Initialize useful constants
        self.I = np.eye(3)
        self.I_6 = np.eye(6)
        self.Z = np.zeros((3,3))
        self.Z_3_6 = np.zeros((3,6))
        self.Z_6_3 = np.zeros((6,3))
        self.Z_6 = np.zeros((6,6))
        self.Z_3_9 = np.zeros((3,9))
        self.Z_4_6 = np.zeros((4,6))
        self.Z_3_12 = np.zeros((3,12))

    def updateF(self):
        # Predefine subblocks
        # - Constants
        I = self.I
        Z = self.Z
        
        # - Define measured accel and estimated accel bias
        a_m = self.u[0:3]
        a_b = self.x[10:13]
        
        # - Define rotation matrix from quaternion
        R = quat2rot(self.x[6], self.x[7], self.x[8], self.x[9])
        
        # - Define indices: (3:5,6:8)
        r_35_68 = -getSkew(R @ (a_m - a_b)) * self.delta_t
        
        # Define F matrix from delta_x eqn
        self.F = np.block([
            [I, I * self.delta_t, Z, Z, Z],
            [Z, I, r_35_68, -R * self.delta_t, Z],
            [Z, Z, I, Z, -R * self.delta_t],
            [Z, Z, Z, I, Z],
            [Z, Z, Z, Z, I]
        ])
    
    def getProcessModel(self):
        # Update nominal equations using euler intergration and pertrubation update eqn for quaternion
        # Get prior (k-1) states
        p_prior = self.x[0:3]
        v_prior = self.x[3:6]
        q_prior = self.x[6:10]
        b_a_prior = self.x[10:13]
        b_omega_prior = self.x[13:16]
        R_prior = quat2rot(self.x[6], self.x[7], self.x[8], self.x[9])
        
        # Get input measurements
        a_m = self.u[0:3]
        omega_m = self.u[3:6]
        
        # Get delta quaternion (where q is still world <- body)
        omega_prior = omega_m - b_omega_prior
        delta_theta = omega_prior * self.delta_t
        delta_q = euler2quat(delta_theta)
        
        # Propagate discrete nominal states
        p = p_prior + self.delta_t * v_prior + 0.5 * self.delta_t * self.delta_t * (R_prior @ \
            (self.I + (1/3) * self.delta_t * getSkew(omega_m - b_omega_prior)) @ (a_m - b_a_prior) + self.gravity)
        v = v_prior + self.delta_t * (R_prior @ (self.I + 0.5 * self.delta_t * \
            getSkew(omega_m - b_omega_prior)) @ (a_m - b_a_prior) + self.gravity)
        q = quatProduct(q_prior, delta_q)
        b_a = b_a_prior
        b_omega = b_omega_prior
        
        # Define propagated state
        self.x = np.block([p,v,q,b_a,b_omega])
        
    def checkQNorm(self):
        # Check if quaternion is still a unit quaternion
        # Define q
        q = self.x[6:10]
        
        # Define norm of q
        q_norm = np.linalg.norm(q)
        
        # Check if tolerance isn't meet for q_norm
        if not(np.isclose(q_norm, 1, 1e-5)):
            # Update q
            self.x[6:10] = q / q_norm
        
    def propagation(self, u):  
        # Update input to process model
        self.u = u
        
        # Update F matrix linearized at x_{k-1} and u_{k-1}
        self.updateF()
        
        # Propagate nominal state equations with x_{k-1} and u_{k-1}
        self.getProcessModel()
        
        # Propagate covariance matrix
        self.P = self.F @ self.P @ self.F.T + self.Q
    
        # Check for unit quaternion
        self.checkQNorm()

    def updateH(self, meas_type):
        # H = H_x * X_delta x
        
        if meas_type == 'gps':
            # Define H for position and velocity measurement
            # H = np.block([
            #     [I, np.zeros((3,12))],
            #     [np.zeros((3,3)), I, np.zeros((3,9))],
            #     [np.zeros((9,15))]
            #     ])
            
            # Define H for position measurement
            self.H = np.block([
                [self.I, np.zeros((3,12))],
                ])
            
        elif meas_type == 'mag':
            # Define the partial dh/dq (h(\hat{x}) = R(q)^T m_w))
            # - Define matrix constants
            Z = self.Z
            Z_3_9 = self.Z_3_9
            Z_4_6 = self.Z_4_6
            Z_3_12 = self.Z_3_12
            
            # - Define q components
            q = self.x[6:10]
            q_w = q[0]
            q_x = q[1]
            q_y = q[2]
            q_z = q[3]
            
            # - Define m_w components
            m_x = self.m_w[0]
            m_y = self.m_w[1]
            m_z = self.m_w[2]
            
            # - Define dh/dq 
            dh_dq = np.array([[2*m_x*q_w + 2*m_y*q_z - 2*m_z*q_y, 2*m_x*q_x + 2*m_y*q_y + 2*m_z*q_z, -2*m_x*q_y + 2*m_y*q_x - 2*m_z*q_w, -2*m_x*q_z + 2*m_y*q_w + 2*m_z*q_x],
                      [-2*m_x*q_z + 2*m_y*q_w + 2*m_z*q_x, 2*m_x*q_y - 2*m_y*q_x + 2*m_z*q_w, 2*m_x*q_x + 2*m_y*q_y + 2*m_z*q_z, -2*m_x*q_w - 2*m_y*q_z + 2*m_z*q_y],
                      [2*m_x*q_y - 2*m_y*q_x + 2*m_z*q_w, 2*m_x*q_z - 2*m_y*q_w - 2*m_z*q_x, 2*m_x*q_w + 2*m_y*q_z - 2*m_z*q_y, 2*m_x*q_x + 2*m_y*q_y + 2*m_z*q_z]])
            
            # Define H_x for magneometer measurement
            H_x = np.block([self.Z_3_6, dh_dq, self.Z_3_6])
            
            # Define Q_{\delta \theta}
            Q_delta_theta = 0.5 * np.array([[-q_x, -q_y, -q_z],
                                           [q_w, q_z, -q_y],
                                           [-q_z, q_w, q_x],
                                           [q_y, -q_x, q_w]])
            
            # Define X_{\delta x}
            X_delta_x = np.block([[self.I, Z_3_12],
                                  [Z, self.I, Z_3_9],
                                  [Z_4_6, Q_delta_theta, Z_4_6],
                                  [Z_3_9, self.I, Z],
                                  [Z_3_12 , self.I]])
            
            # Define H
            self.H = H_x @ X_delta_x

    def updateG(self, delta_x):
        # Define reset matrix
        # Define constant matrices
        I = self.I
        I_6 = self.I_6
        Z_6_3 = self.Z_6_3
        Z_6 = self.Z_6 

        # Get delta_theta from delta_x update
        delta_theta = delta_x[6:9]
        
        # Define G
        self.G = np.block([
            [I_6, Z_6, Z_6_3],
            [Z_6_3.T, I + getSkew(0.5 * delta_theta), Z_6_3.T],
            [Z_6, Z_6_3, I_6]
            ])
    
    def correction(self, y, meas_type):        
        # Update H measurment matrix
        self.updateH(meas_type)
        
        # GPS measurement setup
        if meas_type == 'gps': 
            # GPS measurement covaranice matrix
            N = self.N_gps
            
            # Position estimate
            y_hat = self.x[0:3]
            
            # Meaurement
            self.y_gps = y
           
        # Magneomter measurement setup
        elif meas_type == 'mag':
            # Magneometer measurement covariance matrix
            N = self.N_mag 
            
            # Magnetometer body frame estiamte
            R = quat2rot(self.x[6], self.x[7], self.x[8], self.x[9])
            y_hat = R.T @ self.m_w
            
            # Magneometer measurement
            y = y / np.linalg.norm(y)
            self.y_mag = y
            
        else:
            raise ValueError(f"Error: Unknown Measurement type named: {meas_type}!")
            
        # Get S matrix for Kalman gain
        S = self.H @ self.P @ self.H.T + N
        detS = np.linalg.det(S)
        
        # Check for singularity in S calculation and compute kalman gain K
        if np.isclose(detS,0,1e-5):
            # - Regularization ensures inverse exists
            K = self.P @ self.H.T @ np.linalg.inv(S + np.eye(S.shape[0]) * 1e-7)
        else:
            # - Traditional K calculation
            K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Get error state
        delta_x = K @ (y - y_hat)
        
        # Correct error state covariance
        I = np.eye(15)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ N @ K.T
    
        # Correct state estimate
        # - Update position and velocity states
        self.x[0:6] = self.x[0:6] + delta_x[0:6]
        
        # - Update quaternion state
        q = self.x[6:10]
        delta_theta = delta_x[6:9]
        delta_q = euler2quat(delta_theta)
        self.x[6:10] = quatProduct(delta_q,q) # Global error definition
        
        # - Update bias states
        self.x[10:] = self.x[10:] + delta_x[9:]
        
        # Check for unit quaternion
        self.checkQNorm()
        
        # Apply reset
        # - delta_x = 0
        # - Update covaraince matrix
        self.updateG(delta_x)
        self.P = self.G @ self.P @ self.G.T
    