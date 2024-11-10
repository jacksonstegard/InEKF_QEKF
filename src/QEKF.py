# ===================================================================
#   Quaternion Extended Kalman Filter Class
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   This class defines the required methods for the Quaternion Extended 
#   Kalman Filter. Note that the attitude error is represented in the
#   world frame. The class assumes the use of accelerometer and gyroscope
#   inputs as well as GPS inputs to estimate the state of a drone based
#   off data collected in the MAD dataset.
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
        self.N = np.diag(initial_input['v'])
        
        # Initialize time step and gravity
        self.delta_t = initial_input['deltaT']
        self.gravity = initial_input['g']
        
        # Initialize initial measurment 
        self.u = np.zeros(6)
        self.y = np.zeros(15)
        
        # Initialize useful constants
        self.I = np.eye(3)

    def getF(self):
        # Predefine subblocks
        # - Define identity
        I = self.I
        
        # - Define zeros matrix
        Z = np.zeros((3,3))
        
        # - Define measured accel and estimated accel bias
        a_m = self.u[0:3]
        a_b = self.x[10:13]
        
        # - Define rotation matrix from quaternion
        R = quat2rot(self.x[6], self.x[7], self.x[8], self.x[9])
        
        # - Define indices: (3:5,6:8)
        r_35_68 = -getSkew(R @ (a_m - a_b)) * self.delta_t
        
        # Define F matrix from delta_x eqn
        F = np.block([
            [I, I * self.delta_t, Z, Z, Z],
            [Z, I, r_35_68, -R * self.delta_t, Z],
            [Z, Z, I, Z, -R * self.delta_t],
            [Z, Z, Z, I, Z],
            [Z, Z, Z, Z, I]
        ])
        
        return F
    
    def getProcessModel(self):
        # Update nominal equations using euler intergration and pertrubation update eqn for quaternion
        # Get prior (k-1) states
        p_prior = self.x[0:3]
        v_prior = self.x[3:6]
        q_prior = self.x[6:10]
        b_a_prior = self.x[10:13]
        b_omega_prior = self.x[13:16]
        R_prior = quat2rot(self.x[6], self.x[7], self.x[8], self.x[9])
        
        # Get measurements
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
        
    def checkQNorm(self, x):
        # Check if quaternion is still a unit quaternion
        # Define q
        q = x[6:10]
        
        # Define norm of q
        q_norm = np.linalg.norm(q)
        
        # Check tolerance isn't meet for q_norm
        if not(np.isclose(q_norm, 1, 1e-5)):
            x[6:10] = q / q_norm
            
        return x
        
    def propagation(self, u):  
        # Update input to process model
        self.u = u
        
        # Get F matrix linearized at x_{k-1} and u_{k-1}
        F = self.getF()
        
        # Propagate nominal state equations with x_{k-1} and u_{k-1}
        self.getProcessModel()
        
        # Propagate covariance matrix
        self.P = F @ self.P @ F.T + self.Q
    
        # Check for unit quaternion
        self.x = self.checkQNorm(self.x)

    def getH(self):
        # H = H_x * X_delta x
        # Note H equation can be simplified of simple gps model
        # Define identity
        I = self.I
        
        # Define H for position and velocity
        # H = np.block([
        #     [I, np.zeros((3,12))],
        #     [np.zeros((3,3)), I, np.zeros((3,9))],
        #     [np.zeros((9,15))]
        #     ])
        
        # Define H for position measurement
        H = np.block([
            [I, np.zeros((3,12))],
            [np.zeros((12,15))]
            ])
        
        return H

    def getMeasurementModel(self):
        # Define estimate of state
        y_hat = np.block([self.x[0:6],np.zeros(9)])
        
        return y_hat

    def getG(self, delta_x):
        # Define reset matrix
        # Define identity and zeros matrices
        I3 = self.I
        I6 = np.eye(6)
        Z63 = np.zeros((6,3))
        Z6 = np.zeros((6,6))
        
        # Get delta_theta from delta_x update
        delta_theta = delta_x[6:9]
        
        # Define G
        G = np.block([
            [I6, Z6, Z63],
            [Z63.T, I3 + getSkew(0.5 * delta_theta), Z63.T],
            [Z6, Z63, I6]
            ])
    
        return G
    
    def correction(self, y):
        # Update measurement y
        self.y = y
        
        # Get measurment matrix
        H = self.getH()
        
        # Get S matrix for Kalman gain
        S = H @ self.P @ H.T + self.N
        detS = np.linalg.det(S)
        
        # Check for singularity in S calculation and compute kalman gain K
        if np.isclose(detS,0,1e-5):
            # - Regularization ensures inverse exists
            K = self.P @ H.T @ np.linalg.inv(S + np.eye(15) * 1e-7)
        else:
            # - Traditional K calculation
            K = self.P @ H.T @ np.linalg.inv(S)
        
        # Get error state
        delta_x = K @ (self.y - self.getMeasurementModel())
        
        # Correct error state covariance
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.N @ K.T
        
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
        self.x = self.checkQNorm(self.x)
        
        # Apply reset
        # - delta_x = 0
        # - Update covaraince matrix
        G = self.getG(delta_x)
        self.P = G @ self.P @ G.T
    