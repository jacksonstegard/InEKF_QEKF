# ===================================================================
#   Compute State Tuple for Invarient Extended Kalman Filter
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   Convert quaternion state vector to state tuple for InEKF
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# ---------------------------------
# Local Imports
# ---------------------------------
try:
    from utils.quat2rot import quat2rot
except ModuleNotFoundError:
    from quat2rot import quat2rot

# Convert quaternion state vector to InEKF state tuple
def quat2Tuple(xVec):
    # Define individual states
    p = xVec[0:3].reshape(3, 1)  # Reshape to (3,1)
    v = xVec[3:6].reshape(3, 1)  # Reshape to (3,1)
    q = xVec[6:10]
    b_a = xVec[10:13].reshape(3, 1)     # Reshape to (3,1)   
    b_omega = xVec[13:16].reshape(3, 1) # Reshape to (3,1)
    R = quat2rot(q[0], q[1], q[2], q[3])
    
    # Define SE_2(3) matrix x
    Z = np.zeros((1,3))
    x = np.block([
        [R, v, p],
        [Z, 1, 0],
        [Z, 0, 1]
        ])
    
    # Define vector state
    theta = np.block([
            [b_omega],
            [b_a]
            ])
    
    # Return state tuple
    return x, theta


# Testing
if __name__ == "__main__":
    # Define a state
    x = np.array([1,2,3,4,5,6,0.1,0.2,0.3,0.4,7,8,9,10,11,12])
    
    # Test function
    xOut, theta = quat2Tuple(x)