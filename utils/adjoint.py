# ===================================================================
#   Define Adjoint
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   Get the adjoint for the state tuple (SE_2(3) and 6 element vector)
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# ---------------------------------
# Local Imports
# ---------------------------------
try:
    from utils.getSkew import getSkew
except ModuleNotFoundError:
    from getSkew import getSkew

# Adjoint function
def adjoint(x):
    # Get states (R,v,p) from state tuple
    R = x[0:3,0:3] 
    v = x[0:3,3]
    p = x[0:3,4]
    
    # Predefine zeros matrix
    Z = np.zeros((3,3))
    
    # Define individual parts of adjoint with states (R,v,p) and biases (b_omega, b_a)
    x_adj = np.block([
            [R, Z, Z],
            [getSkew(v) @ R, R, Z],
            [getSkew(p) @ R, Z, R]
            ])
    theta_adj = np.eye(6)
    
    # Define state tuple adjoint
    adj = np.block([
            [x_adj, np.zeros((9,6))],
            [np.zeros((6,9)), theta_adj]
            ])
    
    return adj


# Testing 
if __name__ == "__main__":
    # Define a state
    x = np.arange(1, 9*9+1).reshape(9, 9)

    # Test function
    adj = adjoint(x)