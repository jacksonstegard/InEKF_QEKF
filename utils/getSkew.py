# ===================================================================
#   Generate Skew Symeteric Matrix
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   Convert three element vector to skew symetric matrix
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# Get skew matrix function
def getSkew(var):
    # Define skew-symmetric matrix
    # Assume var is a 3 by 1 array
    out = np.array([[0      , -var[2], var[1]],
                    [var[2] , 0      , -var[0]],
                    [-var[1], var[0] , 0     ]])
    return out
    