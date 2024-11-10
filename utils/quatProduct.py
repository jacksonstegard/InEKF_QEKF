# ===================================================================
#   Compute Quaternion Product
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   Compute quaternion product
# -------------------------------------------------------------------
#   Equation source:
#   Equation 13 in https://arxiv.org/abs/1711.02508
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# Quaternion product
def quatProduct(p,q):
    # Define first row (w component)
    row0 = p[0] * q[0] - p[1:4].T @ q[1:4]
    
    # Define last three rows (x,y,z components)
    row13 = p[0] * q[1:4] + q[0] * p[1:4] + np.cross(p[1:4],q[1:4])
    
    # Define quaternion product
    out = np.concatenate(([row0], row13))
    
    return out

if __name__ == "__main__":
    p = np.array([1, 0.5, 0.25, 0.1])
    q = np.array([0.3, 0.2, 0.1, 0.6])
    out = quatProduct(p, q)