# ===================================================================
#   Convert Quaternion to Euler Angles
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   Conversions of quaternion to euler angles. Assumes a euler zyx
#   rotation.
# -------------------------------------------------------------------
#   Source of code:
#   https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation
# -------------------------------------------------------------------
#   Equation source:
#   https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# Quaternion to Euler angles
def quat2euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)
    
    euler = np.concatenate((X.reshape(-1,1),Y.reshape(-1,1),Z.reshape(-1,1)), axis=1)
    
    return euler