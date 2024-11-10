# ===================================================================
#   Convert Quaternion to Rotation Matrix
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   Conversion of quaternion to rotation matrix (zyx)
# -------------------------------------------------------------------
#   Equation source:
#   Equation 115 in https://arxiv.org/abs/1711.02508
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# Quaternion to rotation matrix
def quat2rot(w,x,y,z):
    # Define squares
    wsqr = w*w
    xsqr = x*x
    ysqr = y*y
    zsqr = z*z
    
    # Define top row
    r11 = wsqr + xsqr - ysqr - zsqr
    r12 = 2 * (x * y - w * z)
    r13 = 2 * (x * z + w * y)
    
    # Define second row
    r21 = 2 * (x * y + w * z)
    r22 = wsqr - xsqr + ysqr - zsqr
    r23 = 2 * (y * z - w * x)
    
    # Define bottom row
    r31 = 2 * (x * z - w * y)
    r32 = 2 * (y * z + w * x)
    r33 = wsqr - xsqr - ysqr + zsqr
    
    # Define rotation matrix
    R = np.array([[r11, r12, r13],
                  [r21, r22, r23],
                  [r31, r32, r33]])
                  
    return R