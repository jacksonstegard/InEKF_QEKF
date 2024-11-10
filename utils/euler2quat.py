# ===================================================================
#   Convert Euler Angle to Quaternion
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   Convert three Euler angles for a zyx rotation to quaternion
# -------------------------------------------------------------------
#   Equation source:
#   https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# Method defined in "Quaternion kinematics for the error-state Kalman filter"
# def euler2quat(deltaTheta):
#     # Get norm
#     deltaThetaNorm = np.linalg.norm(deltaTheta)
    
#     # Protect against divide by zero error
#     if deltaThetaNorm < 1e-8:
#         deltaThetaNorm = 1e-8
    
#     # Get real part
#     deltaThetaReal = np.array([np.cos(deltaThetaNorm * 0.5)])
    
#     # Get imaginary part
#     deltaThetaImag = (deltaTheta / deltaThetaNorm) * np.sin(deltaThetaNorm * 0.5)
    
#     # Get delta quaternion
#     qDelta = np.block([deltaThetaReal,deltaThetaImag])

#     return qDelta

# Euler to quaternion alternate method
def euler2quat(deltaTheta):
    # Define angles
    roll = deltaTheta[0]
    pitch = deltaTheta[1]
    yaw = deltaTheta[2]

    # Compute half angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Calculate quaternion components
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


# Testing
# if __name__ == "__main__":
#     # Example inputs
#     omega = np.array([-0.00411792, -0.25057273,  0.0013741])
#     deltaT = 1
    
#     # Convert to quaternion
#     q1 = euler2quat(omega * deltaT)
    
#     # Manual change shows identical
#     # Get norm
#     omegaNorm = np.linalg.norm(omega)
    
#     # Get real part
#     qReal = np.array([np.cos(omegaNorm * deltaT * 0.5)])
    
#     # Get imaginary part
#     qImag = (omega / omegaNorm) * np.sin(omegaNorm * deltaT * 0.5)
    
#     # Get delta quaternion
#     q2 = np.block([qReal,qImag])
#     print(q2)