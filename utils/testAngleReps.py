# ===================================================================
#   Test Euler Angle, Quaternion, and Rotation Matrix Conversions
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   Test functions that convert between euler angles and representations
#   of rotation
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# ---------------------------------
# Local Imports
# ---------------------------------
from euler2quat import euler2quat
from quat2euler import quat2euler
from quat2rot import quat2rot
from rot2Euler import rot2Euler

# Euler to rotation matrix with zyx sequence
def euler_to_rotation_matrix_zyx(roll, pitch, yaw):
    # Rotation around Z-axis (yaw)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Rotation around Y-axis (pitch)
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation around X-axis (roll)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Combine rotations: R = R_z * R_y * R_x
    R = R_z @ R_y @ R_x

    return R

# Main Test
if __name__ == "__main__":
    # Create example euler angles
    eulerAngles = np.array([0.9,-1.56,1.58]) # Needs to be between -pi/2 and pi/2
    
    # Euler -> Quaternion  -> Euler
    quaternionMine = euler2quat(eulerAngles)
    eulerFromQMine = quat2euler(quaternionMine[0],quaternionMine[1],quaternionMine[2],quaternionMine[3])
    
    # - Euler -> Rotation -> Euler
    RMine = euler_to_rotation_matrix_zyx(eulerAngles[0],eulerAngles[1],eulerAngles[2])
    eulerFromRMine2 =  rot2Euler(RMine)
    
    # - Euler -> Quaternion -> Rotation Matrix -> Euler (Mine)
    RFromQMine = quat2rot(quaternionMine[0],quaternionMine[1],quaternionMine[2],quaternionMine[3])
    eulerFromQandRMine2 = rot2Euler(RFromQMine)
    

    
    #########################
    ######### OLD ###########
    #########################
    # Modules
    # from scipy.spatial.transform import Rotation as R
    
    # # Test with known working function
    # # - Euler -> Quaternion  -> Euler
    # r = R.from_euler('zyx', eulerAngles, degrees=False)
    # quaternionKnown = r.as_quat()
    # eulerFromQKnown = euler2quat([quaternionKnown[1], quaternionKnown[2], quaternionKnown[3], quaternionKnown[0]])
    
    # quaternionMine = euler2quat(eulerAngles)
    # eulerFromQMine = quat2euler(quaternionMine[0],quaternionMine[1],quaternionMine[2],quaternionMine[3])
    
    # # - Euler -> Rotation -> Euler
    # r = R.from_euler('zyx', eulerAngles, degrees=False)
    # RKnown = r.as_matrix() 
    # eulerFromRKnown =  rot2Euler(RKnown)
    # eulerFromRKnown2 =  R2ypr(RKnown)
    
    # RMine = euler_to_rotation_matrix_zyx(eulerAngles[2],eulerAngles[1],eulerAngles[0])
    # eulerFromRMine = rot2Euler(RMine)
    # eulerFromRMine2 =  R2ypr(RMine)
    
    # # - Euler -> Quaternion -> Rotation Matrix -> Euler (Mine)
    # RFromQMine = quat2rot(quaternionMine[0],quaternionMine[1],quaternionMine[2],quaternionMine[3])
    # eulerFromQandRMine = rot2Euler(RFromQMine)
    # eulerFromQandRMine2 = R2ypr(RFromQMine)
    
    # # - Euler -> Quaternion -> Rotation Matrix -> Euler (Known)
    # rr = R.from_euler('zyx', eulerAngles, degrees=False)
    # r2 = rr.as_quat()
    # r3 = R.from_quat(r2)
    # r4 = r3.as_matrix()
    # r5 = R.from_matrix(r4)
    # r6 = r5.as_euler('zyx', degrees=False)
    
    ########################
    # # Test my quaternion functions
    # quaternionMine = euler2quat(eulerAngles)
    
    # eulerFromQMine = quat2euler(quaternionMine[0],quaternionMine[1],quaternionMine[2],quaternionMine[3])
    
    # RFromQMine = quat2rot(quaternionMine[0],quaternionMine[1],quaternionMine[2],quaternionMine[3])
    
    # # Test my rotation matrix functions
    # eulerFromRKnown = rot2Euler(RKnown)
    # eulerFromRMine = rot2Euler(RFromQMine)
    
    # # Is q->euler the same as R->euler?
    # eulerFromQ = quat2euler(quaternionMine[0],quaternionMine[1],quaternionMine[2],quaternionMine[3]) # Issue must be in euler2quat?
    # eulerFromR = rot2Euler(RKnown) # Just show this inversion can be done
    # eulerFromOtherR = rotation_angles(RKnown, order='zyx')
    
    # aa = euler_to_quaternion(eulerAngles[0],eulerAngles[1],eulerAngles[2])
    # bb = quat2euler(aa[0],aa[1],aa[2],aa[3])
    
    # r = R.from_quat(np.array([quaternionKnown[1], quaternionKnown[2], quaternionKnown[3], quaternionKnown[0]]))
    # eulerKnown = r.as_euler('zyx', degrees=False)
    
    # r = R.from_matrix(RKnown)
    # eulerFromRKnown = r.as_euler('zyx', degrees=False)