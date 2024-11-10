# ===================================================================
#   Convert Rotation Matrix to Euler Angles
# ===================================================================
#   Author: Jack Sonstegard
# -------------------------------------------------------------------
#   Description:
#   Conversion of rotation matrix (zyx) to euler angles
# -------------------------------------------------------------------
#   Code based off:
#   https://github.com/UMich-CURLY-teaching/UMich-ROB-530-public/blob/main/code-examples/Python/matrix_groups/lib.py
# ===================================================================

# ---------------------------------
# External Libraries
# ---------------------------------
import numpy as np

# Convert rotation matrix to euler angles
def rot2Euler(R):
    # If R is 3x3, convert it to 1x3x3 for uniform processing
    if R.shape == (3, 3):
        R = np.expand_dims(R, axis=0)  # Add a new dimension, making it 1x3x3
    
    # Calculate yaw (Z-axis rotation)
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])

    # Calculate pitch (Y-axis rotation)
    pitch = np.arctan2(-R[:, 2, 0], R[:, 0, 0] * np.cos(yaw) + R[:, 1, 0] * np.sin(yaw))

    # Calculate roll (X-axis rotation)
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])

    # Combine the results into a single array of shape (n, 3)
    euler_angles = np.stack([roll, pitch, yaw], axis=0)
    
    # If the input was 3x3, return a 1D array instead of a 2D array
    if euler_angles.shape[0] == 1:
        return euler_angles[0]
    
    return euler_angles


# Testing
if __name__ == "__main__":
    # Create example R
    R = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    
    # Call function
    E = rot2Euler(R)
    
    print(E)
    
    # Create 2x3x3 R
    RR = np.array([R, R])
    
    # Check rot2Euler still works with RR having extra demension
    E = rot2Euler(RR)
    print(E)
    
    
    
    
