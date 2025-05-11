import numpy as np
from py_attainability.lie import Pose2, Twist2
def calc_poses(base_pose, mat_segment_twists):
    """
    Given a base pose g_0 and N semgent twists along a rod, compute the 
    N+1 poses between/at the end of each segment
    """
    N_twists = mat_segment_twists.shape[1]
    N_poses = N_twists + 1
    ds = 1/N_twists
    
    poses = np.zeros([N_poses, 3, 3])
    poses[0, :, :] = base_pose
    
    for i in range(N_twists):
        rdelta_pose = Twist2.expm(mat_segment_twists[:, i] * ds)
        poses[i+1, :, :] = poses[i, :, :] @ rdelta_pose

    return poses