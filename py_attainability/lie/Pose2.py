import numpy as np
import Rot2, Omega2
from scipy.linalg import logm

def rotation(mat_pose2):
    """
    Extract the rotation matrix component of a pose2 matrix
    """
    return mat_pose2[0:2, 0:2]

def translation(mat_pose2):
    """
    Extract the translation vector component of a pose2 matrix
    """
    return mat_pose2[0:2, 2]

def hat(v_pose2):
    """
    Convert vector representation of a pose2, aka [x, y, theta], into
    a pose matrix, aka a 2D homogenous transform matrix
    """
    x, y, theta = v_pose2
    
    Pose2_out = np.eye(3)
    Pose2_out[0:2, 0:2] = Rot2.hat(theta)
    Pose2_out[0:2, 2] = np.array([x, y])
    return Pose2_out

def vee(mat_pose2):
    """
    Convert a 2D homogenous transform matrix into its standard vector representation
    by extracting the relevant components
    """
    v_Pose2_out = np.zeros([3, 1])
    v_Pose2_out[0:2] = mat_pose2[0:2, 2]
    v_Pose2_out[2] = Rot2.vee(mat_pose2[0:2, 0:2])

def logm(mat_pose2):
    raise NotImplementedError

def adjoint(mat_pose2):
    """
    Create the adjoint action matrix for this transformation
    For derivation, see https://arxiv.org/pdf/1812.01537
    """
    R = rotation(mat_pose2)
    t = translation(mat_pose2)
    
    adj_out = np.eye(3)
    adj_out[0:2, 0:2] = R
    adj_out[0:2, 2] = t

def left_lifted_action(mat_pose2):
    """
    Left lifted action on SE2
    Aka the jacobian of a left-action delta_g*g evaluated at g
    Formula derived by Ross L Hatton (GM ver 2022/12/4 pg 151)
    """
    TeLg = np.eye(3)
    TeLg[0:2, 0:2] = rotation(mat_pose2)
    return TeLg

def right_lifted_action(mat_pose2):
    """
    Right lifted action on SE2
    Aka the jacobian of a right-action g*delta_g evaluated at g
    Formula derived by Ross L Hatton (GM ver 2022/12/4 pg 167)
    """
    TeRg = np.eye(3)
    TeRg[0, 2] = -mat_pose2[1, 2]
    TeRg[1, 2] = mat_pose2[0, 2]
    return TeRg

