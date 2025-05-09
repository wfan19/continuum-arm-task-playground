import numpy as np
import Omega2, Rot2, Pose2

def hat(v_twist2):
    """
    Lift the vector form of a twist into full matrix form
    """
    vx, vy, omega = v_twist2
    mat_skew_sym = Omega2.hat(omega)
    mat_twist2 = np.zeros(3)
    mat_twist2[0:2, 0:2] = mat_skew_sym
    mat_twist2[0:2, 2] = np.array([vx, vy])

def vee(mat_twist2):
    """
    Convert a 2D twist matrix into its 1D vector component
    by extracting relevant information
    """
    v_twist2 = np.zeros([3, 1])

    v_twist2[0:2] = mat_twist2[0:2, 2]
    v_twist2[2] = mat_twist2[1, 0]

def expm(v_twist2):
    """
    Analytic exponential map in SE2
    For reference, see https://arxiv.org/pdf/1812.01537
    """
    vx, vy, omega = v_twist2
    
    skew_sym_1 = Omega2.hat(1)
    if np.abs(omega) > 1e-5:
        V = (1/omega) * (np.sin(omega) * np.eye(2) + ((1-np.cos(omega)) * skew_sym_1))
    else:
        V = np.eye(2)
    
    R = Rot2.hat(omega)
    Pose2_out = np.eye(3)
    Pose2_out[0:2, 0:2] = R 
    Pose2_out[0:2, 2] = V @ np.array([[vx], [vy]])