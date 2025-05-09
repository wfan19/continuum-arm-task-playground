import numpy as np
from scipy.linalg import logm

def hat(theta):
    """
    Create a 2D rotation matrix based on the scalar rotation angle
    For all derivations, see https://arxiv.org/pdf/1812.01537
    """
    mat_SO2_out = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return mat_SO2_out

def vee(mat_in):
    """
    Convert a 2D rotation matrix into its scalar rotation angle
    """
    return np.atan2(mat_in[1, 0], mat_in[0, 0])

def logm(mat_in):
    """
    Compute the so2 element that exponentiates into this rotation matrix
    """
    return logm(mat_in)

def adjoint(mat_in):
    """
    Compute the adjoint-action matrix of SO2, which is itself.
    """
    return mat_in