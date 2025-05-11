import numpy as np
from py_attainability.lie import Rot2

def hat(theta):
    """
    Return the tangent-space representation of a 1D rotation, an element of so2
    The exponential map of this matrix results in its corresponding rotation matrix
    See https://arxiv.org/pdf/1812.01537
    """
    return np.array([
        [0, -theta],
        [theta, 0]
    ])

def vee(mat_so2):
    """
    Return the scalar rotational-velocity value associated with the skew-symmetrix so2 matrix
    """
    return mat_so2[1, 0]

def expm(mat_so2):
    """
    Take the analytic exponential map of so2, which is equivalent to constructing a rotation matrix
    """
    theta = vee(mat_so2)
    return Rot2.hat(theta)