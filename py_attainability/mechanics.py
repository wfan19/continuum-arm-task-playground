from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, least_squares

from py_attainability.rod import calc_poses
from py_attainability.lie import Pose2

@dataclass
class Actuator:
    rho:        float
    f_force:    callable
    p_max:      float

    @staticmethod
    def from_lists(rhos, fs, p_maxs):
        """
        Constructor for creating a list of actuator objects from individual lists of parameters
        """
        actuators_out = []
        for rho, f_force, p_max in zip(rhos, fs, p_maxs):
            new_actuator = Actuator(rho, f_force, p_max)
            actuators_out.append(new_actuator)
        return actuators_out

@dataclass
class ArmDesign:
    actuators:  List[Actuator]
    l_0:        float       = 0.5
    g_0:        np.array    = field(default_factory=lambda: Pose2.hat(np.array([0, 0, -np.pi/2])))
    name:       str     = "Arm"

def calc_reaction_wrench(mat_segment_twists: np.array, pressures: np.array, arm_design: ArmDesign) -> np.array:
    """
    Compute the combined reaction wrench exerted upon the arm center by actuators within an arm design,
    held at a given shape and actuated to given pressure
    """
    N_actuators = len(arm_design.actuators)
    N_twists = mat_segment_twists.shape[1]

    # Initialize a bunch of zero matrices
    lengths, strains, forces, curvatures, moments = [np.zeros([N_actuators, N_twists]) for i in range(5)]
    
    lengths_centerline = mat_segment_twists[0, :]
    
    for i_actuator, actuator in enumerate(arm_design.actuators):
        # Use the in-coordinate version of the adjoint action to compute the lengths and curvatures
        # of each actuator based on the centerline lengths and curvature, and the relative positioning
        # TODO - it would be nicer to formulate this using the actual adjoint matrix
        curvatures_centerline = mat_segment_twists[2, :]
        lengths[i_actuator, :] = lengths_centerline - actuator.rho*curvatures_centerline
        strains[i_actuator, :] = (lengths[i_actuator, :] - arm_design.l_0) / arm_design.l_0
        forces[i_actuator, :] = arm_design.actuators[i_actuator].f_force(strains[i_actuator, :], pressures[i_actuator])
        
        curvatures[i_actuator, :] = curvatures_centerline / lengths[i_actuator, :]
        k_moment = (-1/3.5) * (pressures[i_actuator] / arm_design.actuators[i_actuator].p_max)
        moments[i_actuator, :] = k_moment * curvatures[i_actuator, :]

    mat_A = np.zeros([3, N_actuators])
    mat_A[0, :] = np.ones(N_actuators)
    mat_A[2, :] = np.array([actuator.rho for actuator in arm_design.actuators])
    rxn_force = mat_A @ forces

    mat_C = np.zeros([3, N_actuators])
    mat_C[2, :] = np.ones(N_actuators)
    rxn_moment = mat_C @ moments

    return rxn_force + rxn_moment

def calc_external_wrench(mat_segment_twists: np.array, tip_wrench: np.array, g_0: np.array) -> np.array:
    """
    Compute the (material-centric) wrench experienced along an arm at the centerline induced by a tip-wrench
    """
    N_twists = mat_segment_twists.shape[1]
    external_wrenches = np.zeros([3, N_twists])

    poses = calc_poses(g_0, mat_segment_twists)
    g_tip = poses[-1, :, :]

    for i_twist, v_twist in enumerate(mat_segment_twists.T):
        g_i = poses[i_twist, :, :]
        g_i_tip = np.linalg.inv(g_i) @ g_tip    # TODO - implement analytic SE2 inverse?

        # Convert the world frame tip wrench to be in the rod tip's frame
        g_ucirc_right_tip = Pose2.left_lifted_action(g_tip).T @ tip_wrench
        
        # Since the wrench is now in the rod's material-centric frame, we can
        # use the adjoint to find its material-centric contribution on an earlier
        # part of the rod
        g_ucirc_right_i = np.linalg.inv(Pose2.adjoint(g_i_tip)).T @ g_ucirc_right_tip
        external_wrenches[:, i_twist]  = np.squeeze(g_ucirc_right_i)

    return external_wrenches

def equilibrium_residual(mat_segment_twists, arm_design, pressures, tip_wrench):
    reaction_wrenches = calc_reaction_wrench(mat_segment_twists, pressures, arm_design)
    external_wrenches = calc_external_wrench(mat_segment_twists, tip_wrench, arm_design.g_0)

    residual = external_wrenches + reaction_wrenches
    residual[1, :] += mat_segment_twists[1, :] * 1e3
    return residual

def solve_equilibrium_shape(N_segments, arm_design, pressures, tip_wrench):
    default_segment_twists = np.zeros([3, N_segments])
    default_segment_twists[0, :] = arm_design.l_0
     
    # TODO - make it so we either don't need to bias, or can bias in either direction?
    default_segment_twists[2, :] = 0

    def f_residual_vectorized(v_twists):
        mat_segment_twists = np.reshape(v_twists, [3, N_segments])
        residual = equilibrium_residual(mat_segment_twists, arm_design, pressures, tip_wrench)
        return residual.ravel()

    soln = least_squares(f_residual_vectorized, default_segment_twists.ravel(), method="lm", ftol=1e-8, xtol=1e-8)
    print(soln)
    return np.reshape(soln.x, [3, N_segments])