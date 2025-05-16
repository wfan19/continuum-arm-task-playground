from dataclasses import dataclass, field

import numpy as np

from scipy.optimize import least_squares

from py_attainability.mechanics import ArmDesign, solve_equilibrium_shape
from py_attainability.rod import calc_poses
from py_attainability.lie import Pose2

@dataclass
class Task:
    backbone_shape:     np.array
    tip_load:           np.array = field(default_factory=lambda: np.atleast_2d(np.array([0, 0, 0])).T)

def check_feasibility_naive(N_segments: int, arm_design: ArmDesign, task: Task, p_initial = None, poses_to_count = None):
    """
    Naively check the feasibility of a task by solving the control problem - match the task shape while
    subject to the task load - to the best of the arm's ability.
    """
    if p_initial is None:
        p_initial = np.zeros(len(arm_design.actuators))

    if poses_to_count is None:
        poses_to_count = np.zeros(N_segments)
        poses_to_count[-1] = 1

    # Compute target poses (we only need to do this once)
    target_poses = calc_poses(arm_design.g_0, task.backbone_shape)
    
    # Construct control problem
    def cost_tip_pose_dist(pressures):
        # Compute actuated arm shape and poses
        actuated_twists = solve_equilibrium_shape(N_segments, arm_design, pressures, task.tip_load)
        actuated_poses = calc_poses(arm_design.g_0, actuated_twists)

        # Compute pose difference between actuated tip pose and desired tip pose
        K = np.diag([1, 1, 0.05])
        errors = np.zeros(N_segments)

        for i, (actuated_pose, target_pose) in enumerate(zip(actuated_poses, target_poses)):
            pose_delta = Pose2.inv(actuated_pose) @ target_pose
            v_pose_delta = Pose2.vee(pose_delta)
            errors[i] = v_pose_delta.T @ K @ v_pose_delta

        return np.dot(poses_to_count, errors)
        

    # Solve the control problem
    soln = least_squares(cost_tip_pose_dist, p_initial, method='lm')

    # Parse output and return whether it was solved or not, as well as the optimal control
    attainable = soln.fun < 1e-3
    pressures_optim = soln.x

    return attainable, pressures_optim, soln
