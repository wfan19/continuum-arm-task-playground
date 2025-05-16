import streamlit as st
import numpy as np

from py_attainability.lie import Pose2
from py_attainability.plotters import plot_poses
from py_attainability.rod import calc_poses
from py_attainability import mechanics
from py_attainability.task_feasibility import Task, check_feasibility_naive

st.title("Task attainability")

# with st.popover("Simulation parameters"):
#     N_segments = st.number_input("Segment count", min=4, max=10, default_value=5)
N_segments = 5

example_task_no_load = Task(np.diag([0.588, 0, 1.7345]) @ np.ones([3, N_segments]))

backbone_2N_down = np.array([
    [0.59331, 0.59281, 0.59117, 0.58803, 0.58274],
    [4.4337e-19, 2.6011e-4, 5.3124e-4, 8.3713e-4, 1.1915e-3],
    [6.5213e-1, 6.9209e-1, 8.1509e-1, 1.0316, 1.3606]
])
example_task_loaded = Task(backbone_2N_down, np.atleast_2d(np.array([0, 2, 0])).T)

arm_design_1 = mechanics.ArmDesign.make_default()
arm_design_1.l_0 = 0.4
arm_design_1.name="Default arm"

print(check_feasibility_naive(N_segments, arm_design_1, example_task_no_load))
