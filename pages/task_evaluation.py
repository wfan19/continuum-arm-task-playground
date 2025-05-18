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

example_task_no_load = Task(np.diag([0.5367, 0, 1.5541]) @ np.ones([3, N_segments]))

backbone_2N_down = np.array([
    [0.6422529108881496, 0.6417306408605336, 0.6400395036608001, 0.6368277602225578, 0.6314734715285549],
    [4.4337e-19, 2.6011e-4, 5.3124e-4, 8.3713e-4, 1.1915e-3],
    [0.7868293722316826, 0.8260458470399996, 0.9460740261968872, 1.1546573549522114, 1.4653836829094455]
])
example_task_loaded = Task(backbone_2N_down, np.atleast_2d(np.array([0, -2, 0])).T)

arm_design_1 = mechanics.ArmDesign.make_default()
arm_design_1.l_0 = 0.4
arm_design_1.name="Default arm"

# st.text(check_feasibility_naive(N_segments, arm_design_1, example_task_no_load))
st.text(check_feasibility_naive(N_segments, arm_design_1, example_task_loaded, p_initial=np.array([0, 0])))
