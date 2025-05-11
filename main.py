import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np

from py_attainability.lie import Pose2
from py_attainability.plotters import plot_poses
from py_attainability.rod import calc_poses

print("Streamlit reload")
st.header("Arm simulation")

arm_length = st.slider("Arm length [cm]", min_value=10, max_value=100, value=50)
arm_width = st.slider("Arm width [cm]", min_value=1, max_value=20, value=5)
n_nodes = st.number_input("Nodes [#]", min_value=4, max_value=20, value=5)

st.text(f"Arm width: {arm_width}")

# Test plot_poses
n_points = 10
vxs = np.linspace(0.1, 1, n_points)
vys = np.zeros(n_points)
omegas = np.pi/6 * np.ones(n_points)

v_twists = np.array([vxs, vys, omegas])
base_pose = Pose2.hat([0, 0, -np.pi/2])
mat_poses = calc_poses(base_pose, v_twists)

# Create body-point-axes
fig = plot_poses(mat_poses)

st.plotly_chart(fig)