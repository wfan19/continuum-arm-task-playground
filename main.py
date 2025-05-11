import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np

from py_attainability.lie import Pose2
from py_attainability.plotters import plot_poses

print("Streamlit reload")
st.header("Arm simulation")

arm_length = st.slider("Arm length [cm]", min_value=10, max_value=100, value=50)
arm_width = st.slider("Arm width [cm]", min_value=1, max_value=20, value=5)
n_nodes = st.number_input("Nodes [#]", min_value=4, max_value=20, value=5)

st.text(f"Arm width: {arm_width}")

# Test plot_poses
n_points = 10
xs = np.linspace(0, 1, n_points)
ys = np.linspace(0, 0, n_points)
thetas = np.linspace(0, np.pi, n_points)

v_poses = np.array([xs, ys, thetas])
mat_poses = np.array([Pose2.hat(v_pose) for v_pose in v_poses.T])

# Create body-point-axes
fig = plot_poses(mat_poses)

st.plotly_chart(fig)