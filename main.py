import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np

from py_attainability.lie import Pose2
from py_attainability.plotters import plot_poses
from py_attainability.rod import calc_poses
from py_attainability import mechanics, actuators

print("Streamlit reload")
st.header("Arm simulation")

arm_length = st.slider("Arm length [cm]", min_value=0.1, max_value=1.0, value=0.5)
arm_width = st.slider("Arm width [cm]", min_value=0.01, max_value=0.2, value=0.05)
n_nodes = st.number_input("Nodes [#]", min_value=4, max_value=20, value=5)

Q_tip = np.atleast_2d(np.array([-1, 0, 0])).T
pressure = np.array([50, 0])

rhos = [-arm_width, arm_width]
f_forces = [actuators.bellow, actuators.bellow]
p_maxs = [50, 50]
actuators = [mechanics.Actuator(rho, f_force, p_max) for rho, f_force, p_max in zip(rhos, f_forces, p_maxs)]
arm_design = mechanics.ArmDesign(actuators, l_0=arm_length)

eq_shape = mechanics.solve_equilibrium_shape(n_nodes, arm_design, pressure, Q_tip)
print("Equilibrium shape")
print(eq_shape)


# Test plot_poses
mat_poses = calc_poses(arm_design.g_0, eq_shape)

# Create body-point-axes
fig = plot_poses(mat_poses)

st.plotly_chart(fig)