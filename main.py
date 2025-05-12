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
col_params, col_result = st.columns(2)

with col_params:
    st.subheader("Design")
    arm_length = st.slider("Arm length [cm]", min_value=0.1, max_value=1.0, value=0.5)
    arm_width = st.slider("Arm width [cm]", min_value=0.01, max_value=0.2, value=0.05)

    rhos = [-arm_width, arm_width]
    f_forces = [actuators.bellow, actuators.bellow]
    p_maxs = [50, 50]
    actuators = [mechanics.Actuator(rho, f_force, p_max) for rho, f_force, p_max in zip(rhos, f_forces, p_maxs)]
    arm_design = mechanics.ArmDesign(actuators, l_0=arm_length)

    st.divider()

    st.subheader("Control")
    cols = st.columns(len(arm_design.actuators))
    pressures = np.zeros(len(arm_design.actuators))
    for i, col in enumerate(cols):
        with col:
            pressures[i] = st.slider(f"Pressure {i}", min_value=0, max_value=arm_design.actuators[i].p_max, value=0)

    st.divider()

    st.subheader("External load")
    col_load_x, col_load_y, col_load_theta = st.columns(3)
    with col_load_x:
        tip_load_x = st.slider("Tip load X [N]", min_value=-10, max_value=10, value=0)
    with col_load_y:
        tip_load_y = st.slider("Tip load Y [N]", min_value=-10, max_value=10, value=0)
    with col_load_theta:
        tip_load_torque = st.slider("Tip load Torque [Nm]", min_value=-5, max_value=5, value=0)

    Q_tip = np.atleast_2d(np.array([tip_load_x, tip_load_y, tip_load_torque])).T

    st.divider()

    with st.expander("Simulation parameters"):
        n_nodes = st.number_input("Nodes [#]", min_value=4, max_value=20, value=5)
    eq_shape = mechanics.solve_equilibrium_shape(n_nodes, arm_design, pressures, Q_tip)

    # Test plot_poses
    mat_poses = calc_poses(arm_design.g_0, eq_shape)

# Create body-point-axes
with col_result:
    st.subheader("Result")
    fig = go.Figure()
    plot_poses(mat_poses, fig)

    st.plotly_chart(fig)