from typing import List

import streamlit as st
from streamlit_vertical_slider import vertical_slider as st_vert_slider
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

from py_attainability.lie import Pose2
from py_attainability.plotters import plot_poses
from py_attainability.rod import calc_poses
from py_attainability import mechanics, actuators

def initialize_st_state():
    # Initialize session state variables
    state_vars = [
        "arm_designs"
    ]

    for state_var_name in state_vars:
        if state_var_name not in st.session_state:
            st.session_state[state_var_name] = None

def arm_designs_list_to_df(arm_designs: List[mechanics.ArmDesign]):
    arm_names = []
    arm_n_actuators = []
    actuator_rhos = []
    for arm_design in arm_designs:
        arm_names.append(arm_design.name)
        arm_n_actuators.append(len(arm_design.actuators))
        actuator_rhos.append([actuator.rho for actuator in arm_design.actuators])

    dict_arm_designs = {
        "Names": arm_names,
        "N actuators": arm_n_actuators,
        "Actuator radii": actuator_rhos
    }

    df_out = pd.DataFrame(dict_arm_designs)
    return df_out

def make_arm_designs():
    arm_width = 0.03
    arm_length = 0.4
    rhos = [-arm_width, arm_width]
    f_forces = [actuators.bellow, actuators.bellow]
    p_maxs = [50, 50]
    muscles = mechanics.Actuator.from_lists(rhos, f_forces, p_maxs)
    arm_design_1 = mechanics.ArmDesign(muscles, l_0=arm_length)

    arm_design_2 = arm_design_1
    for actuator in arm_design_2.actuators:
        actuator.rho *= 2

    four_bellow_radii = [-0.1, -0.03, 0.03, 0.06]
    f_forces = [actuators.bellow for i in range(4)]
    p_maxs = [50 for i in range(4)]
    muscles = mechanics.Actuator.from_lists(four_bellow_radii, f_forces, p_maxs)
    arm_design_3 = mechanics.ArmDesign(muscles, l_0=arm_length)

    st.session_state.arm_designs = [arm_design_1, arm_design_2, arm_design_3]


initialize_st_state()
make_arm_designs()

st.header("Arm simulation")

arm_designs_df = arm_designs_list_to_df(st.session_state.arm_designs)
selection = st.dataframe(arm_designs_df, selection_mode="single-row", on_select="rerun", hide_index=True)
selected_rows = selection["selection"]["rows"]
design_selected = len(selected_rows) != 0

i_selected = None if not design_selected else selected_rows[0]

col_params, col_result = st.columns(2)
with col_params:
    st.subheader("Design")
    arm_length = st.slider("Arm length [cm]", min_value=0.1, max_value=1.0, value=0.5, disabled=design_selected)
    arm_width = st.slider("Arm width [cm]", min_value=0.01, max_value=0.2, value=0.05, disabled=design_selected)

    # Create arm design objects
    if design_selected:
        arm_design = st.session_state.arm_designs[i_selected]
    else:
        rhos = [-arm_width, arm_width]
        f_forces = [actuators.bellow, actuators.bellow]
        p_maxs = [50, 50]
        actuators = [mechanics.Actuator(rho, f_force, p_max) for rho, f_force, p_max in zip(rhos, f_forces, p_maxs)]
        arm_design = mechanics.ArmDesign(actuators, l_0=arm_length)

    st.divider()

    # CONTROL UI
    st.subheader("Control")
    cols = st.columns(len(arm_design.actuators))
    pressures = np.zeros(len(arm_design.actuators))
    for i, col in enumerate(cols):
        with col:
            pressures[i] = st_vert_slider(
                label=f"Pressure {i}", 
                min_value=0,
                max_value=arm_design.actuators[i].p_max,
                default_value=0,
                thumb_color="#ee2b8b",
                slider_color="#ee2b8b"
            )

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

    # SIMULATION PARAMETERS UI
    with st.popover("Simulation parameters"):
        n_segs = st.number_input("Segments [#]", min_value=4, max_value=20, value=5)
    eq_shape = mechanics.solve_equilibrium_shape(n_segs, arm_design, pressures, Q_tip)


# Create body-point-axes
with col_result:
    st.subheader("Result")
    fig = go.Figure()
    mat_poses = calc_poses(arm_design.g_0, eq_shape)
    plot_poses(mat_poses, fig)

    st.plotly_chart(fig)