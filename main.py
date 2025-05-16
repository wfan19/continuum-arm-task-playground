from typing import List
import copy
import uuid
from contextlib import contextmanager

import streamlit as st
from streamlit_vertical_slider import vertical_slider as st_vert_slider
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

from py_attainability.lie import Pose2
from py_attainability.plotters import plot_poses
from py_attainability.rod import calc_poses
from py_attainability import actuator_models, mechanics
from py_attainability.utils import st_horizontal

def make_default_arm_designs():
    """
    Placeholder function to create some stored arm-designs for testing purposes
    """
    arm_design_1 = mechanics.ArmDesign.make_default()
    arm_design_1.l_0 = 0.4
    arm_design_1.name="Default arm"

    arm_design_2 = copy.deepcopy(arm_design_1)
    for actuator in arm_design_2.actuators:
        actuator.rho *= 2
    arm_design_2.name="Twice as wide"

    four_bellow_radii = [-0.1, -0.03, 0.03, 0.06]
    models = [actuator_models.Bellow for i in range(4)]
    actuators = mechanics.Actuator.from_lists(four_bellow_radii, models)
    arm_design_3 = mechanics.ArmDesign(actuators, l_0=0.4)

    return [arm_design_1, arm_design_2, arm_design_3]


def initialize_st_state():
    # Initialize session state variables
    state_vars_and_defaults = {
        "arm_designs": make_default_arm_designs(),      # List of arm designs
        "arm_design": mechanics.ArmDesign.make_default(),        # Arm design stored in the interactive-design section
        "dataframe_key": str(uuid.uuid4()),
    }

    for state_var_name, default_val in state_vars_and_defaults.items():
        if state_var_name not in st.session_state:
            st.session_state[state_var_name] = default_val

def arm_designs_list_to_df(arm_designs: List[mechanics.ArmDesign]):
    arm_names = []
    arm_n_actuators = []
    actuator_rhos = []
    arm_lengths = []
    actuator_models = []
    for arm_design in arm_designs:
        arm_names.append(arm_design.name)
        arm_n_actuators.append(len(arm_design.actuators))
        arm_lengths.append(arm_design.l_0)
        actuator_rhos.append([actuator.rho for actuator in arm_design.actuators])
        actuator_models.append([actuator.model.__name__ for actuator in arm_design.actuators])

    dict_arm_designs = {
        "Names": arm_names,
        "Actuators": arm_n_actuators,
        "Length [m]": arm_lengths,
        "Actuator radii [m]": actuator_rhos,
        "Actuator models": actuator_models
    }

    df_out = pd.DataFrame(dict_arm_designs)
    return df_out

def cb_load_selected_arm(selection):
    st.session_state.dataframe_key = str(uuid.uuid4())
    selected_rows = selection["selection"]["rows"]
    design_selected = len(selected_rows) != 0

    if design_selected:
        i_selected = selected_rows[0]
        print(f"Setting the following arm to be the st.sesion_state.arm_design")
        print(st.session_state.arm_designs[i_selected])

        st.session_state.arm_design = st.session_state.arm_designs[i_selected]
    
def cb_delete_selected_arm(selection):
    selected_rows = selection["selection"]["rows"]
    design_selected = len(selected_rows) != 0

    if design_selected:
        i_selected = selected_rows[0]
        st.session_state.arm_designs.pop(i_selected)
    

def cb_on_change_n_actuators():
    n_actuators_new = st.session_state.KEY_N_ACTUATORS
    if n_actuators_new < len(st.session_state.arm_design.actuators):
        st.session_state.arm_design.actuators.pop()
    elif n_actuators_new > len(st.session_state.arm_design.actuators):
        st.session_state.arm_design.actuators.append(mechanics.Actuator(0.0, actuator_models.Bellow))
    else:
        raise IndexError
    
def cb_on_change_actuator_rho(index: int):
    actuator_rho_new = st.session_state[f"RHO_{index}"]
    st.session_state.arm_design.actuators[index].rho = actuator_rho_new

@st.dialog("Save design")
def cb_save_design_dialog():
    name = st.text_input("Design name", value=f"Arm {len(st.session_state.arm_designs)}")
    if st.button("Save", key="KEY_SAVE_ARM_DIALOG"):
        design_to_save = copy.deepcopy(st.session_state.arm_design)
        design_to_save.name = name
        st.session_state.arm_designs.append(design_to_save)
        st.rerun()

def main():
    print("========================= Streamlit refresh =========================")
    # Initialize streamlit state variables
    initialize_st_state()
    st.set_page_config(layout="wide")

    st.header("Arm designer")
    st.text("Select an existing design - or create a new one!")
    arm_designs_df = arm_designs_list_to_df(st.session_state.arm_designs)
    selection = st.dataframe(arm_designs_df, selection_mode="single-row", on_select="rerun", hide_index=True, key=st.session_state.dataframe_key)
    selected_rows = selection["selection"]["rows"]
    design_selected = len(selected_rows) != 0
    i_selected = None if not design_selected else selected_rows[0]
    with st_horizontal():
        st.button("Edit", on_click=cb_load_selected_arm, args=(selection,))
        if design_selected:
            st.button("Delete", on_click=cb_delete_selected_arm, args=(selection,))

    st.subheader("Design")
    st.session_state.arm_design.l_0 = st.slider("Arm length [m]", min_value=0.1, max_value=1.0, value=0.5, disabled=design_selected)

    st.text("Actuators")
    n_actuators = st.number_input(
        "Number of actuators [#]",
        min_value=2,
        max_value=6,
        value=len(st.session_state.arm_design.actuators),
        disabled=design_selected,
        key="KEY_N_ACTUATORS",
        on_change=cb_on_change_n_actuators
    )
    actuators = []
    cols_actuator_design = st.columns(n_actuators, border=True)
    model_classes = actuator_models.ActuatorModel.__subclasses__()
    map_model_name_to_class = {subclass.__name__: subclass for subclass in model_classes}
    for i_col, col in enumerate(cols_actuator_design):
        with col:
            actuator_model_name = st.selectbox("Model", map_model_name_to_class.keys(), key=f"actuator_model_{i_col}", disabled=design_selected)
            model_i = map_model_name_to_class[actuator_model_name]
            default_radius = st.session_state.arm_design.actuators[i_col].rho
            radius_i = st.number_input(
                "Position [m]",
                min_value=-0.5, max_value=0.5, value=default_radius,
                key=f"RHO_{i_col}",
                on_change=cb_on_change_actuator_rho, args=(i_col, ),
                disabled=design_selected
            )

        actuator_i = mechanics.Actuator(radius_i, model_i)
        actuators.append(actuator_i)
    st.session_state.arm_design.actuators = actuators

    with st_horizontal():
        if st.button("Save", disabled=design_selected):
            cb_save_design_dialog()

        with st.popover("Arm design debug"):
            st.write(st.session_state.arm_design)

    # Create arm design objects
    if design_selected:
        arm_design = st.session_state.arm_designs[i_selected]
    else:
        arm_design = st.session_state.arm_design

    st.divider()

    col_params, col_result = st.columns(2)
    with col_params:

        # CONTROL UI
        st.subheader("Control")
        cols = st.columns(len(arm_design.actuators))
        pressures = np.zeros(len(arm_design.actuators))
        for i, col in enumerate(cols):
            with col:
                pressures[i] = st_vert_slider(
                    label=f"Pressure {i}", 
                    min_value=arm_design.actuators[i].model.p_bounds[0],
                    max_value=arm_design.actuators[i].model.p_bounds[1],
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
        st.subheader("Simulated Arm")
        fig = go.Figure()
        mat_poses = calc_poses(arm_design.g_0, eq_shape)
        plot_poses(mat_poses, fig)
        fig.update_layout(margin=dict(t=30, b=0), height=700, xaxis_range=[-0.6, 0.6], yaxis_range=[-1, 0.3])

        st.plotly_chart(fig)
        
if __name__ == "__main__":
    main()