import streamlit as st
import plotly

st.header("Arm simulation")

arm_length = st.slider("Arm length [cm]", min_value=10, max_value=100, value=50)
arm_width = st.slider("Arm width [cm]", min_value=1, max_value=20, value=5)

st.text(f"Arm width: {arm_width}")