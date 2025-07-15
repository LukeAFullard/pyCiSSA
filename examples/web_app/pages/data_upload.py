import streamlit as st

# Load CSS
with open("../assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Data Upload")

st.write("This is where you will upload your data.")
