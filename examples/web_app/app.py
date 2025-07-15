import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="pycissa",
    page_icon="assets/pycissa.jpg",
    layout="wide",
)

# Load CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App header
st.image("assets/pycissa.jpg", width=150)
st.title("pycissa: Time Series Analysis with CISSA")

# Main content
st.write("Welcome to the pycissa web app! This is the main page.")
