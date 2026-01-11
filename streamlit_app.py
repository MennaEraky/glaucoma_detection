import streamlit as st

st.set_page_config(
    page_title="Glaucoma Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# You don't need to manually list pages here. 
# Streamlit will automatically put 1_Test_Model and 2_Model_Comparison 
# in the sidebar.

st.title("🧠 Glaucoma Detection System")

st.markdown(
    """
    ### Deep Learning-based Retinal Fundus Analysis
    Welcome to the Glaucoma Detection System. This tool uses advanced CNN architectures 
    to assist in the identification of glaucomatous features in retinal images.
    
    **Available Modules:**
    1. **Test Model:** Upload a fundus image for real-time classification.
    2. **Model Comparison:** View performance metrics (Accuracy, F1-Score) across different architectures.
    """
)

st.info("👈 Select a module from the sidebar to begin.")
