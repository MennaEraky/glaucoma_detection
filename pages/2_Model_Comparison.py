import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Config (Must be first)
st.set_page_config(
    page_title="Glaucoma Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Working Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Test Model", "Model Comparison"]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":
    st.title("🧠 Glaucoma Detection System")
    st.markdown("""
    ### Deep Learning-based Retinal Fundus Analysis
    Welcome to the Glaucoma Detection System. 
    
    **Available Modules:**
    1. **Test Model:** Upload a fundus image for real-time classification.
    2. **Model Comparison:** View performance metrics across different architectures.
    """)
    st.info("👈 Use the sidebar menu to select a module.")

# =====================================================
# TEST MODEL PAGE
# =====================================================
elif page == "Test Model":
    st.title("🖼️ Test the Trained Model")
    st.write("Upload a retinal image to detect signs of Glaucoma.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        st.success("Image received. Model analysis would run here.")

# =====================================================
# MODEL COMPARISON PAGE (Your Data & Visuals)
# =====================================================
elif page == "Model Comparison":
    st.title("📊 Model Performance Comparison")

    # --- Section 1: Pretraining Strategy ---
    st.header("🧠 Pretraining Strategy Comparison")
    pretraining_data = {
        "Model": ["DenseNet121_CheXNet", "DenseNet121_Fundus", "ResNet50_Fundus"],
        "Precision": [0.899038, 0.856846, 0.716904],
        "Recall": [0.748, 0.826, 0.704],
        "F1_Score": [0.816594, 0.841141, 0.710394],
        "Accuracy": [0.870170, 0.879444, 0.778207]
    }
    df_pretrain = pd.DataFrame(pretraining_data)
    
    # Plotly Grouped Bar Chart for Pretraining
    df_melt_pre = df_pretrain.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_pre = px.bar(df_melt_pre, x="Model", y="Score", color="Metric", barmode="group",
                     title="Pretraining Strategy Comparison")
    st.plotly_chart(fig_pre, use_container_width=True)

    # --- Section 2: CNN Architecture ---
    st.header("🏗️ CNN Architecture Comparison")
    architecture_data = {
        "Model": ["ResNet50", "DenseNet121", "VGG16", "EfficientNetB1", "Xception", "MobileNet"],
        "Accuracy": [0.8717, 0.8717, 0.8601, 0.8570, 0.8447, 0.8192],
        "Precision": [0.8408, 0.9196, 0.8289, 0.8588, 0.8300, 0.7247],
        "Recall": [0.824, 0.732, 0.804, 0.754, 0.752, 0.858],
        "F1": [0.8323, 0.8151, 0.8162, 0.8030, 0.7891, 0.7857],
        "AUC": [0.9327, 0.9326, 0.9290, 0.9258, 0.9181, 0.9122]
    }
    df_arch = pd.DataFrame(architecture_data)
    
    # Heatmap Visualization for Architectures
    fig_heat = px.imshow(
        df_arch.set_index("Model"),
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Viridis',
        title="Metric Heatmap: Architectures vs Performance"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Notes:\n- All results evaluated at a threshold of **0.5**.")
