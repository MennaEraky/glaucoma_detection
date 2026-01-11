import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px  # For advanced visualizations

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("📊 Model Performance Comparison")

# =====================================================
# SECTION 1: Pretraining Strategy Comparison
# =====================================================
st.header("🧠 Pretraining Strategy Comparison")

pretraining_data = {
    "Model": ["DenseNet121_CheXNet", "DenseNet121_Fundus", "ResNet50_Fundus"],
    "Precision": [0.899038, 0.856846, 0.716904],
    "Recall":    [0.748,    0.826,    0.704],
    "F1_Score":  [0.816594, 0.841141, 0.710394],
    "Accuracy":  [0.870170, 0.879444, 0.778207]
}
df_pretrain = pd.DataFrame(pretraining_data)

# Display Table
st.dataframe(df_pretrain, use_container_width=True)

# Visualizing All Metrics Side-by-Side
st.subheader("Multi-Metric Comparison")
df_pretrain_melted = df_pretrain.melt(id_vars="Model", var_name="Metric", value_name="Score")

fig_pretrain = px.bar(
    df_pretrain_melted, 
    x="Model", 
    y="Score", 
    color="Metric", 
    barmode="group",
    height=400,
    title="Side-by-Side Comparison of Pretraining Metrics",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_pretrain, use_container_width=True)

# =====================================================
# SECTION 2: CNN Architecture Comparison
# =====================================================
st.header("🏗️ CNN Architecture Comparison (Threshold = 0.5)")

architecture_data = {
    "Model": ["ResNet50", "DenseNet121", "VGG16", "EfficientNetB1", "Xception", "MobileNet"],
    "Accuracy":  [0.8717, 0.8717, 0.8601, 0.8570, 0.8447, 0.8192],
    "Precision": [0.8408, 0.9196, 0.8289, 0.8588, 0.8300, 0.7247],
    "Recall":    [0.824,  0.732,  0.804,  0.754,  0.752,  0.858],
    "F1":        [0.8323, 0.8151, 0.8162, 0.8030, 0.7891, 0.7857],
    "AUC":       [0.9327, 0.9326, 0.9290, 0.9258, 0.9181, 0.9122]
}
df_arch = pd.DataFrame(architecture_data)

st.dataframe(df_arch, use_container_width=True)

# Heatmap Visualization
st.subheader("Performance Intensity Heatmap")
fig_heat = px.imshow(
    df_arch.set_index("Model"),
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu_r', # Red to Blue scale
    title="Metric Strengths per Architecture"
)
st.plotly_chart(fig_heat, use_container_width=True)

# =====================================================
# NOTES
# =====================================================
st.markdown(
    """
    **Key Takeaways:**
    - **DenseNet121** provides the best balance of Precision and AUC.
    - **Fundus fine-tuning** significantly boosts Recall compared to general pretraining.
    - **MobileNet** shows high recall but struggles with precision, making it sensitive but less specific.
    """
)
