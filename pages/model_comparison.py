import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def app():
    st.title("📊 Model Performance Comparison")

    # =====================================================
    # 1. CNN From Scratch (CNN_plain ONLY)
    # =====================================================
    st.header("🧱 CNN From Scratch")

    cnn_data = {
        "Model": ["CNN_plain"],
        "Threshold": [0.50],
        "Accuracy": [0.8199],
        "Precision": [0.8938],
        "Recall": [0.606],
        "F1": [0.7223],
        "AUC": [0.8672],
    }

    df_cnn = pd.DataFrame(cnn_data)
    st.dataframe(df_cnn, use_container_width=True)

    metric_cnn = st.selectbox(
        "Select metric (CNN_plain)",
        ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    )

    fig0, ax0 = plt.subplots(figsize=(6, 4))
    ax0.bar(df_cnn["Model"], df_cnn[metric_cnn])
    ax0.set_ylabel(metric_cnn)
    ax0.set_title(f"{metric_cnn} – CNN_plain")
    plt.tight_layout()
    st.pyplot(fig0)

    # =====================================================
    # 2. Pretraining Strategy Comparison (Transfer Learning)
    # =====================================================
    st.header("🧠 Pretraining Strategy Comparison")

    pretraining_data = {
        "Model": [
            "DenseNet121_CheXNet",
            "DenseNet121_Fundus",
            "ResNet50_Fundus"
        ],
        "Precision": [0.899038, 0.856846, 0.716904],
        "Recall":    [0.748, 0.826, 0.704],
        "F1_Score":  [0.816594, 0.841141, 0.710394],
        "Accuracy":  [0.870170, 0.879444, 0.778207]
    }

    df_pretrain = pd.DataFrame(pretraining_data)
    st.dataframe(df_pretrain, use_container_width=True)

    metric_pretrain = st.selectbox(
        "Select metric (Pretraining)",
        ["Accuracy", "Precision", "Recall", "F1_Score"]
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(df_pretrain["Model"], df_pretrain[metric_pretrain])
    ax1.set_ylabel(metric_pretrain)
    ax1.set_title(f"{metric_pretrain} – Pretraining Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig1)

    # =====================================================
    # 3. Pretrained CNN Architecture Comparison
    # =====================================================
    st.header("🏗️ Pretrained CNN Architecture Comparison (Threshold = 0.5)")

    architecture_data = {
        "Model": [
            "ResNet50", "DenseNet121", "VGG16",
            "EfficientNetB1", "Xception", "MobileNet"
        ],
        "Threshold": [0.5] * 6,
        "Accuracy":  [0.8717, 0.8717, 0.8601, 0.8570, 0.8447, 0.8192],
        "Precision": [0.8408, 0.9196, 0.8289, 0.8588, 0.8300, 0.7247],
        "Recall":    [0.824,  0.732,  0.804,  0.754,  0.752,  0.858],
        "F1":        [0.8323, 0.8151, 0.8162, 0.8030, 0.7891, 0.7857],
        "AUC":       [0.9327, 0.9326, 0.9290, 0.9258, 0.9181, 0.9122]
    }

    df_arch = pd.DataFrame(architecture_data)
    st.dataframe(df_arch, use_container_width=True)

    metric_arch = st.selectbox(
        "Select metric (Pretrained CNNs)",
        ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(df_arch["Model"], df_arch[metric_arch])
    ax2.set_ylabel(metric_arch)
    ax2.set_title(f"{metric_arch} – Pretrained CNN Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig2)

    # =====================================================
    # 4. Classical ML Model Comparison
    # =====================================================
    st.header("💻 Classical Machine Learning Models")

    classical_data = {
        "Rank": [1, 2, 3, 4, 5],
        "Approach": [
            "simple_features", "advanced_features", "engineered_features",
            "simple_features", "simple_features"
        ],
        "Model": [
            "XGBoost (GPU)", "SVC (GPU)", "SVC (GPU)", "SVC (GPU)", "Random Forest (GPU)"
        ],
        "Precision": [0.825890, 0.813100, 0.812573, 0.808324, 0.822647],
        "Recall":    [0.825986, 0.814385, 0.813805, 0.809745, 0.808585],
        "F1_Score":  [0.825937, 0.811669, 0.810962, 0.808624, 0.797811],
        "Accuracy":  [0.825986, 0.814385, 0.813805, 0.809745, 0.808585]
    }

    df_classical = pd.DataFrame(classical_data)
    st.dataframe(df_classical, use_container_width=True)

    metric_classical = st.selectbox(
        "Select metric (Classical ML)",
        ["Accuracy", "Precision", "Recall", "F1_Score"]
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.bar(df_classical["Model"], df_classical[metric_classical])
    ax3.set_ylabel(metric_classical)
    ax3.set_title(f"{metric_classical} – Classical ML Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig3)

    # =====================================================
    # Notes
    # =====================================================
    st.markdown("""
    **Notes:**
    - CNN_plain represents training from scratch
    - All pretrained CNN results use ImageNet or medical-domain pretraining
    - Threshold is fixed at **0.5** for fair comparison
    - AUC is emphasized due to class imbalance in glaucoma datasets
    """)


# Render page
app()
