import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def app():
    st.title("📊 Model Performance Comparison")

    # =====================================================
    # 1. Classical ML Models (FULL RESULTS)
    # =====================================================
    st.header("💻 Classical ML Models (All Experiments)")

    classical_data = {
        "Rank": list(range(1, 21)),
        "Approach": [
            "simple_features","advanced_features","engineered_features","simple_features","simple_features",
            "simple_features","simple_features","advanced_features","advanced_features","engineered_features",
            "engineered_features","engineered_features_roi","advanced_features","engineered_features",
            "engineered_features_roi","engineered_features_roi","engineered_features_roi",
            "engineered_features","advanced_features","engineered_features_roi"
        ],
        "Model": [
            "XGBoost (GPU)","SVC (GPU)","SVC (GPU)","SVC (GPU)","Random Forest (GPU)",
            "Logistic Regression (GPU)","KNN (GPU)","Logistic Regression (GPU)","KNN (GPU)","KNN (GPU)",
            "Logistic Regression (GPU)","SVC (GPU)","XGBoost (GPU)","XGBoost (GPU)","KNN (GPU)",
            "Logistic Regression (GPU)","XGBoost (GPU)","Random Forest (GPU)",
            "Random Forest (GPU)","Random Forest (GPU)"
        ],
        "Pair": [
            "simple_features|XGBoost (GPU)",
            "advanced_features|SVC (GPU)",
            "engineered_features|SVC (GPU)",
            "simple_features|SVC (GPU)",
            "simple_features|Random Forest (GPU)",
            "simple_features|Logistic Regression (GPU)",
            "simple_features|KNN (GPU)",
            "advanced_features|Logistic Regression (GPU)",
            "advanced_features|KNN (GPU)",
            "engineered_features|KNN (GPU)",
            "engineered_features|Logistic Regression (GPU)",
            "engineered_features_roi|SVC (GPU)",
            "advanced_features|XGBoost (GPU)",
            "engineered_features|XGBoost (GPU)",
            "engineered_features_roi|KNN (GPU)",
            "engineered_features_roi|Logistic Regression (GPU)",
            "engineered_features_roi|XGBoost (GPU)",
            "engineered_features|Random Forest (GPU)",
            "advanced_features|Random Forest (GPU)",
            "engineered_features_roi|Random Forest (GPU)"
        ],
        "Accuracy": [
            0.825986,0.814385,0.813805,0.809745,0.808585,
            0.774942,0.771462,0.747100,0.746520,0.746520,
            0.738979,0.714617,0.704756,0.699536,0.688515,
            0.665313,0.640951,0.629930,0.628190,0.625290
        ],
        "Precision": [
            0.825890,0.813100,0.812573,0.808324,0.822647,
            0.772400,0.775821,0.746296,0.742812,0.742846,
            0.738279,0.723158,0.730499,0.728114,0.680760,
            0.659717,0.690223,0.679394,0.686536,0.686056
        ],
        "Recall": [
            0.825986,0.814385,0.813805,0.809745,0.808585,
            0.774942,0.771462,0.747100,0.746520,0.746520,
            0.738979,0.714617,0.704756,0.699536,0.688515,
            0.665313,0.640951,0.629930,0.628190,0.625290
        ],
        "F1": [
            0.825937,0.811669,0.810962,0.808624,0.797811,
            0.770765,0.760344,0.746661,0.740310,0.740183,
            0.738604,0.717201,0.708663,0.703487,0.679930,
            0.661436,0.644021,0.512841,0.506627,0.498708
        ],
        "Support": [1724.0]*20
    }

    df_classical = pd.DataFrame(classical_data)
    st.dataframe(df_classical, use_container_width=True)

    metric_classical = st.selectbox(
        "Select metric (Classical ML)",
        ["Accuracy", "Precision", "Recall", "F1"]
    )

    fig_c, ax_c = plt.subplots(figsize=(10, 5))
    ax_c.bar(df_classical["Pair"], df_classical[metric_classical], color="green")
    ax_c.set_ylabel(metric_classical)
    ax_c.set_title(f"{metric_classical} – Classical ML Comparison")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    st.pyplot(fig_c)

    # =====================================================
    # 2. Transfer Learning (ImageNet)
    # =====================================================
    st.header("🏗️ Transfer Learning Models (ImageNet)")

    df_arch = pd.DataFrame({
        "Model": ["ResNet50","DenseNet121","VGG16","EfficientNetB1","Xception","MobileNet"],
        "Accuracy":[0.8717,0.8717,0.8601,0.8570,0.8447,0.8192],
        "Precision":[0.8408,0.9196,0.8289,0.8588,0.8300,0.7247],
        "Recall":[0.824,0.732,0.804,0.754,0.752,0.858],
        "F1":[0.8323,0.8151,0.8162,0.8030,0.7891,0.7857],
        "AUC":[0.9327,0.9326,0.9290,0.9258,0.9181,0.9122]
    })

    st.dataframe(df_arch, use_container_width=True)

    # =====================================================
    # 3. Transfer Learning (Medical Weights)
    # =====================================================
    st.header("🧠 Transfer Learning Models (Medical Weights)")

    df_pretrain = pd.DataFrame({
        "Model":["DenseNet121_CheXNet","DenseNet121_Fundus","ResNet50_Fundus"],
        "Accuracy":[0.8694,0.8609,0.8640],
        "Precision":[0.8398,0.8361,0.8389],
        "Recall":[0.818,0.796,0.802],
        "F1":[0.8288,0.8156,0.8200],
        "AUC":[0.932,0.929,0.935]
    })

    st.dataframe(df_pretrain, use_container_width=True)

    # =====================================================
    # Notes
    # =====================================================
    st.markdown("""
    **Notes**
    - Classical ML results cover **20 feature–model combinations**
    - Pair = *feature set | classifier*
    - Transfer learning models outperform classical ML in AUC
    - Medical pretraining improves robustness and stability
    """)

# Run app
app()
