import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def app():
    st.title("📊 Model Performance Comparison")


    # =====================================================
    # 3. Classical ML Model Comparison
    # =====================================================
    st.header("💻 Classical ML Models")
    classical_data = {
        "Rank": [1, 2, 3, 4, 5],
        "Approach": [
            "simple_features", "advanced_features", "engineered_features",
            "simple_features", "simple_features"
        ],
        "Model": [
            "XGBoost (GPU)", "SVC (GPU)", "SVC (GPU)", "SVC (GPU)", "Random Forest (GPU)"
        ],
        "Pair": [
            "simple_features|XGBoost (GPU)",
            "advanced_features|SVC (GPU)",
            "engineered_features|SVC (GPU)",
            "simple_features|SVC (GPU)",
            "simple_features|Random Forest (GPU)"
        ],
        "Precision": [0.825890, 0.813100, 0.812573, 0.808324, 0.822647],
        "Recall":    [0.825986, 0.814385, 0.813805, 0.809745, 0.808585],
        "F1_Score":  [0.825937, 0.811669, 0.810962, 0.808624, 0.797811],
        "Support":   [1724, 1724, 1724, 1724, 1724],
        "Accuracy":  [0.825986, 0.814385, 0.813805, 0.809745, 0.808585]
    }
    df_classical = pd.DataFrame(classical_data)
    st.dataframe(df_classical, use_container_width=True)

    metric_classical = st.selectbox(
        "Select metric (Classical ML)",
        ["Accuracy", "Precision", "Recall", "F1_Score"]
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.bar(df_classical["Model"], df_classical[metric_classical], color="green")
    ax3.set_ylabel(metric_classical)
    ax3.set_title(f"{metric_classical} – Classical ML Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig3)

    # =====================================================
    # 2. CNN Architecture Comparison
    # =====================================================
    st.header("🏗️ CNN Architecture Comparison (Threshold = 0.5)")
    architecture_data = {
        "Model": [
            "ResNet50", "DenseNet121", "VGG16",
            "EfficientNetB1", "Xception", "MobileNet"
        ],
        "Threshold": [0.5]*6,
        "Accuracy":  [0.8717, 0.8717, 0.8601, 0.8570, 0.8447, 0.8192],
        "Precision": [0.8408, 0.9196, 0.8289, 0.8588, 0.8300, 0.7247],
        "Recall":    [0.824,  0.732,  0.804,  0.754,  0.752,  0.858],
        "F1":        [0.8323, 0.8151, 0.8162, 0.8030, 0.7891, 0.7857],
        "AUC":       [0.9327, 0.9326, 0.9290, 0.9258, 0.9181, 0.9122]
    }
    df_arch = pd.DataFrame(architecture_data)
    st.dataframe(df_arch, use_container_width=True)

    metric_arch = st.selectbox(
        "Select metric (Architecture)",
        ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(df_arch["Model"], df_arch[metric_arch], color="orange")
    ax2.set_ylabel(metric_arch)
    ax2.set_title(f"{metric_arch} – CNN Architecture Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig2)

    # =====================================================
    # 1. Pretraining Strategy Comparison (Transfer Learning)
    # =====================================================
    st.header("🧠 Transfer Learning Models Comparison")
    pretraining_data = {
        "Model": [
            "DenseNet121_CheXNet",
            "DenseNet121_Fundus",
            "ResNet50_Fundus"
        ],
        "Precision": [0.8398, 0.8361, 0.8389],
        "Recall":    [0.818, 0.796, 0.802],
        "F1_Score":  [0.8288, 0.8156, 0.8200],
        "Accuracy":  [0.8694, 0.8609, 0.8640]
    }
    df_pretrain = pd.DataFrame(pretraining_data)
    st.dataframe(df_pretrain, use_container_width=True)

    metric_pretrain = st.selectbox(
        "Select metric (Pretraining)",
        ["Accuracy", "Precision", "Recall", "F1_Score"]
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(df_pretrain["Model"], df_pretrain[metric_pretrain], color="skyblue")
    ax1.set_ylabel(metric_pretrain)
    ax1.set_title(f"{metric_pretrain} – Pretraining Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig1)

    # =====================================================
    # Notes
    # =====================================================
    st.markdown("""
    **Notes:**
    - All CNN results are evaluated at a fixed threshold of **0.5**
    - DenseNet121 shows the most stable AUC across experiments
    - Fundus fine-tuning improves recall and F1-score
    - Classical ML models were trained on engineered/simple/advanced features
    - XGBoost and SVC provide the highest precision among classical models
    """)

    # =====================================================
    # 4. Overall Comparison (All Models)
    # =====================================================
    st.header("🏁 Overall Comparison (All Models)")
    st.caption("Unifies metrics across all sections so you can compare everything side-by-side.")

    df_pretrain_all = df_pretrain.rename(columns={"F1_Score": "F1"}).copy()
    df_pretrain_all["Category"] = "Pretraining"

    df_arch_all = df_arch.copy()
    df_arch_all["Category"] = "CNN Architecture"

    df_classical_all = df_classical.rename(columns={"F1_Score": "F1"}).copy()
    df_classical_all["Category"] = "Classical ML"

    # Normalize to a common schema
    common_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        for c in common_cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[["Category", "Model"] + common_cols]

    import numpy as np

    df_all = pd.concat(
        [
            normalize(df_pretrain_all),
            normalize(df_arch_all),
            normalize(df_classical_all),
        ],
        ignore_index=True,
    )

    st.dataframe(df_all, use_container_width=True)

    metric_all = st.selectbox(
        "Select metric (Overall)",
        [c for c in common_cols if df_all[c].notna().any()],
        index=0,
    )

    df_plot = df_all[df_all[metric_all].notna()].sort_values(metric_all, ascending=False)

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    colors = df_plot["Category"].map(
        {
            "Pretraining": "skyblue",
            "CNN Architecture": "orange",
            "Classical ML": "green",
        }
    ).fillna("gray")
    ax4.bar(df_plot["Model"], df_plot[metric_all], color=colors)
    ax4.set_ylabel(metric_all)
    ax4.set_title(f"{metric_all} – Overall Comparison")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig4)

    st.subheader("Top performers (per metric)")
    top_rows = []
    for m in [c for c in common_cols if df_all[c].notna().any()]:
        best = df_all[df_all[m].notna()].sort_values(m, ascending=False).head(1)
        if not best.empty:
            top_rows.append(
                {
                    "Metric": m,
                    "Best Model": best.iloc[0]["Model"],
                    "Category": best.iloc[0]["Category"],
                    "Score": float(best.iloc[0][m]),
                }
            )
    st.dataframe(pd.DataFrame(top_rows), use_container_width=True)


# Streamlit runs page files directly; render the page automatically.
app()
