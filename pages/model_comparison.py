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
        "Accuracy": [
            0.8259860788863109,
            0.814385150812065,
            0.8138051044083526,
            0.8097447795823666,
            0.808584686774942,
            0.7749419953596288,
            0.771461716937355,
            0.7470997679814385,
            0.7465197215777262,
            0.7465197215777262,
            0.7389791183294664,
            0.7146171693735499,
            0.7047563805104409,
            0.6995359628770301,
            0.6885150812064965,
            0.6653132250580046,
            0.6409512761020881,
            0.6299303944315545,
            0.6281902552204176,
            0.6252900232018561,
        ],
        "Approach": [
            "simple_features",
            "advanced_features",
            "engineered_features",
            "simple_features",
            "simple_features",
            "simple_features",
            "simple_features",
            "advanced_features",
            "advanced_features",
            "engineered_features",
            "engineered_features",
            "engineered_features_roi",
            "advanced_features",
            "engineered_features",
            "engineered_features_roi",
            "engineered_features_roi",
            "engineered_features_roi",
            "engineered_features",
            "advanced_features",
            "engineered_features_roi",
        ],
        "F1_Score": [
            0.8259371263454202,
            0.81166880822529,
            0.8109622646706018,
            0.8086235548849365,
            0.7978108896310822,
            0.7707653752893594,
            0.7603440362857895,
            0.7466608721118986,
            0.7403095493163357,
            0.7401826371260304,
            0.7386037038123678,
            0.717201026305534,
            0.7086631616342969,
            0.7034871766299337,
            0.6799301718718453,
            0.6614361450451277,
            0.6440207784808474,
            0.5128407187271072,
            0.506627280699901,
            0.4987081322536019,
        ],
        "Model": [
            "XGBoost (GPU)",
            "SVC (GPU)",
            "SVC (GPU)",
            "SVC (GPU)",
            "Random Forest (GPU)",
            "Logistic Regression (GPU)",
            "KNN (GPU)",
            "Logistic Regression (GPU)",
            "KNN (GPU)",
            "KNN (GPU)",
            "Logistic Regression (GPU)",
            "SVC (GPU)",
            "XGBoost (GPU)",
            "XGBoost (GPU)",
            "KNN (GPU)",
            "Logistic Regression (GPU)",
            "XGBoost (GPU)",
            "Random Forest (GPU)",
            "Random Forest (GPU)",
            "Random Forest (GPU)",
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
            "engineered_features_roi|Random Forest (GPU)",
        ],
        "Precision": [
            0.8258904669595404,
            0.8131002919913572,
            0.8125731927790986,
            0.8083243781128965,
            0.8226468952162858,
            0.7723995998539444,
            0.7758206617613244,
            0.7462957880998855,
            0.7428121128330161,
            0.7428459194570634,
            0.7382788659354076,
            0.723157587474,
            0.7304987968158899,
            0.7281137628819493,
            0.6807596147964916,
            0.6597169016560023,
            0.6902231376632906,
            0.6793937235312146,
            0.6865359573593474,
            0.6860564923182493,
        ],
        "Rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "Recall": [
            0.8259860788863109,
            0.814385150812065,
            0.8138051044083526,
            0.8097447795823666,
            0.808584686774942,
            0.7749419953596288,
            0.771461716937355,
            0.7470997679814385,
            0.7465197215777262,
            0.7465197215777262,
            0.7389791183294664,
            0.7146171693735499,
            0.7047563805104409,
            0.6995359628770301,
            0.6885150812064965,
            0.6653132250580046,
            0.6409512761020881,
            0.6299303944315545,
            0.6281902552204176,
            0.6252900232018561,
        ],
        "Support": [
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
            1724.0,
        ],
    }
    df_classical = pd.DataFrame(classical_data)
    st.dataframe(df_classical, use_container_width=True)

    metric_classical = st.selectbox(
        "Select metric (Classical ML)",
        ["Accuracy", "Precision", "Recall", "F1_Score"]
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    # Use `Pair` instead of `Model` here because multiple rows share the same model name (e.g., SVC)
    # and the real comparison is between (features | model) pairs.
    ax3.bar(df_classical["Pair"], df_classical[metric_classical], color="green")
    ax3.set_ylabel(metric_classical)
    ax3.set_title(f"{metric_classical} – Classical ML Comparison")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)

    # =====================================================
    # 2. PTransfer Learning Models Comparison (ImageNet)
    # =====================================================
    st.header("🏗️ Transfer Learning Models Comparison (ImageNet) ")
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
        "Select metric (Transfer Learning Models (ImageNet))",
        ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(df_arch["Model"], df_arch[metric_arch], color="orange")
    ax2.set_ylabel(metric_arch)
    ax2.set_title(f"{metric_arch} – Transfer Learning  (ImageNet)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig2)

    # =====================================================
    # 1.Transfer Learning Models Comparison (ImageNet)
    # =====================================================
    st.header("🧠 Transfer Learning Models Comparison (Medical-weights)")
    pretraining_data = {
        "Model": [
            "DenseNet121_CheXNet",
            "DenseNet121_Fundus",
            "ResNet50_Fundus"
        ],
        "Precision": [0.8398, 0.8361, 0.8389],
        "Recall":    [0.818, 0.796, 0.802],
        "F1_Score":  [0.8288, 0.8156, 0.8200],
        "Accuracy":  [0.8694, 0.8609, 0.8640],
        "AUC":       [0.932, 0.929, 0.935]
    }
    df_pretrain = pd.DataFrame(pretraining_data)
    st.dataframe(df_pretrain, use_container_width=True)

    metric_pretrain = st.selectbox(
        "Select metric (Transfer Learning )",
        ["Accuracy", "Precision", "Recall", "F1_Score", "AUC"]
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(df_pretrain["Model"], df_pretrain[metric_pretrain], color="skyblue")
    ax1.set_ylabel(metric_pretrain)
    ax1.set_title(f"{metric_pretrain} – Transfer Learning  Comparison")
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
    df_pretrain_all["Category"] = "Transfer Learning Models Medical Weights "

    df_arch_all = df_arch.copy()
    df_arch_all["Category"] = "Transfer Learning Models  (ImageNet)"

    df_classical_all = df_classical.rename(columns={"F1_Score": "F1"}).copy()
    df_classical_all["Category"] = "Classical ML"
    # Compare classical methods by `Pair` (features|model), not by model name alone.
    df_classical_all["Model"] = df_classical_all["Pair"]

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

    # Filters + unified table
    category_filter = st.multiselect(
        "Filter categories",
        options=sorted(df_all["Category"].unique().tolist()),
        default=sorted(df_all["Category"].unique().tolist()),
    )
    df_all_filtered = df_all[df_all["Category"].isin(category_filter)].copy()

    st.dataframe(df_all_filtered, use_container_width=True)

    metric_all = st.selectbox(
        "Select metric (Overall)",
        [c for c in common_cols if df_all_filtered[c].notna().any()],
        index=0,
    )

    df_plot = df_all_filtered[df_all_filtered[metric_all].notna()].sort_values(metric_all, ascending=False)

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    colors = df_plot["Category"].map(
        {
            "Transfer Learning Models Medical Weights ": "skyblue",
            "Transfer Learning Models  (ImageNet)": "orange",
            "Classical ML": "green",
        }
    ).fillna("gray")
    ax4.bar(df_plot["Model"], df_plot[metric_all], color=colors)
    ax4.set_ylabel(metric_all)
    ax4.set_title(f"{metric_all} – Overall Comparison")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig4)

    # Quick "best" cards
    st.subheader("🏅 Best models (quick view)")
    cols = st.columns(5)
    for i, m in enumerate([c for c in common_cols if df_all_filtered[c].notna().any()]):
        best = df_all_filtered[df_all_filtered[m].notna()].sort_values(m, ascending=False).head(1)
        if best.empty:
            continue
        cols[i].metric(
            label=m,
            value=f"{best.iloc[0][m]:.4f}",
            delta=f"{best.iloc[0]['Model']} • {best.iloc[0]['Category']}",
        )

    st.subheader("Top performers (per metric)")
    top_rows = []
    for m in [c for c in common_cols if df_all_filtered[c].notna().any()]:
        best = df_all_filtered[df_all_filtered[m].notna()].sort_values(m, ascending=False).head(1)
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
