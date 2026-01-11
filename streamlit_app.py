import streamlit as st
from pathlib import Path

# 1. Page Configuration
st.set_page_config(
    page_title="Glaucoma Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Glaucoma Detection System")
st.markdown("### 🩺 Machine Learning-based Retinal Fundus Analysis")


st.header("📚 Background: Glaucoma and the Dataset")
st.markdown(
    """
**Quick facts (at a glance):**
- 🧿 **Imaging**: Retinal fundus photographs (optic disc-centered)
- 🧩 **Task**: Binary classification → **Normal** vs **Glaucoma**
- 📐 **Input size**: **512 × 512**
"""
)

with st.expander("🧠 1) Glaucoma Overview", expanded=True):
    st.markdown(
        """
Glaucoma is a chronic eye disease characterized by progressive damage to the optic nerve, which is
essential for transmitting visual information from the eye to the brain. It is one of the leading causes
of irreversible blindness worldwide. The disease often develops slowly and may remain asymptomatic in
its early stages, making early detection particularly challenging.

Clinically, glaucoma leads to structural changes in the optic nerve head, including increased cup-to-disc
ratio, neuroretinal rim thinning, and retinal nerve fiber layer (RNFL) loss. These changes are commonly
visible in retinal fundus images, which makes fundus photography a valuable, non-invasive tool for
glaucoma screening and diagnosis.

Early diagnosis and timely treatment can significantly slow disease progression and preserve vision.
For this reason, automated and computer-aided diagnosis systems based on medical imaging and machine
learning have become increasingly important, especially in large-scale screening programs.
"""
    )

with st.expander("🗂️ 2) Dataset Description", expanded=True):
    st.markdown(
        """
The dataset used in this project consists of retinal fundus images collected for the purpose of glaucoma
classification. Each image is labeled as either:

- ✅ **Healthy (Normal)**
- ⚠️ **Glaucoma**

The images are RGB color fundus photographs centered on the optic disc region, which is the primary
anatomical area affected by glaucoma. Prior to model training, the images are resized to a fixed
resolution of **512 × 512** pixels to ensure compatibility with standard CNN architectures.
"""
    )


with st.expander("🧩 3.1) Dataset Challenges", expanded=False):
    st.markdown(
        """
- **Class imbalance**: Glaucoma cases are often underrepresented compared to healthy samples
- **Subtle visual differences**: Early-stage glaucoma exhibits minimal structural changes
- **Variability in image quality**: Differences in illumination, focus, and acquisition devices

These challenges motivate the use of advanced evaluation metrics such as precision, recall, F1-score,
and AUC-ROC, rather than accuracy alone.
"""
    )

with st.expander("🎯 4) Purpose of the Dataset in This Project", expanded=True):
    st.markdown(
        """
The dataset is used to:

- 🧠 Train and evaluate ML models for automated glaucoma detection
- 🏗️ Compare different model architectures and feature engineering strategies
- 📈 Analyze model performance using clinically relevant metrics

By combining deep learning and classical machine learning methods, this project aims to provide a
comprehensive evaluation of automated glaucoma detection approaches and contribute toward reliable
AI-assisted ophthalmic screening systems.
"""
    )

st.divider()
st.subheader("🖼️ Example Fundus Images")

repo_root = Path(__file__).resolve().parent
img_normal = repo_root / "class0.png"
img_glaucoma = repo_root / "class_1.png"

col1, col2 = st.columns(2, gap="large")
with col1:
    st.image(str(img_normal), caption="✅ Healthy (Normal) example", use_container_width=True)
with col2:
    st.image(str(img_glaucoma), caption="⚠️ Glaucoma example", use_container_width=True)

st.caption("🔬 For research & educational use only — not a medical diagnosis.")
