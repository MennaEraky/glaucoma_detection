import os
from pathlib import Path
import gdown  # lazy import so app can start without it

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


# Default model location (relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _REPO_ROOT / "models" / "LAST_glaucoma_model.keras"

# Google Drive model reference for optional auto-download.
# Can be either a file id ("...") or a full share link ("https://drive.google.com/file/d/.../view?...").
GDRIVE_MODEL_REF = "1OAZgc2VA9DBXALdDItvhVt-JFGeNT5JI"


def _cache_resource_if_available(fn):
    if st is not None:
        return st.cache_resource(fn)
    return fn


@_cache_resource_if_available
def get_model_path() -> str:
    """
    Returns a local filesystem path to the trained model.


    Priority:
    1) GLAUCOMA_MODEL_PATH env var (if set)
    2) ./models/LAST_glaucoma_model.keras (repo-relative)

    If missing, attempts a Google Drive download **only** if `gdown` is installed.
    Otherwise, raises FileNotFoundError with a helpful message.
    """
    env_path = os.getenv("GLAUCOMA_MODEL_PATH")
   

    # Allow overriding the download reference via env var, and support full share links.
    gdrive_ref = os.getenv("GLAUCOMA_MODEL_GDRIVE_URL") or GDRIVE_MODEL_REF
    url = gdrive_ref if "drive.google.com" in gdrive_ref else f"https://drive.google.com/uc?id={gdrive_ref}"
    if st is not None:
        st.info("⬇️ Downloading model from Google Drive...")
    # fuzzy=True lets gdown handle various Drive URL formats, including /file/d/... links.
    gdown.download(url, str(model_path), quiet=False, fuzzy=True)

    if not model_path.exists():
        raise FileNotFoundError(
            "Model download did not produce a file.\n"
            f"- Expected at: {model_path}\n"
            "- Fix: download/copy the model manually, or verify the Google Drive file id."
        )

    return str(model_path)
