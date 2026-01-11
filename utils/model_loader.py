import os
import zipfile
from pathlib import Path

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


def _is_valid_keras_file(path: Path) -> bool:
    """
    A `.keras` file is a zip archive. We also sanity-check size to avoid treating
    small HTML error pages as valid downloads.
    """
    try:
        return (
            path.exists()
            and path.is_file()
            and path.stat().st_size > 50_000  # ~50KB: avoids tiny HTML/redirect pages
            and zipfile.is_zipfile(path)
        )
    except OSError:
        return False


@_cache_resource_if_available
def get_model_path() -> str:
    """
    Returns a local filesystem path to the trained model.

    Priority:
    1) GLAUCOMA_MODEL_PATH env var (if set)
    2) ./models/LAST_glaucoma_model.keras (repo-relative)

    If missing/invalid, attempts a Google Drive download if `gdown` is installed.
    Otherwise, raises FileNotFoundError with a helpful message.
    """
    env_path = os.getenv("GLAUCOMA_MODEL_PATH")
    model_path = Path(env_path).expanduser().resolve() if env_path else _DEFAULT_MODEL_PATH

    if _is_valid_keras_file(model_path):
        return str(model_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Try optional download path (kept non-fatal if gdown is missing)
    try:
        import gdown  # lazy import so app can start without it
    except ModuleNotFoundError as e:
        raise FileNotFoundError(
            "Model file not found.\n"
            f"- Looked for: {model_path}\n"
            "- Fix: either place the `.keras` file there, or set env var `GLAUCOMA_MODEL_PATH`.\n"
            "- Optional: install `gdown` to enable auto-download."
        ) from e

    # If a bad/partial file exists, remove it so we can retry cleanly.
    if model_path.exists() and not _is_valid_keras_file(model_path):
        try:
            model_path.unlink()
        except OSError:
            pass

    # Allow overriding the download reference via env var, and support full share links.
    gdrive_ref = os.getenv("GLAUCOMA_MODEL_GDRIVE_URL") or GDRIVE_MODEL_REF
    url = gdrive_ref if "drive.google.com" in gdrive_ref else f"https://drive.google.com/uc?id={gdrive_ref}"

    if st is not None:
        st.info("⬇️ Downloading model from Google Drive...")

    # fuzzy=True lets gdown handle various Drive URL formats, including /file/d/... links.
    try:
        out = gdown.download(url, str(model_path), quiet=False, fuzzy=True)
        st.info("Model Downloaded Succesfuly ✅.")

    except Exception as e:
        raise RuntimeError(
            "Failed to download the model from Google Drive.\n"
            f"- URL/ID: {gdrive_ref}\n"
            f"- Target: {model_path}\n"
            "Fix: ensure the Drive file is shared as 'Anyone with the link', and try again."
        ) from e

    if not _is_valid_keras_file(model_path):
        raise FileNotFoundError(
            "Model download did not produce a valid `.keras` file.\n"
            f"- URL/ID: {gdrive_ref}\n"
            f"- Download result: {out}\n"
            f"- File: {model_path}\n"
            "- Fix: verify the link is public and points to the actual `.keras` file (not a folder)."
        )

    return str(model_path)

