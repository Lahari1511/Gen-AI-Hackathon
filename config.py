"""Central settings. Values come from the .env file (never hard-code keys)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _secret(name: str, default: str = "") -> str:
    """Read a setting from .env / environment, or from Streamlit Cloud secrets when deployed."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default

# --- API keys ---
GEMINI_API_KEY = _secret("GEMINI_API_KEY", "")
HF_TOKEN = _secret("HF_TOKEN", "")

# --- Models ---
GEMINI_MODEL = _secret("GEMINI_MODEL", "gemini-2.5-flash")
QWEN_MODEL = _secret("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# "hf" = Hugging Face Inference API, "ollama" = local Ollama, "label" = no VLM (use the given label)
VLM_BACKEND = _secret("VLM_BACKEND", "hf")

# --- Dataset ---
HF_DATASET = _secret("HF_DATASET", "kannanwisen/Indian-Traffic-Sign-Classification")

# --- Files ---
REFERENCE_FILE = ROOT / "data" / "reference_signs.json"
RESULTS_DIR = ROOT / "results"

# --- Image quality thresholds (tuned for small sign images) ---
MIN_SIDE_PX = 32          # smaller than this = too small to read
BLUR_THRESHOLD = 60.0     # Laplacian variance below this = blurry
DARK_THRESHOLD = 40       # mean brightness below this = too dark
BRIGHT_THRESHOLD = 225    # mean brightness above this = washed out

DISCLAIMER = (
    "Educational use only. This tool does not give live driving instructions. "
    "Sign meanings and rules can vary by region; always follow official road signs, "
    "traffic police and the law."
)
