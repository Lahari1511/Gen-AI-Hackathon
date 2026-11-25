"""Streamlit user interface - Traffic Sign Educational Explanation Tool (PS 113).

Run with:  streamlit run app.py
"""
import streamlit as st

from src import config, pipeline, preprocess

st.set_page_config(page_title="Traffic Sign Explainer", page_icon="🚦", layout="wide")

BADGE = {"high": ("🟢", "High"), "medium": ("🟠", "Medium"), "low": ("🔴", "Low")}
SECTION_TITLES = {
    "meaning": "📘 Meaning",
    "purpose": "🎯 Purpose",
    "importance": "⭐ Importance",
    "uncertainty": "❓ Uncertainty",
}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    backend_label = st.radio(
        "Vision model (Qwen2.5-VL)",
        ["Hugging Face API", "Ollama (local)", "Label only (no vision)"],
        index={"hf": 0, "ollama": 1, "label": 2}.get(config.VLM_BACKEND, 0),
        help="Hugging Face API needs HF_TOKEN. Ollama needs the model pulled locally.",
    )
    backend = {"Hugging Face API": "hf", "Ollama (local)": "ollama", "Label only (no vision)": "label"}[backend_label]

    st.subheader("Status")
    st.write(("✅" if config.GEMINI_API_KEY else "⚠️") + " Gemini API key")
    st.write(("✅" if config.HF_TOKEN else "⚠️") + " Hugging Face token")
    st.caption(f"LLM: {config.GEMINI_MODEL}  \nVLM: {config.QWEN_MODEL if backend == 'hf' else config.OLLAMA_MODEL}")

    st.subheader("🎲 Dataset sample")
    st.caption(config.HF_DATASET)
    if st.button("Load random dataset image"):
        try:
            from src import dataset
            with st.spinner("Loading dataset from Hugging Face (first time can take a minute)..."):
                img, label = dataset.sample(1)[0]
            st.session_state["sample"] = (img, label)
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
    if "sample" in st.session_state and st.button("Clear sample"):
        del st.session_state["sample"]

# ---------------- Header ----------------
st.title("🚦 Traffic Sign Educational Explanation Tool")
st.caption("Upload a traffic sign photo to learn what it means, why it is used and why it matters.")
st.warning("⚠️ " + config.DISCLAIMER)

left, right = st.columns([1, 1.4], gap="large")

with left:
    sample = st.session_state.get("sample")
    upload = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    image_input, true_label = None, ""
    if upload is not None:
        image_input = upload.getvalue()
        st.image(image_input, caption="Uploaded image", width="stretch")
    elif sample is not None:
        image_input, true_label = sample
        st.image(image_input, caption=f"Dataset sample — true label: {true_label}", width="stretch")

    user_label = st.text_input("Class label (optional)", value=true_label,
                               help="If you know the sign name, enter it. The app checks it against the image.")
    question = st.text_input("Your question (optional)", placeholder="e.g. What does this sign mean?")
    go = st.button("🔍 Explain this sign", type="primary", width="stretch",
                   disabled=image_input is None)

with right:
    if go and image_input is not None:
        with st.spinner("Analysing the sign and writing the explanation..."):
            try:
                res = pipeline.run(image_input, user_label=user_label.strip(), question=question.strip(), backend=backend)
            except Exception as e:  # last-resort guard so the app never crashes
                res = pipeline.Result(status="error", message=f"Unexpected error: {e}")
        st.session_state["result"] = res

    res = st.session_state.get("result")
    if res is None:
        st.info("👈 Upload an image (or load a dataset sample) and click **Explain this sign**.")
    elif res.status == "refused":
        st.error("🛑 " + res.message)
    elif res.status == "error":
        st.error("❌ " + res.message)
        st.caption("Tip: check your internet connection and API keys in the .env file, then try again.")
    elif res.status == "no_sign":
        st.info("🔎 " + res.message)
        if res.vision and res.vision.description:
            st.caption("What the model saw: " + res.vision.description)
    else:
        icon, level = BADGE[res.confidence]
        st.subheader(f"Recognised sign: {res.sign_name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{icon} {level}")
        c2.metric("Category", res.category or "—")
        c3.metric("Label check", res.label_match)

        if res.status == "uncertain":
            st.warning("⚠️ " + res.message)
            if res.alternatives:
                st.markdown("**Possible signs (for learning only):**")
                for a in res.alternatives:
                    st.markdown(f"- **{a['name']}** — {a['meaning']}")
        else:
            for key, title in SECTION_TITLES.items():
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.write(res.sections.get(key, ""))
            src = "Gemini (grounded on verified reference)" if res.explanation_source == "gemini" \
                else "Verified reference text (Gemini not used)"
            st.caption(f"Explanation source: {src}")
            if res.message:
                st.caption("ℹ️ " + res.message)
            if res.alternatives:
                st.caption("Other possibilities: " + ", ".join(a["name"] for a in res.alternatives))

        if res.notes:
            with st.expander("Why this confidence level?"):
                for n in res.notes:
                    st.write("• " + n)

    if res is not None and res.image_check is not None:
        with st.expander("🔧 Technical details"):
            st.json({
                "status": res.status,
                "image_quality": res.image_check.quality if res.image_check.ok else "invalid",
                "image_metrics": res.image_check.metrics,
                "vision_backend": res.vision.backend if res.vision else None,
                "vision_confidence": res.vision.confidence if res.vision else None,
                "vision_description": res.vision.description if res.vision else None,
                "total_seconds": round(res.seconds, 2),
            })

st.divider()
st.caption("PS 113 · Qwen2.5-VL (vision) + Gemini (explanation) · Dataset: "
           f"{config.HF_DATASET} (Hugging Face) · Reference meanings based on IRC:67-2022 · Educational prototype.")
