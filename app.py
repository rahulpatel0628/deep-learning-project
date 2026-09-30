import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Make src importable when running: streamlit run app.py
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.inference import load_attention_model, predict_from_vectors

st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #08111f 0%, #101a31 55%, #12152b 100%);
        color: #f5f7fb;
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(40,72,120,.55), rgba(71,45,116,.48));
        border: 1px solid rgba(120,180,255,.20);
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.55rem; }
    .hero p { color: #b7c4d8; margin: .55rem 0 0; font-size: 1.02rem; }
    .metric-card {
        background: rgba(24,35,58,.88);
        border: 1px solid rgba(130,160,210,.16);
        border-radius: 18px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #5dd9ff; }
    .metric-label { color: #aebbd0; font-size: .85rem; }
    .notice {
        background: rgba(255,170,82,.09);
        border: 1px solid rgba(255,170,82,.28);
        border-radius: 12px;
        padding: .9rem 1rem;
        color: #e7edf7;
    }
    .result {
        background: linear-gradient(135deg, rgba(29,71,88,.75), rgba(25,48,80,.75));
        border: 1px solid rgba(80,220,165,.25);
        border-radius: 20px;
        padding: 1.3rem;
        text-align: center;
    }
    .result-emotion { font-size: 2.4rem; font-weight: 850; text-transform: capitalize; }
    .result-confidence { color: #5dd9ff; font-size: 1.25rem; font-weight: 700; }
    section[data-testid="stSidebar"] { background: #0a1222; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Cached resources
# -----------------------------
@st.cache_data
def load_dataset():
    path = PROJECT_ROOT / "data" / "MELD_dataset_with_emotions.csv"
    df = pd.read_csv(path)
    df["predictions"] = df["Multi_Model_Predictions"].apply(json.loads)
    df["video_features"] = df["predictions"].apply(lambda x: x["video"])
    df["audio_features"] = df["predictions"].apply(lambda x: x["audio"])
    df["text_features"] = df["predictions"].apply(lambda x: x["text"])
    return df


@st.cache_resource

def load_model():
    return load_attention_model()


df = load_dataset()
model, class_names, device = load_model()

# -----------------------------
# Helpers
# -----------------------------
def metric_card(value, label):
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def predict_row(row):
    return predict_from_vectors(
        row["text_features"],
        row["audio_features"],
        row["video_features"],
        model=model,
        class_names=class_names,
        device=device,
    )


def attention_dataframe(attention):
    modalities = ["Text", "Audio", "Video"]
    arr = np.asarray(attention, dtype=float)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    return pd.DataFrame(arr, index=modalities, columns=modalities)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🧠 Project Menu")
page = st.sidebar.radio(
    "Navigate",
    ["Prediction", "Model Comparison", "Ablation Study", "Attention Analysis", "About"],
)
st.sidebar.divider()
st.sidebar.caption(f"Device: {device}")
st.sidebar.caption("Model: Attention Fusion")
st.sidebar.caption("Input: 3 × 7-D pre-extracted modality vectors")

# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🧠 Multimodal Emotion Recognition</h1>
      <p>Attention-Based Fusion of Text, Audio and Visual Modality Representations</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Prediction page
# -----------------------------
if page == "Prediction":
    st.subheader("🔮 Emotion Prediction")
    st.markdown(
        '<div class="notice"><b>Demo mode:</b> this version uses the pre-extracted '
        '7-dimensional modality prediction vectors already present in the MELD CSV. '
        'It does not yet convert arbitrary raw audio/video into model vectors.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    search = st.text_input("Search an utterance", placeholder="e.g. happy, why, hello...")
    candidates = df
    if search.strip():
        mask = candidates["Utterance"].astype(str).str.contains(search.strip(), case=False, na=False)
        candidates = candidates[mask]
    if len(candidates) == 0:
        st.warning("No matching utterance found. Try another search term.")
        st.stop()

    options = candidates.index.tolist()
    selected_index = st.selectbox(
        "Choose a dataset sample",
        options,
        format_func=lambda i: f"{i}: {df.loc[i, 'Utterance'][:100]} — actual: {df.loc[i, 'Emotion']}",
    )
    row = df.loc[selected_index]

    st.markdown("### Selected Input")
    st.info(str(row["Utterance"]))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(str(row["Emotion"]).capitalize(), "Actual emotion")
    with c2:
        metric_card("7-D", "Text vector")
    with c3:
        metric_card("7-D", "Audio vector")
    with c4:
        metric_card("7-D", "Video vector")

    if st.button("🔮 Predict Emotion", type="primary", use_container_width=True):
        result = predict_row(row)
        st.session_state["last_result"] = result
        st.session_state["last_index"] = selected_index

    result = st.session_state.get("last_result")
    if result is not None:
        st.divider()
        left, right = st.columns([1, 1.5])
        with left:
            st.markdown(
                f'<div class="result"><div class="result-emotion">{result["emotion"]}</div>'
                f'<div class="result-confidence">Confidence: {result["confidence"] * 100:.2f}%</div></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.caption(f"Model prediction vs actual: {row['Emotion'].capitalize()}")
            if result["emotion"].lower() == str(row["Emotion"]).lower():
                st.success("Correct prediction for this sample.")
            else:
                st.warning("This sample was misclassified by the current model.")

        with right:
            probs = pd.DataFrame(
                {"Emotion": list(result["probabilities"].keys()),
                 "Probability": list(result["probabilities"].values())}
            ).sort_values("Probability", ascending=False)
            st.markdown("#### Emotion probabilities")
            st.bar_chart(probs.set_index("Emotion"), horizontal=True)

# -----------------------------
# Model comparison
# -----------------------------
elif page == "Model Comparison":
    st.subheader("📊 Model Comparison")
    path = PROJECT_ROOT / "outputs" / "model_comparison.csv"
    results = pd.read_csv(path)
    st.dataframe(results, use_container_width=True, hide_index=True)

    metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "Macro-F1"] if c in results.columns]
    if metric_cols:
        st.markdown("### Performance comparison")
        chart_df = results.set_index(results.columns[0])[metric_cols]
        st.bar_chart(chart_df)

    st.markdown("### Key result")
    st.success(
        "The recorded experiment reports Attention Fusion at 36.50% accuracy and "
        "0.2344 Macro-F1, compared with 35.25% accuracy and 0.2147 Macro-F1 for concatenation."
    )

# -----------------------------
# Ablation
# -----------------------------
elif page == "Ablation Study":
    st.subheader("🔬 Ablation Study")
    path = PROJECT_ROOT / "outputs" / "ablation_results.csv"
    results = pd.read_csv(path)
    st.dataframe(results, use_container_width=True, hide_index=True)

    numeric = [c for c in ["Accuracy", "Macro-F1"] if c in results.columns]
    if numeric:
        st.markdown("### Missing-modality performance")
        chart_df = results.set_index(results.columns[0])[numeric]
        st.bar_chart(chart_df)

    st.markdown("### Interpretation")
    st.write(
        "Removing text causes the largest performance drop in the recorded ablation. "
        "Removing video also reduces performance. Removing audio slightly improves the "
        "reported score, suggesting that the current audio representation may add noise "
        "under this experimental configuration."
    )

# -----------------------------
# Attention analysis
# -----------------------------
elif page == "Attention Analysis":
    st.subheader("🧠 Attention Analysis")
    st.write("Select a dataset sample to inspect the 3×3 modality attention matrix.")

    idx = st.selectbox(
        "Sample",
        df.index.tolist(),
        format_func=lambda i: f"{i}: {df.loc[i, 'Utterance'][:90]}",
    )
    result = predict_row(df.loc[idx])
    att_df = attention_dataframe(result["attention"])

    c1, c2 = st.columns([1.05, 1.4])
    with c1:
        st.markdown("#### Attention matrix")
        st.dataframe(att_df.style.format("{:.3f}"), use_container_width=True)
        st.caption("Rows are query modalities; columns are attended-to modalities.")

    with c2:
        st.markdown("#### Normalized attention received")
        received = att_df.mean(axis=0)
        received = received / received.sum()
        received_df = pd.DataFrame({"Modality": received.index, "Attention": received.values})
        st.bar_chart(received_df.set_index("Modality"))
        st.caption("This is a normalized summary of attention received, not a causal feature attribution.")

    st.markdown("#### Prediction")
    st.success(f"Predicted emotion: **{result['emotion'].capitalize()}** — {result['confidence'] * 100:.2f}% confidence")

# -----------------------------
# About
# -----------------------------
elif page == "About":
    st.subheader("ℹ️ About the Project")
    st.markdown(
        """
        ### Attention-Based Multimodal Emotion Recognition

        This application demonstrates the project's trained **Attention Fusion Model**.

        **Current implementation**
        - MELD-based processed CSV
        - 4,000 samples
        - 7 emotion classes
        - Text, Audio and Video modality prediction vectors
        - 7 dimensions per modality
        - 64-dimensional modality projection
        - 4-head Multi-Head Attention
        - MLP classifier for 7 emotions

        **Emotion classes**
        `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

        **Important scope note**

        The current application uses pre-extracted modality-level prediction vectors from
        the CSV. Raw video/audio encoders are not part of this version. A future version can
        add BERT/DistilBERT for text, a stronger audio encoder and a visual encoder such as
        ResNet/ViT, followed by cross-modal attention.
        """
    )

    st.markdown("### Dataset distribution")
    counts = df["Emotion"].value_counts().rename_axis("Emotion").to_frame("Samples")
    st.bar_chart(counts)

    st.markdown("### Project files")
    st.code(
        """app.py
src/model.py
src/preprocessing.py
src/training.py
src/evaluation.py
src/experiments.py
src/inference.py
models/best_attention_model.pth
data/MELD_dataset_with_emotions.csv""",
        language="text",
    )

st.divider()
st.caption("College Project • Attention-Based Multimodal Emotion Recognition")
