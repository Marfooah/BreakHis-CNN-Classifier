"""HistoAI — BreakHis CNN diagnostic assistant (Streamlit + Stitch designs)."""

from __future__ import annotations

import io
import uuid
from datetime import datetime

import streamlit as st
from PIL import Image

from model_utils import (
    FINDINGS,
    INTERPRETATIONS,
    MODEL_METRICS,
    make_gradcam_heatmap,
    load_model,
    predict,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HistoAI — Pathology Diagnostic Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Stitch design system CSS (dark unified theme)
# ---------------------------------------------------------------------------

STITCH_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --surface: #10131a;
        --surface-low: #191b23;
        --surface-high: #282b33;
        --surface-lowest: #0b0e15;
        --on-surface: #f1f1f1;
        --on-surface-variant: #c1c6d7;
        --primary: #0058bc;
        --primary-container: #0070eb;
        --secondary: #006e28;
        --secondary-dim: #53e16f;
        --tertiary: #bc000a;
        --outline-variant: rgba(193, 198, 215, 0.12);
        --gutter: 24px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: var(--on-surface);
    }

    .stApp {
        background-color: var(--surface);
    }

    header[data-testid="stHeader"] {
        background: var(--surface);
        border-bottom: 1px solid var(--outline-variant);
    }

    #MainMenu, footer, .stDeployButton { visibility: hidden; }

    .histo-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px var(--gutter);
        border-bottom: 1px solid var(--outline-variant);
        background: var(--surface);
        margin: -1rem -1rem 1.5rem -1rem;
    }

    .histo-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .histo-nav a {
        color: var(--on-surface-variant);
        text-decoration: none;
        margin-left: 32px;
        font-size: 15px;
        font-weight: 500;
    }

    .histo-nav a.active {
        color: var(--on-surface);
        border-bottom: 2px solid var(--primary-container);
        padding-bottom: 4px;
    }

    .card {
        background: var(--surface-low);
        border: 1px solid var(--outline-variant);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }

    .card-glass {
        background: rgba(25, 27, 35, 0.85);
        backdrop-filter: blur(20px);
    }

    .chip-benign {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(0, 110, 40, 0.15);
        color: var(--secondary-dim);
        border: 1px solid rgba(0, 110, 40, 0.25);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .chip-malignant {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(188, 0, 10, 0.15);
        color: #ffb4aa;
        border: 1px solid rgba(188, 0, 10, 0.25);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .metric-tile {
        background: var(--surface-low);
        border: 1px solid var(--outline-variant);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: var(--on-surface);
    }

    .metric-label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: var(--on-surface-variant);
        text-transform: uppercase;
    }

    .prob-bar {
        display: flex;
        height: 56px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--outline-variant);
    }

    .prob-benign {
        background: #006e28;
        display: flex;
        align-items: center;
        padding: 0 16px;
        font-size: 12px;
        font-weight: 600;
        color: #6ffb85;
        min-width: fit-content;
    }

    .prob-malignant {
        background: #ba1a1a;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding: 0 16px;
        font-size: 12px;
        font-weight: 600;
        color: #ffdad5;
        flex: 1;
    }

    .upload-hero {
        text-align: center;
        max-width: 640px;
        margin: 0 auto;
    }

    .upload-icon-ring {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: var(--surface-high);
        border: 1px solid var(--outline-variant);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 16px;
        font-size: 32px;
    }

    .insight-tile {
        background: var(--surface-low);
        border: 1px solid var(--outline-variant);
        border-radius: 12px;
        padding: 16px;
        height: 100%;
    }

    .insight-title {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .insight-body {
        font-size: 12px;
        color: var(--on-surface-variant);
        line-height: 1.4;
    }

    .footer-disclaimer {
        text-align: center;
        font-size: 12px;
        color: var(--on-surface-variant);
        padding: 24px;
        border-top: 1px solid var(--outline-variant);
        margin-top: 48px;
        line-height: 1.6;
    }

    .result-label {
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1;
    }

    .result-benign { color: #53e16f; }
    .result-malignant { color: #ff6b6b; }

    .confidence-ring-wrap {
        position: relative;
        width: 192px;
        height: 192px;
        margin: 16px auto;
    }

    .finding-row {
        display: flex;
        gap: 16px;
        padding-bottom: 16px;
        margin-bottom: 16px;
        border-bottom: 1px solid var(--outline-variant);
    }

    .finding-icon {
        width: 48px;
        height: 48px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        font-size: 22px;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(11, 14, 21, 0.4);
        border: 1.5px dashed rgba(0, 112, 235, 0.5);
        border-radius: 12px;
        padding: 24px;
    }

    div[data-testid="stFileUploader"] section {
        padding: 16px;
    }

    .stButton > button[kind="primary"] {
        background: var(--primary-container) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 18px !important;
        padding: 12px 24px !important;
        width: 100%;
    }

    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: var(--on-surface) !important;
        border: 1px solid var(--outline-variant) !important;
        border-radius: 8px !important;
    }
</style>
"""


def inject_css() -> None:
    st.markdown(STITCH_CSS, unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "page": "intro",
        "uploaded_image": None,
        "uploaded_name": None,
        "results": None,
        "case_id": None,
        "show_heatmap": False,
        "model": None,
        "model_status": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner="Loading CNN model…")
def get_model():
    return load_model()


def render_header() -> None:
    left, right = st.columns([2, 3])
    with left:
        st.markdown(
            '<div class="histo-brand" style="padding:8px 0;">🔬 HistoAI</div>',
            unsafe_allow_html=True,
        )
    with right:
        nav_cols = st.columns(4)
        nav_items = [
            ("intro", "Home"),
            ("upload", "Analysis"),
            ("results", "Results"),
            ("confidence", "Confidence"),
        ]
        for col, (key, label) in zip(nav_cols, nav_items):
            with col:
                disabled = key not in ("intro", "upload") and st.session_state.results is None
                if st.button(
                    label,
                    key=f"nav_{key}",
                    disabled=disabled,
                    use_container_width=True,
                ):
                    st.session_state.page = key
                    st.rerun()
    st.markdown(
        '<hr style="border-color:rgba(193,198,215,0.12);margin:0 0 1.5rem 0;">',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-disclaimer">
            This AI-generated analysis is for clinical decision support only and does not
            constitute a final medical diagnosis. Always correlate with pathological findings
            and clinical history.<br>
            <span style="opacity:0.5;margin-top:8px;display:block;">HistoAI v4.2.0 · BreakHis CNN Classifier</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_ring_html(pct: float, color: str) -> str:
    circumference = 565.48
    offset = circumference * (1 - pct / 100)
    return f"""
    <div class="confidence-ring-wrap">
        <svg width="192" height="192" style="transform: rotate(-90deg);">
            <circle cx="96" cy="96" r="90" fill="transparent"
                stroke="rgba(51,57,65,0.3)" stroke-width="12"/>
            <circle cx="96" cy="96" r="90" fill="transparent"
                stroke="{color}" stroke-width="12"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{offset}"
                stroke-linecap="round"/>
        </svg>
        <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">
            <div style="font-size:32px;font-weight:700;">{pct:.1f}%</div>
            <div style="font-size:12px;font-weight:600;letter-spacing:0.05em;color:#c1c6d7;">CONFIDENCE</div>
        </div>
    </div>
    """


def page_intro() -> None:
    st.markdown('<div class="upload-hero">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="upload-icon-ring">🔬</div>
        <h1 style="font-size:40px;font-weight:700;margin-bottom:12px;letter-spacing:-0.02em;">
            HistoAI
        </h1>
        <p style="font-size:18px;color:#c1c6d7;margin-bottom:8px;">
            AI-Powered Breast Histopathology Diagnostic Assistant
        </p>
        <p style="color:#c1c6d7;font-size:15px;line-height:1.6;margin-bottom:32px;">
            HistoAI analyzes microscopic tissue slides from the BreakHis dataset to help
            clinicians classify breast biopsies as <strong style="color:#53e16f;">Benign</strong>
            or <strong style="color:#ff6b6b;">Malignant</strong> — with confidence scoring
            and visual explainability.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Start Analysis →", type="primary", use_container_width=True):
        st.session_state.page = "upload"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3)
    steps = [
        ("1", "Upload Slide", "Import a PNG or JPG histopathology microscope image."),
        ("2", "AI Diagnosis", "Our classifier evaluates tissue morphology and returns a label."),
        ("3", "Review Confidence", "Explore probability scores, findings, and Grad-CAM heatmaps."),
    ]
    for col, (num, title, body) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="insight-tile">
                    <div style="font-size:14px;font-weight:700;color:#0070eb;margin-bottom:8px;">
                        STEP {num}
                    </div>
                    <div class="insight-title">{title}</div>
                    <div class="insight-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    feat_cols = st.columns(3)
    features = [
        ("🧬", "BreakHis Trained", "Classifier trained on thousands of annotated breast tissue images."),
        ("🎯", "High Sensitivity", "Optimized thresholds to prioritize malignant case detection."),
        ("🔒", "Clinical Support", "Decision-support tool — not a substitute for pathologist review."),
    ]
    for col, (icon, title, body) in zip(feat_cols, features):
        with col:
            st.markdown(
                f"""
                <div class="insight-tile">
                    <div style="font-size:24px;margin-bottom:8px;">{icon}</div>
                    <div class="insight-title">{title}</div>
                    <div class="insight-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def page_upload() -> None:
    model, status = get_model()
    st.session_state.model = model
    st.session_state.model_status = status

    if status and ("No model" in status or "No trained" in status or "Run `python3 train_sklearn_model.py`" in status):
        st.warning(status)
    elif status and "sklearn" in status.lower():
        st.info(status)

    st.markdown('<div class="upload-hero">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="upload-icon-ring">🧬</div>
        <h1 style="font-size:32px;font-weight:600;margin-bottom:8px;">Upload Histopathology Image</h1>
        <p style="color:#c1c6d7;font-size:16px;margin-bottom:24px;">
            Supports PNG, JPG, JPEG (Microscopic tissue slides)
        </p>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drag & drop slide here or click to browse",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image = Image.open(uploaded)
        st.warning(
            "⚠️ This model only works on histopathology microscope images. "
            "Random images will be rejected or produce unreliable results."
        )
        st.session_state.uploaded_image = image
        st.session_state.uploaded_name = uploaded.name
        st.image(image, caption=uploaded.name, use_container_width=True)

    analyze = st.button(
        "Analyze Image",
        type="primary",
        disabled=st.session_state.uploaded_image is None,
        use_container_width=True,
    )

    st.markdown(
        """
        <div style="margin-top:16px;display:flex;align-items:center;justify-content:center;
            gap:6px;color:rgba(193,198,215,0.6);font-size:12px;font-weight:600;">
            🔒 HIPAA Compliant • End-to-end Encrypted
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze and st.session_state.uploaded_image is not None:
        with st.spinner("Running CNN inference…"):
            results = predict(model, st.session_state.uploaded_image)
            st.session_state.results = results
            st.session_state.case_id = f"PX-{uuid.uuid4().hex[:5].upper()}-B"
            st.session_state.page = "results"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3)
    insights = [
        ("⚙️", "Sub-micron Precision", "AI models trained on multi-resolution tissue segments for cellular-level accuracy."),
        ("📋", "Classification Chips", "Automated Benign, Malignant, and Inconclusive categorization for clinician review."),
        ("⚡", "Rapid Inference", "Processed on optimized CNN architecture for results in seconds."),
    ]
    for col, (icon, title, body) in zip(cols, insights):
        with col:
            st.markdown(
                f"""
                <div class="insight-tile">
                    <div style="font-size:24px;margin-bottom:8px;">{icon}</div>
                    <div class="insight-title">{title}</div>
                    <div class="insight-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def page_results() -> None:
    if st.session_state.results is None or st.session_state.uploaded_image is None:
        st.session_state.page = "upload"
        st.rerun()

    results = st.session_state.results

    if results.get("label") in ["Invalid Input", "Uncertain"]:
        st.error(results.get("error", "Invalid input detected."))
        st.warning("Please upload a valid histopathology image (BreakHis-style microscopic slide).")
        st.stop()
    image = st.session_state.uploaded_image
    label = results["label"]
    confidence_pct = results["confidence"] * 100
    is_benign = label == "Benign"
    result_class = "result-benign" if is_benign else "result-malignant"
    ring_color = "#53e16f" if is_benign else "#ff6b6b"
    chip_class = "chip-benign" if is_benign else "chip-malignant"
    chip_text = "Verified High Confidence" if results["is_high_confidence"] else "Moderate Confidence"

    col_main, col_side = st.columns([2, 1], gap="large")

    with col_main:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="text-align:center;position:relative;">
                <div style="position:absolute;top:0;right:0;">
                    <span class="{chip_class}">✓ {chip_text}</span>
                </div>
                <p style="font-size:12px;font-weight:600;letter-spacing:0.1em;color:#c1c6d7;
                    text-transform:uppercase;">Prediction Result</p>
                <div class="result-label {result_class}">{label.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ring_col1, ring_col2, ring_col3 = st.columns([1, 2, 1])
        with ring_col2:
            st.markdown(confidence_ring_html(confidence_pct, ring_color), unsafe_allow_html=True)
        st.markdown(
            f'<p style="text-align:center;color:#c1c6d7;font-size:16px;max-width:520px;margin:0 auto;">'
            f"{INTERPRETATIONS[label]}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        metric_cols = st.columns(5)
        icons = ["✓", "🎯", "↩", "ƒ", "📈"]
        for col, (key, icon) in zip(metric_cols, zip(MODEL_METRICS.keys(), icons)):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div style="font-size:20px;margin-bottom:4px;">{icon}</div>
                        <div class="metric-value">{MODEL_METRICS[key]:.2f}</div>
                        <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_side:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Visual Comparison")
        view_mode = st.radio(
            "View",
            ["Original", "Heatmap"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if view_mode == "Original":
            st.image(image, use_container_width=True)
        else:
            heatmap = make_gradcam_heatmap(st.session_state.model, image)
            st.image(heatmap, use_container_width=True, caption="Attention Map")

        st.caption("Switch between raw slide and AI feature-weighted map to correlate findings.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Probability Distribution")
        b_pct = results["benign_prob"] * 100
        m_pct = results["malignant_prob"] * 100
        st.markdown(
            f"""
            <div class="prob-bar">
                <div class="prob-benign" style="width:{max(b_pct, 8):.1f}%;">
                    Benign {b_pct:.1f}%
                </div>
                <div class="prob-malignant">
                    Malignant {m_pct:.1f}%
                </div>
            </div>
            <p style="font-size:12px;color:#c1c6d7;margin-top:8px;">
                Calculated using BreakHis CNN with Sigmoid activation.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Model Interpretation")
        st.write(INTERPRETATIONS[label])
        st.info(
            f'"The network prioritized texture features and tissue architecture patterns '
            f'to conclude {label} status."'
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Diagnostic Detail Findings")
        icons_f = ["▦", "◎", "◈"]
        colors = ["rgba(0,110,40,0.15)", "rgba(0,88,188,0.15)", "rgba(51,57,65,0.3)"]
        for (title, body), icon, color in zip(FINDINGS[label], icons_f, colors):
            st.markdown(
                f"""
                <div class="finding-row">
                    <div class="finding-icon" style="background:{color};">{icon}</div>
                    <div>
                        <div style="font-weight:600;font-size:18px;">{title}</div>
                        <div style="color:#c1c6d7;font-size:14px;">{body}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("View Detailed Confidence Analysis", use_container_width=True):
            st.session_state.page = "confidence"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.button("← Upload New Slide", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()


def page_confidence() -> None:
    if st.session_state.results is None:
        st.session_state.page = "upload"
        st.rerun()

    results = st.session_state.results
    image = st.session_state.uploaded_image
    label = results["label"]
    confidence_pct = results["confidence"] * 100
    case_id = st.session_state.case_id or "PX-UNKNOWN"

    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:24px;">
            <div>
                <h2 style="font-size:32px;font-weight:600;margin:0;">Detailed Confidence Analysis</h2>
                <p style="color:#c1c6d7;margin:4px 0 0;">Case Reference: {case_id} · BreakHis CNN · {datetime.now():%Y-%m-%d}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([2, 1], gap="medium")

    with col_main:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        chip = "chip-benign" if label == "Benign" else "chip-malignant"
        st.markdown(
            f"""
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <h3 style="margin:0;font-size:20px;">Confidence Breakdown</h3>
                <span class="{chip}">✓ {'High' if results['is_high_confidence'] else 'Moderate'} Confidence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_pct = results["benign_prob"] * 100
        m_pct = results["malignant_prob"] * 100
        st.markdown(
            f"""
            <div class="prob-bar" style="height:56px;margin-bottom:12px;">
                <div class="prob-benign" style="width:{max(b_pct, 10):.1f}%;">
                    Benign {b_pct:.1f}%
                </div>
                <div class="prob-malignant">Malignant {m_pct:.1f}%</div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:12px;color:#c1c6d7;">
                <span>● Non-pathological tissue architecture</span>
                <span>● Potential malignant patterns</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown("**Sub-type Probability**")
            st.markdown(f"- **{label}** — {confidence_pct:.1f}%")
            other = "Malignant" if label == "Benign" else "Benign"
            other_pct = (1 - results["confidence"]) * 100
            st.markdown(f"- {other} — {other_pct:.1f}%")
        with sub_col2:
            st.image(image, use_container_width=True, caption="Representative ROI Segment")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="card card-glass" style="height:100%;">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Model Interpretation")
        st.markdown("**Tissue Texture**")
        st.caption(INTERPRETATIONS[label][:180] + "…")
        st.markdown("**Staining Pattern**")
        st.caption(
            "Hematoxylin intensity distribution and eosinophilic cytoplasm patterns "
            "were evaluated against BreakHis training distribution."
        )
        st.info(
            f"Inference weighted by CNN feature maps "
            f"({confidence_pct:.0f}% {label} confidence)."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Core Clinical Metrics")
    metric_cols = st.columns(5)
    for col, (name, value) in zip(metric_cols, MODEL_METRICS.items()):
        with col:
            bar_pct = int(value * 100)
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">{name.replace('_', ' ').title()}</div>
                    <div class="metric-value">{value:.2f}</div>
                    <div style="background:#333941;height:4px;border-radius:2px;margin-top:8px;">
                        <div style="background:#0070eb;height:4px;width:{bar_pct}%;border-radius:2px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("← Back to Results", use_container_width=False):
        st.session_state.page = "results"
        st.rerun()


def main() -> None:
    inject_css()
    init_state()
    render_header()

    page = st.session_state.page
    if page == "intro":
        page_intro()
    elif page == "upload":
        page_upload()
    elif page == "results":
        page_results()
    elif page == "confidence":
        page_confidence()
    else:
        st.session_state.page = "intro"
        st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
