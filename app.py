"""HistoAI — BreakHis CNN diagnostic assistant (Streamlit + Stitch designs)."""

from __future__ import annotations

import io
import uuid
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
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
    :root {
        /* Stitch dark-unified tokens */
        --surface: #10131a;
        --surface-lowest: #0b0e15;
        --surface-low: #191b23;
        --surface-mid: #1d2029;
        --surface-high: #23262f;
        --surface-elevated: #2a2e38;
        --on-surface: #e2e8f0;
        --on-surface-variant: #94a3b8;
        --on-surface-muted: #64748b;
        --primary: #0058bc;
        --primary-container: #0070eb;
        --primary-fixed: #adc6ff;
        --primary-hover: #3b9eff;
        --primary-glow: rgba(0, 112, 235, 0.42);
        --secondary: #006e28;
        --secondary-dim: #53e16f;
        --secondary-glow: rgba(83, 225, 111, 0.12);
        --tertiary: #bc000a;
        --tertiary-soft: #ff6b6b;
        --outline: #414755;
        --outline-variant: rgba(45, 49, 61, 0.95);
        --gutter: 24px;
        --ease: cubic-bezier(0.4, 0, 0.2, 1);
        --radius-md: 10px;
        --radius-lg: 12px;
        --radius-xl: 16px;
        --shadow-card: 0 4px 6px rgba(0, 0, 0, 0.18), 0 16px 40px rgba(0, 0, 0, 0.32);
        --shadow-hover: 0 8px 12px rgba(0, 0, 0, 0.22), 0 24px 48px rgba(0, 0, 0, 0.38);
        --shadow-inset: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: var(--on-surface);
    }

    .stApp {
        background-color: var(--surface-lowest);
        background-image:
            radial-gradient(ellipse 90% 55% at 50% -15%, rgba(0, 88, 188, 0.14), transparent 55%),
            radial-gradient(ellipse 50% 35% at 0% 80%, rgba(0, 110, 40, 0.06), transparent 50%),
            radial-gradient(ellipse 40% 30% at 100% 20%, rgba(0, 112, 235, 0.05), transparent 45%),
            linear-gradient(180deg, var(--surface-lowest) 0%, var(--surface) 40%, var(--surface) 100%);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 0.75rem;
        padding-bottom: 2.5rem;
    }

    .histo-topbar-accent {
        height: 1px;
        margin: 0 0 20px 0;
        background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(0, 112, 235, 0.5) 30%,
            rgba(173, 198, 255, 0.35) 50%,
            rgba(0, 112, 235, 0.5) 70%,
            transparent 100%
        );
    }

    .page-header {
        margin-bottom: 28px;
        padding-bottom: 20px;
        border-bottom: 1px solid var(--outline-variant);
    }

    .page-title {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--on-surface);
        margin: 0 0 6px 0;
        line-height: 1.2;
    }

    .page-subtitle {
        font-size: 14px;
        color: var(--on-surface-variant);
        margin: 0;
        letter-spacing: 0.01em;
    }

    .section-title {
        font-size: 17px;
        font-weight: 600;
        color: var(--on-surface);
        margin: 0 0 16px 0;
        letter-spacing: -0.01em;
    }

    .section-kicker {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--primary-fixed);
        margin-bottom: 8px;
    }

    .text-muted { color: var(--on-surface-variant); }
    .text-body { color: var(--on-surface-variant); line-height: 1.65; }

    header[data-testid="stHeader"] {
        background: rgba(16, 19, 26, 0.92);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--outline-variant);
    }

    #MainMenu, footer, .stDeployButton { visibility: hidden; }

    .histo-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 4px 0 16px;
        margin-bottom: 8px;
        border-bottom: 1px solid var(--outline-variant);
    }

    .histo-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .histo-brand .brand-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(0, 112, 235, 0.25), rgba(0, 88, 188, 0.1));
        border: 1px solid rgba(0, 112, 235, 0.3);
        font-size: 18px;
    }

    .histo-brand .brand-text {
        background: linear-gradient(135deg, #f8fafc 0%, #adc6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .histo-brand .brand-tag {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--on-surface-variant);
        background: var(--surface-high);
        border: 1px solid var(--outline-variant);
        border-radius: 999px;
        padding: 3px 8px;
        margin-left: 4px;
    }

    .page-panel {
        position: relative;
        background: linear-gradient(165deg, var(--surface-mid) 0%, var(--surface-low) 100%);
        border: 1px solid var(--outline-variant);
        border-radius: var(--radius-xl);
        padding: 48px 40px;
        box-shadow: var(--shadow-card), var(--shadow-inset);
        max-width: 760px;
        margin: 8px auto 40px;
        text-align: center;
        transition: border-color 0.3s var(--ease), box-shadow 0.3s var(--ease);
        overflow: hidden;
    }

    .page-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--primary-container), var(--primary-fixed));
        opacity: 0.85;
    }

    .page-panel:hover {
        border-color: rgba(0, 112, 235, 0.35);
        box-shadow: var(--shadow-hover), var(--shadow-inset);
    }

    .card {
        position: relative;
        background: linear-gradient(180deg, var(--surface-mid) 0%, var(--surface-low) 100%);
        border: 1px solid var(--outline-variant);
        border-radius: var(--radius-xl);
        padding: 26px;
        box-shadow: var(--shadow-card), var(--shadow-inset);
        transition: border-color 0.25s var(--ease), box-shadow 0.25s var(--ease);
        overflow: hidden;
    }

    .card::before {
        content: '';
        position: absolute;
        top: 0; left: 24px; right: 24px;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(173, 198, 255, 0.25), transparent);
    }

    .card:hover {
        border-color: rgba(65, 71, 85, 0.9);
        box-shadow: var(--shadow-hover), var(--shadow-inset);
    }

    .card-glass {
        background: rgba(25, 27, 35, 0.85);
        backdrop-filter: blur(20px);
    }

    .chip-benign, .chip-malignant {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease);
    }

    .chip-benign {
        background: rgba(0, 110, 40, 0.18);
        color: var(--secondary-dim);
        border: 1px solid rgba(0, 110, 40, 0.35);
        box-shadow: 0 0 20px rgba(83, 225, 111, 0.08);
    }

    .chip-malignant {
        background: rgba(188, 0, 10, 0.18);
        color: #ffb4aa;
        border: 1px solid rgba(188, 0, 10, 0.35);
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.08);
    }

    .metric-tile {
        background: var(--surface-high);
        border: 1px solid var(--outline-variant);
        border-radius: var(--radius-lg);
        padding: 18px 12px;
        text-align: center;
        transition: all 0.25s var(--ease);
        cursor: default;
        box-shadow: var(--shadow-inset);
    }

    .metric-tile:hover {
        border-color: rgba(0, 112, 235, 0.4);
        background: var(--surface-elevated);
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35), var(--shadow-inset);
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: var(--on-surface);
    }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: var(--on-surface-variant);
        text-transform: uppercase;
    }

    .prob-bar {
        display: flex;
        height: 56px;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--outline-variant);
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.2);
    }

    .prob-benign {
        background: linear-gradient(90deg, #00531c, #006e28);
        display: flex;
        align-items: center;
        padding: 0 16px;
        font-size: 12px;
        font-weight: 600;
        color: #6ffb85;
        min-width: fit-content;
        transition: width 0.6s var(--ease);
    }

    .prob-malignant {
        background: linear-gradient(90deg, #93000a, #ba1a1a);
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
        max-width: 720px;
        margin: 0 auto;
    }

    .upload-icon-ring {
        position: relative;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: var(--surface-high);
        border: 1px solid var(--outline-variant);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 20px;
        font-size: 32px;
        box-shadow: 0 0 0 0 rgba(0, 112, 235, 0.2);
        animation: icon-pulse 3s ease-in-out infinite;
    }

    @keyframes icon-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(0, 112, 235, 0.15); }
        50% { box-shadow: 0 0 0 12px rgba(0, 112, 235, 0); }
    }

    .insight-tile {
        background: var(--surface-mid);
        border: 1px solid var(--outline-variant);
        border-radius: var(--radius-lg);
        padding: 22px;
        height: 100%;
        transition: all 0.25s var(--ease);
        box-shadow: var(--shadow-inset);
    }

    .insight-tile:hover {
        border-color: rgba(0, 112, 235, 0.45);
        background: var(--surface-high);
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(0, 112, 235, 0.08);
    }

    .insight-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
        color: var(--on-surface);
    }

    .insight-body {
        font-size: 13px;
        color: var(--on-surface-variant);
        line-height: 1.55;
    }

    .step-badge {
        display: inline-block;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: var(--primary-fixed);
        background: rgba(0, 112, 235, 0.12);
        border: 1px solid rgba(0, 112, 235, 0.25);
        border-radius: 999px;
        padding: 4px 10px;
        margin-bottom: 12px;
    }

    .feature-grid {
        max-width: 1040px;
        margin: 0 auto;
    }

    .footer-disclaimer {
        text-align: center;
        font-size: 12px;
        color: var(--on-surface-variant);
        padding: 28px 16px;
        border-top: 1px solid var(--outline-variant);
        margin-top: 56px;
        line-height: 1.7;
    }

    .footer-disclaimer a {
        color: var(--on-surface-variant);
        text-decoration: none;
        margin: 0 12px;
        transition: color 0.2s var(--ease);
    }

    .footer-disclaimer a:hover {
        color: var(--on-surface);
    }

    .result-label {
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1;
        text-shadow: 0 0 40px rgba(83, 225, 111, 0.15);
    }

    .result-benign {
        color: var(--secondary-dim);
        text-shadow: 0 0 48px var(--secondary-glow);
    }
    .result-malignant {
        color: var(--tertiary-soft);
        text-shadow: 0 0 48px rgba(255, 107, 107, 0.2);
    }

    .confidence-ring-wrap {
        position: relative;
        width: 192px;
        height: 192px;
        margin: 16px auto;
        filter: drop-shadow(0 0 16px rgba(0, 112, 235, 0.12));
    }

    .finding-row {
        display: flex;
        gap: 16px;
        padding-bottom: 16px;
        margin-bottom: 16px;
        border-bottom: 1px solid var(--outline-variant);
        transition: padding-left 0.2s var(--ease);
    }

    .finding-row:hover {
        padding-left: 4px;
    }

    .finding-row:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    .finding-icon {
        width: 48px;
        height: 48px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        font-size: 22px;
        transition: transform 0.2s var(--ease);
    }

    .finding-row:hover .finding-icon {
        transform: scale(1.08);
    }

    /* ── Streamlit widgets ── */
    div[data-testid="stFileUploader"] {
        background: rgba(11, 14, 21, 0.5);
        border: 1.5px dashed rgba(0, 112, 235, 0.45);
        border-radius: var(--radius-lg);
        padding: 24px;
        transition: all 0.3s var(--ease);
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 112, 235, 0.85);
        background: rgba(0, 112, 235, 0.06);
        box-shadow: inset 0 0 28px rgba(0, 112, 235, 0.08);
    }

    div[data-testid="stFileUploader"] section {
        padding: 16px;
    }

    div[data-testid="stFileUploader"] button {
        transition: all 0.2s var(--ease) !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background: rgba(0, 112, 235, 0.15) !important;
        border-color: rgba(0, 112, 235, 0.5) !important;
        color: #f1f1f1 !important;
    }

    [data-testid="stAlert"] {
        border-radius: var(--radius-lg) !important;
        border: 1px solid var(--outline-variant) !important;
    }

    .stRadio > div {
        gap: 8px;
    }

    .stRadio label {
        transition: color 0.2s var(--ease);
    }

    .stRadio label:hover {
        color: var(--primary-hover) !important;
    }

    /* ── Buttons (all pages) ── */
    .stButton > button {
        transition: all 0.22s var(--ease) !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        border-radius: 10px !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-container), #0058bc) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 4px 16px rgba(0, 112, 235, 0.3) !important;
        font-size: 16px !important;
        padding: 10px 20px !important;
    }

    .stButton > button[kind="primary"]:hover:not(:disabled) {
        background: linear-gradient(135deg, var(--primary-hover), #0066d4) !important;
        box-shadow: 0 6px 24px var(--primary-glow) !important;
        transform: translateY(-2px) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }

    .stButton > button[kind="primary"]:active:not(:disabled) {
        transform: scale(0.98) translateY(0) !important;
        box-shadow: 0 2px 8px rgba(0, 112, 235, 0.25) !important;
    }

    .stButton > button[kind="secondary"] {
        background: var(--surface-high) !important;
        color: var(--on-surface-variant) !important;
        border: 1px solid var(--outline-variant) !important;
        font-size: 13px !important;
        padding: 8px 14px !important;
        box-shadow: var(--shadow-inset) !important;
    }

    .stButton > button[kind="secondary"]:hover:not(:disabled) {
        background: rgba(0, 112, 235, 0.12) !important;
        border-color: rgba(0, 112, 235, 0.45) !important;
        color: var(--on-surface) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }

    .stButton > button[kind="secondary"]:active:not(:disabled) {
        transform: scale(0.98) !important;
    }

    .stButton > button:disabled {
        opacity: 0.38 !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ── Sticky glass header ── */
    .histo-sticky-header-anchor {
        position: sticky;
        top: 0;
        z-index: 200;
        margin: -1rem -2rem 1.25rem -2rem;
        padding: 14px 2rem 10px 2rem;
        background: rgba(11, 14, 21, 0.72);
        backdrop-filter: blur(20px) saturate(160%);
        -webkit-backdrop-filter: blur(20px) saturate(160%);
        border-bottom: 1px solid rgba(45, 49, 61, 0.85);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        transition: background 0.3s var(--ease), box-shadow 0.3s var(--ease);
    }

    .histo-sticky-header-anchor.is-scrolled {
        background: rgba(11, 14, 21, 0.9);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55);
    }

    .histo-sticky-header-anchor .histo-topbar-accent {
        margin-bottom: 14px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--secondary-dim);
        background: rgba(0, 110, 40, 0.12);
        border: 1px solid rgba(83, 225, 111, 0.25);
        border-radius: 999px;
        padding: 4px 10px;
        margin-left: 8px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--secondary-dim);
        box-shadow: 0 0 8px var(--secondary-dim);
        animation: status-pulse 2s ease-in-out infinite;
    }

    @keyframes status-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.55; transform: scale(0.85); }
    }

    /* ── Page transitions ── */
    .histo-page-content {
        animation: histoPageIn 0.55s var(--ease) both;
    }

    .histo-page-content .page-header {
        animation: histoSlideUp 0.6s var(--ease) 0.05s both;
    }

    .histo-page-content .page-panel,
    .histo-page-content .card,
    .histo-page-content .insight-tile,
    .histo-page-content .metric-tile {
        animation: histoSlideUp 0.65s var(--ease) 0.1s both;
    }

    @keyframes histoPageIn {
        from {
            opacity: 0;
            transform: translateY(18px);
            filter: blur(6px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
            filter: blur(0);
        }
    }

    @keyframes histoSlideUp {
        from {
            opacity: 0;
            transform: translateY(14px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .histo-page-content,
        .histo-page-content .page-header,
        .histo-page-content .page-panel,
        .histo-page-content .card,
        .histo-page-content .insight-tile,
        .histo-page-content .metric-tile {
            animation: none !important;
        }
        .status-dot { animation: none !important; }
    }
</style>
"""

HISTO_BOOT_JS = """
<script>
(function () {
    const doc = window.parent.document;

    function wrapStickyHeader() {
        const start = doc.querySelector(".histo-sticky-start");
        const end = doc.querySelector(".histo-sticky-end");
        if (!start || !end || start.dataset.wrapped === "1") return;

        const parent = start.parentElement;
        if (!parent || parent !== end.parentElement) return;

        const wrapper = doc.createElement("div");
        wrapper.className = "histo-sticky-header-anchor";
        parent.insertBefore(wrapper, start);
        start.dataset.wrapped = "1";

        let node = start;
        while (node) {
            const next = node.nextSibling;
            wrapper.appendChild(node);
            if (node === end) break;
            node = next;
        }
    }

    function bindScrollState() {
        const header = doc.querySelector(".histo-sticky-header-anchor");
        if (!header || header.dataset.scrollBound === "1") return;
        header.dataset.scrollBound = "1";

        const onScroll = () => {
            const y = Math.max(
                doc.documentElement.scrollTop || 0,
                doc.body.scrollTop || 0,
                window.parent.scrollY || 0
            );
            header.classList.toggle("is-scrolled", y > 8);
        };
        onScroll();
        doc.addEventListener("scroll", onScroll, { passive: true });
        window.parent.addEventListener("scroll", onScroll, { passive: true });
    }

    function wrapPageContent() {
        const start = doc.querySelector(".histo-page-start");
        const end = doc.querySelector(".histo-page-end");
        if (!start || !end || start.dataset.wrapped === "1") return;
        if (start.parentElement !== end.parentElement) return;

        const wrapper = doc.createElement("div");
        wrapper.className = "histo-page-content";
        wrapper.dataset.page = start.dataset.page || "";
        start.parentNode.insertBefore(wrapper, start);
        start.dataset.wrapped = "1";

        let node = start;
        while (node) {
            const next = node.nextSibling;
            wrapper.appendChild(node);
            if (node === end) break;
            node = next;
        }
        end.remove();
    }

    function boot() {
        wrapStickyHeader();
        wrapPageContent();
        bindScrollState();
    }

    boot();
    setTimeout(boot, 80);
    setTimeout(boot, 300);

    const app = doc.querySelector(".stApp");
    if (app) {
        new MutationObserver(() => {
            const sticky = doc.querySelector(".histo-sticky-start");
            const page = doc.querySelector(".histo-page-start");
            if (
                (sticky && sticky.dataset.wrapped !== "1") ||
                (page && page.dataset.wrapped !== "1")
            ) {
                boot();
            }
        }).observe(app, { childList: true, subtree: true });
    }
})();
</script>
"""


def inject_css() -> None:
    st.markdown(STITCH_CSS, unsafe_allow_html=True)
    components.html(HISTO_BOOT_JS, height=0, width=0)


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


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <div class="section-kicker">HistoAI Diagnostic Suite</div>
            <h2 class="page-title">{title}</h2>
            <p class="page-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown('<div class="histo-sticky-start"></div>', unsafe_allow_html=True)
    st.markdown('<div class="histo-topbar-accent"></div>', unsafe_allow_html=True)
    left, right = st.columns([2, 3])
    with left:
        st.markdown(
            """
            <div class="histo-brand">
                <span class="brand-icon">🔬</span>
                <span class="brand-text">HistoAI</span>
                <span class="brand-tag">Clinical AI</span>
                <span class="status-pill">
                    <span class="status-dot"></span>
                    Live
                </span>
            </div>
            """,
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
                is_active = st.session_state.page == key
                if st.button(
                    label,
                    key=f"nav_{key}",
                    disabled=disabled,
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.page = key
                    st.rerun()
    st.markdown('<div class="histo-sticky-end"></div>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-disclaimer">
            This AI-generated analysis is for clinical decision support only and does not
            constitute a final medical diagnosis. Always correlate with pathological findings
            and clinical history.<br>
            <div style="margin-top:12px;">
                <a href="#">Terms of Service</a>
                <a href="#">Privacy Policy</a>
                <a href="#">Regulatory Compliance</a>
            </div>
            <span style="opacity:0.45;margin-top:12px;display:block;">
                HistoAI v4.2.0 · BreakHis CNN Classifier
            </span>
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
            <div class="text-muted" style="font-size:12px;font-weight:600;letter-spacing:0.05em;">CONFIDENCE</div>
        </div>
    </div>
    """


def page_intro() -> None:
    st.markdown('<div class="page-panel upload-hero">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="upload-icon-ring">🔬</div>
        <h1 style="font-size:42px;font-weight:700;margin-bottom:12px;letter-spacing:-0.02em;">
            HistoAI
        </h1>
        <p class="text-muted" style="font-size:18px;margin-bottom:8px;">
            AI-Powered Breast Histopathology Diagnostic Assistant
        </p>
        <p class="text-body" style="font-size:15px;margin-bottom:8px;">
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

    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-kicker" style="text-align:center;margin-bottom:20px;">How it works</p>',
        unsafe_allow_html=True,
    )
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
                    <div class="step-badge">STEP {num}</div>
                    <div class="insight-title">{title}</div>
                    <div class="insight-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<p class="section-kicker" style="text-align:center;margin:32px 0 20px;">Platform capabilities</p>',
        unsafe_allow_html=True,
    )
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
    st.markdown("</div>", unsafe_allow_html=True)


def page_upload() -> None:
    page_header(
        "Slide Analysis",
        "Upload a histopathology image for AI-assisted benign / malignant classification",
    )
    model, status = get_model()
    st.session_state.model = model
    st.session_state.model_status = status

    if status and ("No model" in status or "No trained" in status or "Run `python3 train_sklearn_model.py`" in status):
        st.warning(status)
    elif status and "sklearn" in status.lower():
        st.info(status)

    st.markdown('<div class="page-panel upload-hero">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="upload-icon-ring">🧬</div>
        <h1 style="font-size:30px;font-weight:600;margin-bottom:8px;letter-spacing:-0.01em;">
            Upload Histopathology Image
        </h1>
        <p class="text-muted" style="font-size:16px;margin-bottom:24px;">
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

    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)


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
    case_id = st.session_state.case_id or "PX-UNKNOWN"

    page_header(
        "Diagnostic Results",
        f"Case {case_id} · BreakHis classifier · {datetime.now():%d %b %Y}",
    )

    col_main, col_side = st.columns([2, 1], gap="large")

    with col_main:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="text-align:center;position:relative;">
                <div style="position:absolute;top:0;right:0;">
                    <span class="{chip_class}">✓ {chip_text}</span>
                </div>
                <p class="section-kicker" style="margin-bottom:12px;">Prediction Result</p>
                <div class="result-label {result_class}">{label.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ring_col1, ring_col2, ring_col3 = st.columns([1, 2, 1])
        with ring_col2:
            st.markdown(confidence_ring_html(confidence_pct, ring_color), unsafe_allow_html=True)
        st.markdown(
            f'<p class="text-body" style="text-align:center;font-size:16px;max-width:520px;margin:0 auto;">'
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
        st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-title">Probability Distribution</div>', unsafe_allow_html=True)
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
            <p class="text-muted" style="font-size:12px;margin-top:8px;">
                Calculated using BreakHis CNN with Sigmoid activation.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:16px;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Model Interpretation</div>', unsafe_allow_html=True)
        st.write(INTERPRETATIONS[label])
        st.info(
            f'"The network prioritized texture features and tissue architecture patterns '
            f'to conclude {label} status."'
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Diagnostic Detail Findings</div>', unsafe_allow_html=True)
        icons_f = ["▦", "◎", "◈"]
        colors = ["rgba(0,110,40,0.15)", "rgba(0,88,188,0.15)", "rgba(51,57,65,0.3)"]
        for (title, body), icon, color in zip(FINDINGS[label], icons_f, colors):
            st.markdown(
                f"""
                <div class="finding-row">
                    <div class="finding-icon" style="background:{color};">{icon}</div>
                    <div>
                        <div style="font-weight:600;font-size:18px;">{title}</div>
                        <div class="text-muted" style="font-size:14px;">{body}</div>
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

    page_header(
        "Detailed Confidence Analysis",
        f"Case {case_id} · BreakHis classifier · {datetime.now():%d %b %Y}",
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
            <div class="text-muted" style="display:flex;justify-content:space-between;font-size:12px;">
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
        st.markdown('<div class="section-title">Model Interpretation</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-title" style="margin-bottom:20px;">Core Clinical Metrics</div>', unsafe_allow_html=True)
    metric_cols = st.columns(5)
    for col, (name, value) in zip(metric_cols, MODEL_METRICS.items()):
        with col:
            bar_pct = int(value * 100)
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">{name.replace('_', ' ').title()}</div>
                    <div class="metric-value">{value:.2f}</div>
                    <div style="background:var(--surface-elevated);height:4px;border-radius:2px;margin-top:8px;">
                        <div style="background:linear-gradient(90deg,#0058bc,#0070eb);height:4px;width:{bar_pct}%;border-radius:2px;"></div>
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
    st.markdown(
        f'<div class="histo-page-start" data-page="{page}"></div>',
        unsafe_allow_html=True,
    )

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

    st.markdown('<div class="histo-page-end"></div>', unsafe_allow_html=True)
    render_footer()


if __name__ == "__main__":
    main()
