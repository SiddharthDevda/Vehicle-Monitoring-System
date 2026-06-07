import streamlit as st
import cv2
import numpy as np
import tempfile
import pandas as pd
from services.pipeline import process_image

@st.cache_resource
def load_webrtc_libs():
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
    return webrtc_streamer, VideoProcessorBase, av

st.set_page_config(
    page_title="VMS · Vehicle Monitoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:         #070b14;
    --bg2:        #0d1117;
    --bg3:        #161d2b;
    --bg4:        #1c2537;
    --blue:       #2563eb;
    --blue2:      #3b82f6;
    --blue3:      #60a5fa;
    --blue-glow:  rgba(37,99,235,0.35);
    --blue-dim:   rgba(59,130,246,0.1);
    --blue-bdr:   rgba(59,130,246,0.3);
    --cyan:       #06b6d4;
    --cyan-dim:   rgba(6,182,212,0.1);
    --text:       #e2e8f0;
    --text2:      #94a3b8;
    --text3:      #475569;
    --bdr:        rgba(255,255,255,0.06);
    --bdr2:       rgba(255,255,255,0.1);
    --green:      #10b981;
    --green-dim:  rgba(16,185,129,0.12);
    --green-bdr:  rgba(16,185,129,0.35);
    --red:        #ef4444;
    --red-dim:    rgba(239,68,68,0.12);
    --red-bdr:    rgba(239,68,68,0.35);
    --yellow:     #f59e0b;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ═══════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #0a0e1a !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    min-width: 272px !important;
    max-width: 272px !important;
    box-shadow: 8px 0 40px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebarContent"],
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

[data-testid="stSidebarCollapseButton"] {
    background: rgba(37,99,235,0.9) !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 0 20px rgba(37,99,235,0.4) !important;
    width: 34px !important; height: 34px !important;
    position: fixed !important;
    top: 14px !important; left: 14px !important;
    z-index: 9999 !important;
}
[data-testid="stSidebarCollapseButton"] svg { display: none !important; }
[data-testid="stSidebarCollapseButton"]::after {
    content: '☰'; font-size: 1rem; color: #fff; line-height: 1;
}
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stBaseButton-headerNoPadding"] {
    opacity: 1 !important; visibility: visible !important;
    background: rgba(37,99,235,0.9) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] {
    transform: none !important; visibility: visible !important; opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: transparent !important;
    color: #64748b !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 10px !important;
    margin: 0.1rem 0.8rem !important;
    padding: 0.65rem 0.9rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: none !important;
    width: calc(100% - 1.8rem) !important;
    letter-spacing: -0.01em !important;
    transition: all 0.15s ease !important;
    min-height: 44px !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #e2e8f0 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── MAIN PADDING ── */
[data-testid="stMain"] > div { padding: 0 !important; }
.main-content { padding: 1.8rem 2.2rem; }

/* ═══════════════════════════════════════════
   SIDEBAR COMPONENTS
═══════════════════════════════════════════ */
.sb-brand {
    padding: 1.4rem 1.2rem 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: linear-gradient(160deg, rgba(37,99,235,0.1) 0%, transparent 70%);
}
.sb-logo-row {
    display: flex; align-items: center; gap: 0.9rem; margin-bottom: 1rem;
}
.sb-icon {
    width: 46px; height: 46px; flex-shrink: 0;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 0 20px rgba(37,99,235,0.45);
}
.sb-name {
    font-size: 1.05rem; font-weight: 800;
    color: #f1f5f9; letter-spacing: -0.02em; line-height: 1.2;
}
.sb-sub {
    font-size: 0.62rem; color: #475569;
    margin-top: 0.15rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.09em;
}
.sb-live {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 20px;
    padding: 0.22rem 0.7rem;
    font-size: 0.63rem; font-weight: 700;
    color: #10b981; letter-spacing: 0.06em;
    text-transform: uppercase;
}
.sb-live-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #10b981; box-shadow: 0 0 5px #10b981;
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

.sb-section {
    padding: 1.1rem 1.2rem 0.35rem;
    font-size: 0.58rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.18em; color: #334155;
}
.sb-active {
    display: flex; align-items: center; gap: 0.55rem;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.3);
    border-radius: 9px;
    margin: 0.15rem 0.8rem;
    padding: 0.65rem 0.9rem;
    font-size: 0.88rem; font-weight: 700;
    color: #93c5fd; cursor: default;
}
.sb-divider { height: 1px; background: rgba(255,255,255,0.05); margin: 0.5rem 1.2rem; }
.sb-sysinfo { padding: 0.7rem 1.2rem; }
.sb-sysinfo-title {
    font-size: 0.55rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.18em;
    color: #334155; margin-bottom: 0.55rem;
}
.sb-sysinfo-row {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.38rem;
}
.sb-sysinfo-key { font-size: 0.73rem; color: #475569; }
.sb-sysinfo-val { font-size: 0.7rem; color: #10b981; font-weight: 600; }
.sb-footer {
    padding: 0.8rem 1.2rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    display: flex; justify-content: space-between; align-items: center;
}
.sb-footer-txt { font-size: 0.62rem; color: #334155; }
.sb-version {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 4px; padding: 0.1rem 0.45rem;
    font-size: 0.6rem; color: #475569; font-family: 'JetBrains Mono', monospace;
}

/* ═══════════════════════════════════════════
   PAGE HEADER
═══════════════════════════════════════════ */
.page-top {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.4rem;
    border-bottom: 1px solid var(--bdr);
    flex-wrap: wrap;
    gap: 0.75rem;
}
.page-title {
    font-size: 2.2rem; font-weight: 800;
    color: var(--text); letter-spacing: -0.035em; line-height: 1;
}
.page-title b {
    background: linear-gradient(90deg, var(--blue2), var(--cyan));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.page-sub {
    font-size: 0.9rem; color: var(--text2); margin-top: 0.5rem; font-weight: 400;
}
.page-badge {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: var(--blue-dim);
    border: 1px solid var(--blue-bdr);
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-size: 0.75rem; font-weight: 700;
    color: var(--blue3); letter-spacing: 0.06em; text-transform: uppercase;
    white-space: nowrap;
}

/* ═══════════════════════════════════════════
   SECTION LABEL
═══════════════════════════════════════════ */
.s-label {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.16em;
    color: var(--text2); margin-bottom: 0.85rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.s-label::after {
    content: ''; flex: 1; height: 1px; background: var(--bdr);
}

/* ═══════════════════════════════════════════
   STAT CARDS — 4-col desktop, 2-col mobile
═══════════════════════════════════════════ */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.85rem; margin-bottom: 1.8rem;
}
.stat-card {
    background: var(--bg2);
    border: 1px solid var(--bdr);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.stat-card:hover { border-color: var(--bdr2); transform: translateY(-2px); }
.stat-card::after {
    content: ''; position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(circle at top left, rgba(37,99,235,0.05), transparent 65%);
}
.stat-card-b { border-top: 2px solid var(--blue2); }
.stat-card-g { border-top: 2px solid var(--green); }
.stat-card-r { border-top: 2px solid var(--red); }
.stat-card-c { border-top: 2px solid var(--cyan); }
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem; font-weight: 700; line-height: 1;
}
.stat-num-b { color: var(--blue3); }
.stat-num-g { color: var(--green); }
.stat-num-r { color: var(--red); }
.stat-num-c { color: var(--cyan); }
.stat-lbl { font-size: 0.7rem; color: var(--text2); margin-top: 0.4rem; font-weight: 500; }
.stat-ico { position: absolute; top: 1rem; right: 1rem; font-size: 1.3rem; opacity: 0.15; }

/* ═══════════════════════════════════════════
   IMAGE PANELS
═══════════════════════════════════════════ */
.img-panel {
    background: var(--bg2); border: 1px solid var(--bdr);
    border-radius: 12px; overflow: hidden; transition: border-color 0.2s;
}
.img-panel:hover { border-color: var(--bdr2); }
.img-hdr {
    padding: 0.65rem 1rem;
    font-size: 0.68rem; font-weight: 700;
    color: var(--text2); text-transform: uppercase; letter-spacing: 0.08em;
    background: var(--bg3); border-bottom: 1px solid var(--bdr);
    display: flex; align-items: center; gap: 0.4rem;
}
.img-hdr-dot { width: 7px; height: 7px; border-radius: 50%; }

/* ═══════════════════════════════════════════
   RESULT CARDS
═══════════════════════════════════════════ */
.results-wrap {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 0.85rem; margin-top: 0.5rem;
}
.r-card {
    background: var(--bg2); border: 1px solid var(--bdr);
    border-radius: 12px; padding: 1.2rem 1.3rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.r-card:hover { border-color: var(--blue-bdr); transform: translateY(-2px); }
.r-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--blue), var(--cyan));
}
.r-vidx { font-size: 0.6rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.6rem; }
.r-type-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: var(--blue-dim); border: 1px solid var(--blue-bdr);
    border-radius: 5px; padding: 0.18rem 0.6rem;
    font-size: 0.65rem; font-weight: 700; color: var(--blue3);
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.75rem;
}
.r-plate {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem; font-weight: 700; color: var(--text);
    letter-spacing: 0.08em; margin-bottom: 0.7rem;
    padding: 0.4rem 0.7rem;
    background: var(--bg3); border-radius: 6px; border: 1px solid var(--bdr2);
}
.r-plate-miss { font-size: 0.82rem; color: var(--text3); font-family: 'Inter', sans-serif; font-weight: 400; margin-bottom: 0.7rem; font-style: italic; }
.r-belt { display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.28rem 0.75rem; border-radius: 6px; font-size: 0.72rem; font-weight: 700; }
.r-belt-on  { background: var(--green-dim); border: 1px solid var(--green-bdr); color: var(--green); }
.r-belt-off { background: var(--red-dim);   border: 1px solid var(--red-bdr);   color: var(--red); }

/* ═══════════════════════════════════════════
   UPLOAD ZONE
═══════════════════════════════════════════ */
[data-testid="stFileUploader"] > div { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stFileUploader"] label {
    min-height: 170px !important;
    border: 2px dashed rgba(37,99,235,0.3) !important;
    border-radius: 12px !important;
    background: rgba(37,99,235,0.04) !important;
    transition: all 0.2s !important; cursor: pointer !important;
}
[data-testid="stFileUploader"] label:hover { border-color: var(--blue2) !important; background: var(--blue-dim) !important; }

/* ═══════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════ */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--blue), #1d4ed8) !important;
    color: #fff !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
    transition: all 0.2s !important;
    min-height: 44px !important;
}
[data-testid="stButton"] > button:hover {
    box-shadow: 0 6px 20px rgba(37,99,235,0.5) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button {
    background: var(--bg3) !important; color: var(--text2) !important;
    border: 1px solid var(--bdr2) !important; box-shadow: none !important;
    min-height: 44px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--blue2) !important; color: var(--blue3) !important;
    transform: none !important; box-shadow: none !important;
}

/* ═══════════════════════════════════════════
   EMPTY STATE
═══════════════════════════════════════════ */
.empty-box {
    text-align: center; padding: 3.5rem 2rem;
    background: var(--bg2); border: 1px dashed rgba(255,255,255,0.08);
    border-radius: 12px; color: var(--text3); font-size: 0.875rem;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.75rem; opacity: 0.35; display: block; }

/* ═══════════════════════════════════════════
   ABOUT PAGE
═══════════════════════════════════════════ */
.about-hero {
    background: linear-gradient(135deg, var(--bg3) 0%, var(--bg4) 100%);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 16px; padding: 2.2rem 2.4rem;
    margin-bottom: 1.8rem; position: relative; overflow: hidden;
}
.about-hero::before {
    content: '🛡️'; position: absolute;
    right: 2.2rem; top: 50%; transform: translateY(-50%);
    font-size: 7rem; opacity: 0.05;
}
.about-hero-glow {
    position: absolute; top: -60px; right: -60px;
    width: 200px; height: 200px; border-radius: 50%;
    background: radial-gradient(circle, rgba(37,99,235,0.2), transparent 70%);
    pointer-events: none;
}
.about-title { font-size: 1.4rem; font-weight: 800; color: var(--text); letter-spacing: -0.02em; margin-bottom: 0.5rem; }
.about-desc { font-size: 0.875rem; color: var(--text2); line-height: 1.75; max-width: 560px; }

.feat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 0.85rem; }
.feat-card {
    background: var(--bg2); border: 1px solid var(--bdr);
    border-radius: 12px; padding: 1.2rem;
    transition: border-color 0.2s, transform 0.2s; position: relative; overflow: hidden;
}
.feat-card:hover { border-color: rgba(37,99,235,0.3); transform: translateY(-2px); }
.feat-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(37,99,235,0.4), transparent); }
.feat-icon { font-size: 1.6rem; margin-bottom: 0.7rem; }
.feat-title { font-size: 0.88rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem; }
.feat-desc  { font-size: 0.76rem; color: var(--text2); line-height: 1.65; }

.help-3col { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.85rem; }
.help-card { background: var(--bg2); border: 1px solid var(--bdr); border-radius: 12px; padding: 1.2rem; }
.help-card-title { font-size: 0.82rem; font-weight: 700; color: var(--blue3); margin-bottom: 0.9rem; padding-bottom: 0.7rem; border-bottom: 1px solid var(--bdr); display: flex; align-items: center; gap: 0.4rem; }
.help-step { display: flex; gap: 0.7rem; margin-bottom: 0.65rem; align-items: flex-start; }
.help-n { width: 20px; height: 20px; border-radius: 5px; background: var(--blue-dim); border: 1px solid var(--blue-bdr); color: var(--blue3); font-size: 0.6rem; font-weight: 800; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px; font-family: 'JetBrains Mono', monospace; }
.help-t { font-size: 0.79rem; color: var(--text2); line-height: 1.6; }

.tech-bar { background: var(--bg2); border: 1px solid var(--bdr); border-radius: 12px; padding: 1.1rem 1.3rem; display: flex; flex-wrap: wrap; gap: 0.45rem; }
.tech-tag { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.22rem 0.7rem; background: var(--bg3); border: 1px solid var(--bdr2); border-radius: 5px; font-size: 0.7rem; font-weight: 500; color: var(--text2); }

/* ═══════════════════════════════════════════
   DATA EDITOR / ALERT
═══════════════════════════════════════════ */
[data-testid="stDataFrame"] { background: var(--bg2) !important; border-radius: 12px !important; border: 1px solid var(--bdr) !important; }
[data-testid="stAlert"] { background: var(--blue-dim) !important; border: 1px solid var(--blue-bdr) !important; border-radius: 10px !important; color: var(--blue3) !important; }

/* ═══════════════════════════════════════════
   MOBILE — ≤ 768px
═══════════════════════════════════════════ */
@media (max-width: 768px) {

    .main-content { padding: 1rem 0.9rem !important; }

    /* Page header stacks */
    .page-top { flex-direction: column; align-items: flex-start; margin-bottom: 1.2rem; padding-bottom: 1rem; }
    .page-title { font-size: 1.6rem !important; }
    .page-sub   { font-size: 0.8rem !important; }
    .page-badge { font-size: 0.68rem !important; padding: 0.4rem 0.9rem !important; }

    /* Stats: 2 columns */
    .stats-row { grid-template-columns: repeat(2, 1fr) !important; gap: 0.6rem !important; }
    .stat-num  { font-size: 1.7rem !important; }
    .stat-lbl  { font-size: 0.65rem !important; }
    .stat-card { padding: 0.85rem 0.9rem !important; }

    /* Feature grid: 1 column */
    .feat-grid { grid-template-columns: 1fr !important; }

    /* Help cards: 1 column */
    .help-3col { grid-template-columns: 1fr !important; }

    /* Result cards: full width */
    .results-wrap { grid-template-columns: 1fr !important; }

    /* About hero */
    .about-hero { padding: 1.4rem 1.2rem !important; }
    .about-hero::before { display: none; }
    .about-title { font-size: 1.15rem !important; }
    .about-desc  { font-size: 0.8rem !important; }

    /* Remove hardcoded side padding from image column wrappers */
    .img-col-left  { padding: 0 !important; }
    .img-col-right { padding: 0 !important; }

    /* Upload zone shorter */
    [data-testid="stFileUploader"] label { min-height: 110px !important; }

    /* Plate font */
    .r-plate { font-size: 1.1rem !important; }

    /* Section label */
    .s-label { font-size: 0.6rem !important; }

    /* Tech tags */
    .tech-bar { padding: 0.8rem !important; gap: 0.35rem !important; }
    .tech-tag { font-size: 0.65rem !important; padding: 0.2rem 0.55rem !important; }

    /* Bigger touch targets */
    [data-testid="stButton"] > button,
    [data-testid="stDownloadButton"] > button { min-height: 48px !important; font-size: 0.9rem !important; }

    /* Data table scrollable */
    [data-testid="stDataFrame"] { overflow-x: auto !important; }
}

/* ═══════════════════════════════════════════
   SMALL PHONES — ≤ 480px
═══════════════════════════════════════════ */
@media (max-width: 480px) {
    .main-content { padding: 0.75rem 0.65rem !important; }
    .page-title   { font-size: 1.35rem !important; }
    .stat-num     { font-size: 1.45rem !important; }
    .stats-row    { gap: 0.45rem !important; }
    .stat-card    { padding: 0.7rem 0.75rem !important; }
    .stat-ico     { display: none; }
    .r-plate      { font-size: 1rem !important; letter-spacing: 0.04em !important; }
    .feat-icon    { font-size: 1.3rem !important; margin-bottom: 0.5rem !important; }
    .feat-title   { font-size: 0.82rem !important; }
    .feat-desc    { font-size: 0.72rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🖼️ Image Detection"

# ── Force sidebar open on every load ─────────────────────────────────────────
st.markdown("""
<script>
(function() {
    function openSidebar() {
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            var btn = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
            if (btn) btn.click();
        }
    }
    openSidebar();
    setTimeout(openSidebar, 100);
    setTimeout(openSidebar, 400);
})();
</script>
""", unsafe_allow_html=True)

# ── NAV ITEMS ─────────────────────────────────────────────────────────────────
NAV = [
    ("🖼️ Image Detection",  "Detect vehicles from images"),
    ("🎥 Video Detection",  "Process video files"),
    ("📷 Live Camera",      "Real-time webcam stream"),
    ("❓ About & Help",     "Documentation & info"),
]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-logo-row">
            <div class="sb-icon">🛡️</div>
            <div>
                <div class="sb-name">VMS</div>
                <div class="sb-sub">Vehicle Monitoring System</div>
            </div>
        </div>
        <div class="sb-live"><div class="sb-live-dot"></div>AI Engine Active</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Menu</div>', unsafe_allow_html=True)

    NAV_ICONS = {
        "🖼️ Image Detection": ("🖼️", "Image Detection"),
        "🎥 Video Detection":  ("🎥", "Video Detection"),
        "📷 Live Camera":      ("📷", "Live Camera"),
        "❓ About & Help":     ("❓", "About & Help"),
    }
    for label, _ in NAV:
        icon, name = NAV_ICONS[label]
        if st.session_state.page == label:
            st.markdown(f'<div class="sb-active">{icon}&nbsp;&nbsp;{name}</div>', unsafe_allow_html=True)
        else:
            if st.button(f"{icon}  {name}", key=f"nav_{label}", use_container_width=True):
                st.session_state.page = label
                st.rerun()

    st.markdown('<div class="sb-divider" style="margin-top:0.8rem;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-sysinfo">
        <div class="sb-sysinfo-title">System Status</div>
        <div class="sb-sysinfo-row">
            <span class="sb-sysinfo-key">YOLO Models</span>
            <span class="sb-sysinfo-val">● Loaded</span>
        </div>
        <div class="sb-sysinfo-row">
            <span class="sb-sysinfo-key">OCR Engine</span>
            <span class="sb-sysinfo-val">● Ready</span>
        </div>
        <div class="sb-sysinfo-row">
            <span class="sb-sysinfo-key">Pipeline</span>
            <span class="sb-sysinfo-val">● Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-footer">
        <span class="sb-footer-txt">VMS · Streamlit + YOLOv8</span>
        <span class="sb-version">v2.0</span>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ═══════════════════════════════════════════
# ABOUT & HELP PAGE
# ═══════════════════════════════════════════
if page == "❓ About & Help":
    st.markdown("""
    <div class="main-content">
    <div class="page-top">
        <div class="page-title-wrap">
            <div class="page-title">About <b>&amp; Help</b></div>
            <div class="page-sub">System overview, features and usage guide</div>
        </div>
        <div class="page-badge">📖 Documentation</div>
    </div>

    <div class="about-hero">
        <div class="about-hero-glow"></div>
        <div class="about-title">Vehicle Monitoring System</div>
        <div class="about-desc">
            An AI-powered surveillance platform for real-time vehicle detection,
            Indian license plate recognition via multi-stage OCR, and seatbelt compliance
            monitoring — across still images, uploaded video files, and live webcam feeds.
        </div>
    </div>

    <div class="s-label">Core Features</div>
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">🚗</div>
            <div class="feat-title">Vehicle Detection</div>
            <div class="feat-desc">Detects cars, trucks, buses and motorcycles using YOLOv8 with intelligent reclassification logic for edge cases.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🔢</div>
            <div class="feat-title">Plate Recognition</div>
            <div class="feat-desc">3-stage OCR pipeline: YOLO detector → contour localization → EasyOCR fallback. Optimized for Indian plates.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🪑</div>
            <div class="feat-title">Seatbelt Detection</div>
            <div class="feat-desc">Per-vehicle seatbelt compliance detection using a dedicated YOLOv8 classification model.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🎥</div>
            <div class="feat-title">Video Processing</div>
            <div class="feat-desc">Processes every 30th frame of uploaded video files for efficient real-time-like analysis.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📹</div>
            <div class="feat-title">Live Camera Feed</div>
            <div class="feat-desc">WebRTC-powered live webcam stream with frame-by-frame AI inference directly in the browser.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📊</div>
            <div class="feat-title">CSV Export</div>
            <div class="feat-desc">Download all detection results as a structured CSV report for offline review and record keeping.</div>
        </div>
    </div>

    <div class="s-label" style="margin-top:1.8rem;">How to Use</div>
    <div class="help-3col">
        <div class="help-card">
            <div class="help-card-title">🖼️ Image Detection</div>
            <div class="help-step"><div class="help-n">1</div><div class="help-t">Select <b>Image Detection</b> from the sidebar.</div></div>
            <div class="help-step"><div class="help-n">2</div><div class="help-t">Upload a JPG or PNG image containing vehicles.</div></div>
            <div class="help-step"><div class="help-n">3</div><div class="help-t">AI pipeline runs automatically — stats and results appear instantly.</div></div>
            <div class="help-step"><div class="help-n">4</div><div class="help-t">Download the CSV report for record keeping.</div></div>
        </div>
        <div class="help-card">
            <div class="help-card-title">🎥 Video Detection</div>
            <div class="help-step"><div class="help-n">1</div><div class="help-t">Select <b>Video Detection</b> from the sidebar.</div></div>
            <div class="help-step"><div class="help-n">2</div><div class="help-t">Upload an MP4, AVI, or MOV video file.</div></div>
            <div class="help-step"><div class="help-n">3</div><div class="help-t">Every 30th frame is processed and displayed live.</div></div>
            <div class="help-step"><div class="help-n">4</div><div class="help-t">Click <b>Stop</b> to halt processing at any time.</div></div>
        </div>
        <div class="help-card">
            <div class="help-card-title">📷 Live Camera</div>
            <div class="help-step"><div class="help-n">1</div><div class="help-t">Select <b>Live Camera</b> from the sidebar.</div></div>
            <div class="help-step"><div class="help-n">2</div><div class="help-t">Allow browser camera permission when prompted.</div></div>
            <div class="help-step"><div class="help-n">3</div><div class="help-t">Click <b>Start</b> in the WebRTC widget to begin streaming.</div></div>
            <div class="help-step"><div class="help-n">4</div><div class="help-t">Click <b>Stop</b> to end the live session.</div></div>
        </div>
    </div>

    <div class="s-label" style="margin-top:1.8rem;">Tech Stack</div>
    <div class="tech-bar">
        <span class="tech-tag">🧠 YOLOv8</span>
        <span class="tech-tag">👁️ EasyOCR</span>
        <span class="tech-tag">📝 Tesseract</span>
        <span class="tech-tag">🖼️ OpenCV</span>
        <span class="tech-tag">🌐 Streamlit</span>
        <span class="tech-tag">📡 WebRTC</span>
        <span class="tech-tag">🐍 Python 3.10+</span>
        <span class="tech-tag">🔢 NumPy</span>
        <span class="tech-tag">📊 Pandas</span>
    </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════
# VIDEO DETECTION PAGE
# ═══════════════════════════════════════════
if page == "🎥 Video Detection":
    st.markdown("""
    <div class="main-content">
    <div class="page-top">
        <div class="page-title-wrap">
            <div class="page-title">Video <b>Detection</b></div>
            <div class="page-sub">Upload and process video files frame by frame</div>
        </div>
        <div class="page-badge">🎬 Video Mode</div>
    </div>
    <div class="s-label">Upload Video File</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 0.9rem;">', unsafe_allow_html=True)
        uploaded_video = st.file_uploader("MP4 · AVI · MOV", type=["mp4","avi","mov"], label_visibility="collapsed")
        col_btn, _ = st.columns([1, 6])
        with col_btn:
            stop_video = st.button("⏹ Stop Processing", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_video:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(uploaded_video.read())
        cap = cv2.VideoCapture(tmp.name)
        placeholder = st.empty()
        n = 0
        with st.spinner("Processing video stream..."):
            while cap.isOpened():
                if stop_video: break
                ret, frame = cap.read()
                if not ret: break
                n += 1
                if n % 30 != 0: continue
                frame = cv2.resize(frame, (640, 480))
                r = process_image(frame, video_mode=True)
                placeholder.image(r["image"], channels="BGR", use_container_width=True)
        cap.release()
        st.success("✅ Video processing complete")
    st.stop()

# ═══════════════════════════════════════════
# LIVE CAMERA PAGE
# ═══════════════════════════════════════════
elif page == "📷 Live Camera":
    webrtc_streamer, VideoProcessorBase, av = load_webrtc_libs()

    st.markdown("""
    <div class="main-content">
    <div class="page-top">
        <div class="page-title-wrap">
            <div class="page-title">Live <b>Camera</b></div>
            <div class="page-sub">Real-time AI detection from your webcam</div>
        </div>
        <div class="page-badge">📹 Live Mode</div>
    </div>
    <div class="s-label">Camera Feed</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 0.9rem;">', unsafe_allow_html=True)
        st.info("🔒 Camera access required — allow permission when prompted, then click **Start**.")

        class VideoProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                r = process_image(img, video_mode=True)
                return av.VideoFrame.from_ndarray(r["image"], format="bgr24")

        webrtc_streamer(
            key="vms-cam",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════
# IMAGE DETECTION PAGE
# ═══════════════════════════════════════════
st.markdown("""
<div class="main-content">
<div class="page-top">
    <div class="page-title-wrap">
        <div class="page-title">Image <b>Detection</b></div>
        <div class="page-sub">Upload an image to run the full AI detection pipeline</div>
    </div>
    <div class="page-badge">📸 Image Mode</div>
</div>
<div class="s-label">Upload Image</div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div style="padding:0 0.9rem;">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or click — JPG · JPEG · PNG",
        type=["jpg","jpeg","png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("🔍 Running AI pipeline..."):
        result = process_image(image)

    data      = result["data"]
    total     = len(data)
    plates    = sum(1 for d in data if d.get("plate") not in ("Not Found","Skipped","",None))
    belt_on   = sum(1 for d in data if d.get("seatbelt") == "Detected")
    belt_off  = total - belt_on

    st.markdown(f"""
    <div class="main-content" style="padding-top:0.5rem;">
    <div class="s-label">Detection Summary</div>
    <div class="stats-row">
        <div class="stat-card stat-card-b">
            <div class="stat-ico">🚗</div>
            <div class="stat-num stat-num-b">{total:02d}</div>
            <div class="stat-lbl">Total Vehicles</div>
        </div>
        <div class="stat-card stat-card-c">
            <div class="stat-ico">🔢</div>
            <div class="stat-num stat-num-c">{plates:02d}</div>
            <div class="stat-lbl">Plates Recognized</div>
        </div>
        <div class="stat-card stat-card-g">
            <div class="stat-ico">✅</div>
            <div class="stat-num stat-num-g">{belt_on:02d}</div>
            <div class="stat-lbl">Seatbelts On</div>
        </div>
        <div class="stat-card stat-card-r">
            <div class="stat-ico">⚠️</div>
            <div class="stat-num stat-num-r">{belt_off:02d}</div>
            <div class="stat-lbl">Seatbelts Off</div>
        </div>
    </div>
    <div class="s-label">Visual Comparison</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="img-col-left"><div class="img-panel"><div class="img-hdr"><div class="img-hdr-dot" style="background:#475569;"></div>Original Image</div>', unsafe_allow_html=True)
        st.image(image, channels="BGR", use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="img-col-right"><div class="img-panel"><div class="img-hdr"><div class="img-hdr-dot" style="background:#3b82f6;"></div>AI Detection Output</div>', unsafe_allow_html=True)
        st.image(result["image"], channels="BGR", use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="main-content" style="padding-top:1.2rem;"><div class="s-label">Vehicle Results</div>', unsafe_allow_html=True)

    if total == 0:
        st.markdown('<div class="empty-box"><span class="empty-icon">🔍</span>No vehicles detected in this image.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="results-wrap">', unsafe_allow_html=True)
        for i, v in enumerate(data):
            plate = v.get("plate", "Not Found")
            plate_html = (
                f'<div class="r-plate">{plate}</div>'
                if plate not in ("Not Found","Skipped","",None)
                else '<div class="r-plate-miss">⚠ Plate not detected</div>'
            )
            sb     = v.get("seatbelt","Not Detected")
            sb_cls = "r-belt-on"  if sb == "Detected" else "r-belt-off"
            sb_txt = "✓ Seatbelt On" if sb == "Detected" else "✗ No Seatbelt"
            vtype  = v.get("vehicle","Vehicle").title()
            st.markdown(f"""
            <div class="r-card">
                <div class="r-vidx">Vehicle #{i+1}</div>
                <div class="r-type-badge">⬡ {vtype}</div>
                {plate_html}
                <div class="r-belt {sb_cls}">{sb_txt}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="s-label" style="margin-top:1.5rem;">Data Export</div>', unsafe_allow_html=True)
        df = pd.DataFrame(data)
        st.data_editor(df, use_container_width=True, disabled=True, hide_index=True)
        st.download_button(
            label="📥 Download CSV Report",
            data=df.to_csv(index=False),
            file_name="vms_detection_report.csv",
            mime="text/csv",
        )

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="padding: 0 0.9rem 2rem;">
    <div class="empty-box">
        <span class="empty-icon">🛡️</span>
        Upload an image above to begin AI vehicle analysis.
    </div>
    </div>
    """, unsafe_allow_html=True)