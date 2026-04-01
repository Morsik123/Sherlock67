"""
app.py — UAV Telemetry Analysis Dashboard (Streamlit).

Launch:
    streamlit run app.py

Features:
    - Upload Ardupilot .BIN log files via browser
    - Automatic parsing of GPS, IMU, ATT messages
    - Flight metrics: distance (haversine), speed, altitude, acceleration
    - Interactive 3D trajectory visualization (ENU coordinates)
    - Speed/altitude, IMU, attitude time-series charts
    - AI-powered flight analysis (Claude API)
"""

import streamlit as st
import pandas as pd
import tempfile
import os

from parser import to_dataframes, get_sampling_info
from metrics import compute_metrics
from visualizer import (
    build_3d_trajectory,
    build_speed_altitude_chart,
    build_imu_chart,
    build_attitude_chart,
)
from ai_analyst import analyze_flight

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAV Telemetry Analyzer",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sci-Fi Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@500;700&display=swap');

    /* ================= GLOBAL & BACKGROUND ================= */
    [data-testid="stAppViewContainer"] { 
        background: radial-gradient(circle at 50% 0%, #111122 0%, #05050a 100%);
        font-family: 'Rajdhani', sans-serif;
    }

    /* CRT Scanline Overlay Effect */
    [data-testid="stAppViewContainer"]::after {
        content: " ";
        display: block;
        position: fixed;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        z-index: 9999;
        background-size: 100% 2px, 3px 100%;
        pointer-events: none;
    }

    /* Headers (Global) */
    h1, h2, h3 { 
        font-family: 'Orbitron', sans-serif !important;
        color: #e2e8f0 !important; 
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(226, 232, 240, 0.3);
        transition: all 0.3s ease; /* Для плавного ефекту наведення в сайдбарі */
    }

    /* ================= SIDEBAR HOLOGRAPHIC FEEL & ELEMENTS ================= */
    [data-testid="stSidebar"] { 
        background: rgba(10, 15, 25, 0.6) !important; 
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(100, 255, 218, 0.2);
        box-shadow: 5px 0 25px rgba(0, 0, 0, 0.5);
    }

    /* Спеціальний шрифт для сайдбару */
    [data-testid="stSidebar"] div.stMarkdown {
        color: #a8b2d1; /* Трохи тьмяніший за замовчуванням */
        transition: all 0.3s ease;
    }

    /* ================= NEW: SUPER HOVER EFFECTS FOR SIDEBAR ELEMENTS ================= */
    /* Додаємо підсвічування для заголовків у сайдбарі (🛸 UAV Telemetry, etc.) */
    [data-testid="stSidebar"] h1:hover,
    [data-testid="stSidebar"] h2:hover,
    [data-testid="stSidebar"] h3:hover {
        color: #64ffda !important; /* Яскраво-бірюзовий */
        text-shadow: 0 0 15px rgba(100, 255, 218, 0.8) !important; /* Посилене неонове світіння */
        transform: none !important; /* ЗАБОРОНА РУХУ: Не пригати */
    }

    /* Додаємо підсвічування для звичайного тексту та іконок у сайдбарі (Ardupilot log analyzer, etc.) */
    [data-testid="stSidebar"] p:hover,
    [data-testid="stSidebar"] div.stMarkdown:hover {
        color: #64ffda !important;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
        transform: none !important; /* ЗАБОРОНА РУХУ: Не пригати */
    }

    /* Додаємо підсвічування для опцій selectbox */
    div.stSelectbox [aria-selected="true"] {
        color: #64ffda !important;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.5);
    }
    div.stSelectbox [data-baseweb="list"] [aria-selected="true"] {
        background-color: rgba(100,255,218,0.1);
    }

    /* ================= AGGRESSIVE STOP FOR JUMPING (from previous step) ================= */
    [data-testid="stColumn"] > div,
    [data-testid="stBlock"] > div,
    .element-container,
    .stMarkdown {
        transform: none !important;
        transition: transform 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease !important;
        box-shadow: none !important;
    }

    /* ================= METRIC CARDS - STABLE VERSION (from previous step) ================= */
    .metric-card {
        background: linear-gradient(135deg, rgba(20, 25, 40, 0.7) 0%, rgba(10, 15, 25, 0.9) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(100,255,218,0.2);
        border-radius: 8px;
        padding: 20px;
        margin: 8px 0;
        position: relative;
        overflow: hidden;
        transition: border-color 0.4s ease, box-shadow 0.4s ease !important;
        transform: none !important; /* ЗАБОРОНА РУХУ: Усі рухи заборонені */
    }

    /* Залиште лише неоновий блік і тінь, але БЕЗ руху. */
    .metric-card:hover {
        border-color: rgba(100,255,218,0.8);
        box-shadow: 0 0 20px rgba(100, 255, 218, 0.6) !important;
        z-index: 10;
    }

    /* Sweep effect (блік) на картках метрик */
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(100, 255, 218, 0.2), transparent);
        transform: skewX(-20deg);
        transition: 0.5s;
        pointer-events: none;
    }
    .metric-card:hover::before { left: 200%; }

    /* Label and Value styling */
    .metric-label { 
        color: #64ffda; 
        font-family: 'Orbitron', sans-serif;
        font-size: 10px; 
        letter-spacing: 2px; 
        text-transform: uppercase; 
        margin-bottom: 8px;
        opacity: 0.8;
    }
    .metric-value { 
        color: #ffffff; 
        font-size: 32px; 
        font-weight: 700; 
        text-shadow: 0 0 15px rgba(100,255,218,0.5);
    }
    .metric-unit { 
        color: #a8b2d1; 
        font-size: 14px; 
        margin-left: 6px; 
    }

    /* ================= UI ELEMENTS ================= */
    /* Section Titles */
    .section-title {
        color: #64ffda;
        font-family: 'Orbitron', sans-serif;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 4px;
        text-transform: uppercase;
        padding: 16px 0 8px 0;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(to right, #64ffda, transparent) 1;
        margin-bottom: 20px;
        text-shadow: 0 0 12px rgba(100,255,218,0.4);
    }

    /* Animated File Uploader */
    [data-testid="stFileUploader"] { 
        background: rgba(100, 255, 218, 0.02);
        border: 2px dashed rgba(100, 255, 218, 0.4); 
        border-radius: 10px; 
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #64ffda;
        background: rgba(100, 255, 218, 0.08);
        box-shadow: 0 0 20px rgba(100, 255, 218, 0.2) inset;
    }

    /* Super AI Button */
    .stButton > button {
        background: linear-gradient(45deg, #00f2fe 0%, #4facfe 50%, #00f2fe 100%);
        background-size: 200% auto;
        color: #05050a;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: 0.5s;
    }
    .stButton > button:hover {
        background-position: right center; /* Gradient animation */
        box-shadow: 0 0 20px rgba(79, 172, 254, 0.7);
        color: #ffffff;
    }

    /* Tabs - HUD style */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; border-bottom: 1px solid rgba(100,255,218,0.2); }
    .stTabs [data-baseweb="tab"] {
        color: #a8b2d1;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #64ffda !important;
        background: linear-gradient(0deg, rgba(100,255,218,0.1) 0%, transparent 100%);
        border-bottom: 3px solid #64ffda !important;
        text-shadow: 0 0 10px rgba(100,255,218,0.5);
    }

    /* Scrollbar Hologram */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(5, 5, 10, 0.8); }
    ::-webkit-scrollbar-thumb { background: rgba(100, 255, 218, 0.4); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #64ffda; box-shadow: 0 0 10px #64ffda; }

    /* ================= FULLSCREEN LAYOUT FIX (from previous step) ================= */
    .block-container {
        padding-top: 2rem !important;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

def metric_card(label: str, value, unit: str = ''):
    """Render a styled metric card."""
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🛸 UAV Telemetry")
    st.markdown("**Ardupilot .BIN Log Analyzer**")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload .BIN log file(s)",
        type=["BIN", "bin", "log"],
        accept_multiple_files=True,
        help="Ardupilot DataFlash binary logs (.BIN)",
    )

    st.markdown("---")
    st.markdown('<div class="section-title">Visualization</div>', unsafe_allow_html=True)
    color_mode = st.selectbox(
        "3D Color Mode",
        options=['speed', 'time', 'altitude'],
        format_func=lambda x: {'speed': '🎨 Speed', 'time': '⏱ Time', 'altitude': '📈 Altitude'}[x],
    )


    st.markdown("---")
    st.caption("Challenge: BEST::HACKath0n 2026\nSystem: Ardupilot DataFlash v2")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# UAV Telemetry Analysis Dashboard")

if not uploaded_files:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#8892b0;">
        <div style="font-size:64px; margin-bottom:16px;">🛸</div>
        <div style="font-size:20px; color:#ccd6f6; margin-bottom:8px;">
            Upload an Ardupilot .BIN log file to begin
        </div>
        <div style="font-size:14px;">
            Drag & drop a log file in the sidebar, or click "Browse files"
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── File selector (if multiple uploaded) ─────────────────────────────────────
if len(uploaded_files) > 1:
    file_names = [f.name for f in uploaded_files]
    selected_name = st.selectbox("Select log file to analyze:", file_names)
    ufile = next(f for f in uploaded_files if f.name == selected_name)
else:
    ufile = uploaded_files[0]

# ── Parse ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Parsing binary log...")
def load_log(file_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.BIN') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        dfs = to_dataframes(tmp_path)
        info = get_sampling_info(dfs)
        metrics = compute_metrics(dfs)
    finally:
        os.unlink(tmp_path)
    return dfs, info, metrics

file_bytes = ufile.read()
dfs, sampling_info, metrics = load_log(file_bytes, ufile.name)

if 'GPS' not in dfs or len(dfs['GPS']) < 2:
    st.error("⚠️ Could not extract valid GPS data from this file.")
    st.stop()

gps = dfs['GPS']
imu = dfs.get('IMU')
att = dfs.get('ATT')

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"### 📁 `{ufile.name}`")

col_info = st.columns(4)
with col_info[0]:
    st.caption(f"📡 GPS: **{sampling_info.get('GPS', {}).get('freq_hz', '?')} Hz** "
               f"({sampling_info.get('GPS', {}).get('records', 0)} fixes)")
with col_info[1]:
    st.caption(f"📐 IMU: **{sampling_info.get('IMU', {}).get('freq_hz', '?')} Hz** "
               f"({sampling_info.get('IMU', {}).get('records', 0)} samples)")
with col_info[2]:
    st.caption(f"🔄 ATT: **{sampling_info.get('ATT', {}).get('freq_hz', '?')} Hz**")
with col_info[3]:
    st.caption(f"📍 Home: **{gps['Lat'].iloc[0]:.5f}°, {gps['Lng'].iloc[0]:.5f}°**")

st.markdown("---")

# ── Metrics cards ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Flight Summary Metrics</div>', unsafe_allow_html=True)

cols = st.columns(6)
card_data = [
    ("Duration",            f"{metrics['duration_s']:.1f}",        "s"),
    ("Distance (Haversine)",f"{metrics['total_distance_m']:.0f}",   "m"),
    ("Max Horiz Speed",     f"{metrics['max_horiz_speed_ms']:.2f}", "m/s"),
    ("Max Vert Speed",      f"{metrics['max_vert_speed_ms']:.2f}",  "m/s"),
    ("Max Acceleration",    f"{metrics['max_accel_ms2']:.2f}",      "m/s²"),
    ("Max Alt Gain",        f"{metrics['max_altitude_gain_m']:.1f}","m AGL"),
]
for col, (label, val, unit) in zip(cols, card_data):
    with col:
        metric_card(label, val, unit)

# ── 3D Trajectory ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">3D Flight Trajectory (ENU Frame)</div>',
            unsafe_allow_html=True)
fig3d = build_3d_trajectory(gps, color_by=color_mode, title=f"Trajectory — {ufile.name}")
st.plotly_chart(fig3d, use_container_width=True)

# ── Time-series charts ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Time Series</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Speed & Altitude", "📐 IMU Acceleration", "🔄 Attitude"])

with tab1:
    st.plotly_chart(build_speed_altitude_chart(gps), use_container_width=True)

with tab2:
    if imu is not None:
        st.plotly_chart(build_imu_chart(imu), use_container_width=True)
    else:
        st.info("No IMU data available.")

with tab3:
    if att is not None:
        st.plotly_chart(build_attitude_chart(att), use_container_width=True)
    else:
        st.info("No ATT data available.")

# ── Raw data preview ──────────────────────────────────────────────────────────
with st.expander("📊 Raw Data Preview"):
    tab_gps, tab_imu, tab_att = st.tabs(["GPS", "IMU", "ATT"])
    with tab_gps:
        st.dataframe(gps.head(50), use_container_width=True)
    with tab_imu:
        if imu is not None:
            st.dataframe(imu.head(50), use_container_width=True)
    with tab_att:
        if att is not None:
            st.dataframe(att.head(50), use_container_width=True)

