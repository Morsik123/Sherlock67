# UAV Telemetry Analysis & 3D Flight Visualization System

BEST::HACKath0n 2026 — Challenge: Система аналізу телеметрії та 3D-візуалізації польотів БПЛА

---

## Overview

A web dashboard that parses Ardupilot DataFlash .BIN log files, computes flight metrics, renders an interactive 3D trajectory, and optionally generates an AI-powered flight assessment.

### Features
- Binary log parser — custom implementation for Ardupilot DataFlash v2 format (no external mavlink dependency)
- GPS telemetry: position (WGS-84), speed, vertical velocity, altitude
- IMU telemetry: 3-axis acceleration and gyroscope data
- ATT telemetry: Roll, Pitch, Yaw angles
- Haversine distance — algorithmically correct great-circle path length
- Trapezoidal IMU integration — velocity derived from accelerometer data
- WGS-84 → ENU conversion — local Cartesian frame for 3D visualization
- Interactive 3D trajectory — colored by speed, time, or altitude
- AI analysis — Claude API generates Ukrainian-language flight assessment
- Streamlit web UI — upload .BIN files directly in browser

---

## Tech Stack Choice

| Component | Technology | Reason |
|-----------|-----------|--------|
| Parser | Pure Python (struct) | No binary dependencies; full control over DataFlash format |
| DataFrames | pandas | Fast vectorized operations on time-series sensor data |
| Numerics | numpy | Efficient array math for integration and ENU conversion |
| 3D Viz | plotly | Interactive WebGL 3D with zero JS boilerplate; embeds in Streamlit |
| Web UI | streamlit | Fastest Python path from script to browser app; file upload built-in |
| AI | anthropic (Claude) | Best-in-class instruction-following for structured Ukrainian output |

---

## Installation & Launch

### Requirements
- Python 3.10+

### 1. Install dependencies

pip install streamlit pandas numpy plotly anthropic

### 2. Run the dashboard

cd uav-telemetry
streamlit run app.py

The app opens at http://localhost:8501

### 3. Upload a log file

Drag and drop your .BIN file into the sidebar uploader, or click Browse files.

### 4. AI Analysis (optional)

Set your Anthropic API key in the sidebar field, then click Analyze with AI.

Alternatively, set it as an environment variable:

export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py

---

## Project Structure

uav-telemetry/
├── app.py          # Streamlit dashboard (main entry point)
├── parser.py       # Ardupilot DataFlash binary parser
├── metrics.py      # Haversine, trapezoidal integration, flight metrics
├── visualizer.py   # Plotly 3D trajectory + time-series charts
├── ai_analyst.py   # Claude API flight analysis
└── README.md

---

## Algorithm Notes

### Haversine Formula
Great-circle distance between two WGS-84 points:
a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
d = 2·R·arcsin(√a)    where R = 6,371,000 m

### Trapezoidal Integration (IMU → velocity)
v[k+1] = v[k] + 0.5 · (a[k] + a[k+1]) · Δt
> Note on drift: Double-integration of raw accelerometer data accumulates error proportional to σ_acc · t²/2. Production flight controllers fuse GPS + IMU via an Extended Kalman Filter (EKF) to suppress this drift. The IMU-derived speed shown here is for algorithmic demonstration only.

### WGS-84 → ENU Coordinate Conversion
East  = (lon - lon₀) · cos(lat₀) · R · π/180
North = (lat - lat₀) · R · π/180
Up    = alt - alt₀
Valid for trajectories shorter than ~50 km (flat-Earth approximation).

### Why Quaternions Instead of Euler Angles?
Ardupilot stores orientation internally as unit quaternions (4 components: w, x, y, z) rather than Roll/Pitch/Yaw Euler angles. The reason is gimbal lock: when pitch reaches ±90°, roll and yaw axes align and one degree of freedom is lost, making smooth attitude estimation impossible. Quaternions have no such singularity, and rotation composition is a simple multiplication rather than a sequence of trigonometric evaluations at every control loop tick (~400 Hz).

---

## Coordinate Reference
- GPS positions: WGS-84 geographic coordinates (degrees)
- Altitude: metres above Mean Sea Level (MSL), scaled from cm (DataFlash e type)
- ENU frame: metres East/North/Up from first GPS fix
- Speeds: m/s
- Accelerations: m/s²
- Angles: degrees (Roll, Pitch, Yaw)
- Time: microseconds in log (TimeUS), seconds in DataFrames (time_s)

---

