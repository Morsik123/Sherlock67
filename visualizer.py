"""
visualizer.py — 3D interactive flight trajectory visualization using Plotly.

Coordinate system:
    Input:  WGS-84 (Lat/Lon degrees, Altitude metres MSL)
    Output: ENU local Cartesian (East/North/Up metres from home point)

Coloring modes:
    'speed'  — trajectory colored by horizontal GPS speed (m/s)
    'time'   — trajectory colored by elapsed time (s)
    'altitude' — trajectory colored by height above home (m)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from metrics import wgs84_to_enu


def build_3d_trajectory(
    gps_df: pd.DataFrame,
    color_by: str = 'speed',
    title: str = 'UAV Flight Trajectory',
) -> go.Figure:
    """
    Build an interactive 3D Plotly figure of the flight path.

    Args:
        gps_df:   GPS DataFrame from parser.to_dataframes().
        color_by: 'speed' | 'time' | 'altitude'
        title:    Figure title string.

    Returns:
        plotly.graph_objects.Figure
    """
    lats = gps_df['Lat'].values
    lons = gps_df['Lng'].values
    alts = gps_df['Alt'].values

    east, north, up = wgs84_to_enu(lats, lons, alts)

    # Choose color array
    if color_by == 'speed':
        color_vals = gps_df['Spd'].values
        color_label = 'Speed (m/s)'
        colorscale = 'Plasma'
    elif color_by == 'altitude':
        color_vals = up
        color_label = 'Height AGL (m)'
        colorscale = 'Viridis'
    else:  # time
        color_vals = gps_df['time_s'].values
        color_label = 'Time (s)'
        colorscale = 'Turbo'

    # Hover text
    hover = [
        f"t={t:.1f}s<br>E={e:.1f}m N={n:.1f}m Up={u:.1f}m<br>"
        f"Spd={s:.1f}m/s  Alt={a:.0f}m MSL"
        for t, e, n, u, s, a in zip(
            gps_df['time_s'].values, east, north, up,
            gps_df['Spd'].values, alts
        )
    ]

    # Main trajectory line
    line_trace = go.Scatter3d(
        x=east, y=north, z=up,
        mode='lines+markers',
        line=dict(
            color=color_vals,
            colorscale=colorscale,
            width=6,
            colorbar=dict(
                title=color_label,
                thickness=15,
                len=0.6,
            ),
        ),
        marker=dict(
            size=3,
            color=color_vals,
            colorscale=colorscale,
            opacity=0.8,
        ),
        text=hover,
        hoverinfo='text',
        name='Trajectory',
    )

    # Start marker
    start_trace = go.Scatter3d(
        x=[east[0]], y=[north[0]], z=[up[0]],
        mode='markers+text',
        marker=dict(size=10, color='lime', symbol='circle'),
        text=['START'],
        textposition='top center',
        name='Start',
        hoverinfo='text',
        hovertext=f'START  t=0s  Alt={alts[0]:.0f}m',
    )

    # End marker
    end_trace = go.Scatter3d(
        x=[east[-1]], y=[north[-1]], z=[up[-1]],
        mode='markers+text',
        marker=dict(size=10, color='red', symbol='x'),
        text=['END'],
        textposition='top center',
        name='End',
        hoverinfo='text',
        hovertext=f'END  t={gps_df["time_s"].iloc[-1]:.1f}s  Alt={alts[-1]:.0f}m',
    )

    fig = go.Figure(data=[line_trace, start_trace, end_trace])

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        scene=dict(
            xaxis=dict(title='East (m)', backgroundcolor='rgb(15,15,25)',
                       gridcolor='rgba(100,100,100,0.3)', showbackground=True),
            yaxis=dict(title='North (m)', backgroundcolor='rgb(15,15,25)',
                       gridcolor='rgba(100,100,100,0.3)', showbackground=True),
            zaxis=dict(title='Up / AGL (m)', backgroundcolor='rgb(10,10,20)',
                       gridcolor='rgba(100,100,100,0.3)', showbackground=True),
            bgcolor='rgb(10,10,20)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        paper_bgcolor='rgb(10,10,20)',
        plot_bgcolor='rgb(10,10,20)',
        font=dict(color='white'),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(30,30,50,0.8)',
            bordercolor='rgba(150,150,200,0.3)',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=550,
    )

    return fig


def build_speed_altitude_chart(gps_df: pd.DataFrame) -> go.Figure:
    """Dual-axis line chart: horizontal speed and altitude over time."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gps_df['time_s'], y=gps_df['Spd'],
        name='Horiz. Speed (m/s)',
        line=dict(color='#00d4ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.08)',
    ))

    alt_agl = gps_df['Alt'] - gps_df['Alt'].iloc[0]
    fig.add_trace(go.Scatter(
        x=gps_df['time_s'], y=alt_agl,
        name='Altitude AGL (m)',
        yaxis='y2',
        line=dict(color='#ff6b35', width=2, dash='dot'),
    ))

    fig.update_layout(
        xaxis=dict(title='Time (s)', gridcolor='rgba(100,100,100,0.2)'),
        yaxis=dict(title='Speed (m/s)', gridcolor='rgba(100,100,100,0.2)'),
        yaxis2=dict(title='Alt AGL (m)', overlaying='y', side='right',
                    gridcolor='rgba(100,100,100,0.1)'),
        legend=dict(x=0.01, y=0.99),
        paper_bgcolor='rgb(15,15,25)',
        plot_bgcolor='rgb(15,15,25)',
        font=dict(color='white'),
        height=300,
        margin=dict(l=50, r=60, t=20, b=40),
    )
    return fig


def build_imu_chart(imu_df: pd.DataFrame) -> go.Figure:
    """Acceleration components over time."""
    fig = go.Figure()
    for axis, color in [('AccX', '#ff4757'), ('AccY', '#2ed573'), ('AccZ', '#1e90ff')]:
        fig.add_trace(go.Scatter(
            x=imu_df['time_s'], y=imu_df[axis],
            name=axis, line=dict(color=color, width=1.5),
            opacity=0.85,
        ))

    fig.update_layout(
        xaxis=dict(title='Time (s)', gridcolor='rgba(100,100,100,0.2)'),
        yaxis=dict(title='Acceleration (m/s²)', gridcolor='rgba(100,100,100,0.2)'),
        paper_bgcolor='rgb(15,15,25)',
        plot_bgcolor='rgb(15,15,25)',
        font=dict(color='white'),
        height=280,
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def build_attitude_chart(att_df: pd.DataFrame) -> go.Figure:
    """Roll, Pitch, Yaw over time."""
    fig = go.Figure()
    colors = {'Roll': '#ffa502', 'Pitch': '#ff6b81', 'Yaw': '#a29bfe'}
    for col, clr in colors.items():
        if col in att_df.columns:
            fig.add_trace(go.Scatter(
                x=att_df['time_s'], y=att_df[col],
                name=col, line=dict(color=clr, width=1.5),
            ))

    fig.update_layout(
        xaxis=dict(title='Time (s)', gridcolor='rgba(100,100,100,0.2)'),
        yaxis=dict(title='Angle (deg)', gridcolor='rgba(100,100,100,0.2)'),
        paper_bgcolor='rgb(15,15,25)',
        plot_bgcolor='rgb(15,15,25)',
        font=dict(color='white'),
        height=280,
        margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(x=0.01, y=0.99),
    )
    return fig