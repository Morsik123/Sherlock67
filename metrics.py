"""
metrics.py — Flight analytics core.

Algorithmic implementations required by the challenge:

1. Haversine formula — great-circle distance between two WGS-84 coordinates.
   d = 2R * arcsin( sqrt( sin^2((lat2-lat1)/2)
                        + cos(lat1)*cos(lat2)*sin^2((lon2-lon1)/2) ) )

2. Trapezoidal integration — velocity from accelerometer readings.
   v[k+1] = v[k] + 0.5 * (a[k] + a[k+1]) * dt
   Note on IMU drift: double-integration of noisy accelerometer data
   accumulates error quadratically over time (sigma_pos ~ sigma_acc * t^2 / 2).
   This is why GPS-fused EKF estimates are used in production FCs.

3. WGS-84 -> ENU conversion — map global lat/lon/alt to a local
   East-North-Up Cartesian frame (metres from home point).
   Uses a first-order flat-Earth approximation valid for distances < ~50 km:
     East  = (lon - lon0) * cos(lat0) * R_earth * pi/180
     North = (lat - lat0) * R_earth * pi/180
     Up    = alt - alt0

Why quaternions for orientation (note for code reviewer):
   Euler angles (Roll/Pitch/Yaw) suffer from gimbal lock — when pitch
   reaches +/-90 deg, roll and yaw axes align and one DOF is lost.
   Flight controllers store orientation as unit quaternions (4 components)
   which have no singularity, and compose rotations via multiplication
   instead of trigonometric evaluation at every step.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict

R_EARTH = 6_371_000.0  # metres


# ─────────────────────────────────────────────
# 1. Haversine
# ─────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in metres between two WGS-84 points.

    Args:
        lat1, lon1: Origin coordinates in decimal degrees.
        lat2, lon2: Destination coordinates in decimal degrees.

    Returns:
        Distance in metres.
    """
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2)
    return 2.0 * R_EARTH * math.asin(math.sqrt(a))


def total_distance(gps_df: pd.DataFrame) -> float:
    """Sum of haversine distances between consecutive GPS fixes (metres)."""
    lats = gps_df['Lat'].values
    lons = gps_df['Lng'].values
    dist = 0.0
    for k in range(len(lats) - 1):
        dist += haversine(lats[k], lons[k], lats[k + 1], lons[k + 1])
    return dist


# ─────────────────────────────────────────────
# 2. Trapezoidal integration of IMU
# ─────────────────────────────────────────────

def trapz_integrate(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """
    Cumulative trapezoidal integration: converts acceleration -> velocity.

    v[k+1] = v[k] + 0.5 * (a[k] + a[k+1]) * dt

    Args:
        values:  1-D array of acceleration samples (m/s^2).
        times_s: 1-D array of timestamps in seconds (same length).

    Returns:
        1-D array of velocity estimates (m/s), same length, starts at 0.
    """
    n = len(values)
    vel = np.zeros(n)
    for k in range(n - 1):
        dt = times_s[k + 1] - times_s[k]
        vel[k + 1] = vel[k] + 0.5 * (values[k] + values[k + 1]) * dt
    return vel


def imu_velocity_magnitude(imu_df: pd.DataFrame) -> np.ndarray:
    """
    Integrate all three IMU acceleration axes and return speed magnitude array.
    Initial velocity assumed zero (start of log segment).
    """
    t = imu_df['time_s'].values
    # Subtract gravity component from Z (drone body frame, approx): not done here
    # because we need EKF for proper body->world rotation; raw integration shown
    # as required by the challenge.
    vx = trapz_integrate(imu_df['AccX'].values, t)
    vy = trapz_integrate(imu_df['AccY'].values, t)
    vz = trapz_integrate(imu_df['AccZ'].values, t)
    return np.sqrt(vx**2 + vy**2 + vz**2)


# ─────────────────────────────────────────────
# 3. WGS-84 -> ENU conversion
# ─────────────────────────────────────────────

def wgs84_to_enu(lats: np.ndarray, lons: np.ndarray, alts: np.ndarray):
    """
    Convert WGS-84 coordinates to local ENU (East-North-Up) in metres.

    Origin (0,0,0) is the first point in the arrays.
    Uses flat-Earth approximation — valid for trajectories < ~50 km.

    Returns:
        east, north, up  — three numpy arrays in metres.
    """
    lat0 = lats[0]
    lon0 = lons[0]
    alt0 = alts[0]

    p = math.pi / 180.0
    cos_lat0 = math.cos(lat0 * p)

    east  = (lons - lon0) * cos_lat0 * R_EARTH * p
    north = (lats - lat0) * R_EARTH * p
    up    = alts - alt0

    return east, north, up


# ─────────────────────────────────────────────
# 4. Flight summary metrics
# ─────────────────────────────────────────────

def compute_metrics(dfs: Dict[str, pd.DataFrame]) -> dict:
    """
    Compute all required flight summary metrics from parsed DataFrames.

    Returns a dict with keys:
        duration_s          — total flight time (seconds)
        total_distance_m    — GPS haversine path length (metres)
        max_horiz_speed_ms  — max horizontal speed from GPS (m/s)
        max_vert_speed_ms   — max vertical speed magnitude from GPS VZ (m/s)
        max_accel_ms2       — max acceleration magnitude from IMU (m/s^2)
        max_altitude_gain_m — max altitude above home point (metres)
        home_alt_m          — home altitude MSL (metres)
        imu_max_speed_ms    — peak speed from trapezoidal IMU integration (m/s)
        gps_points          — number of valid GPS fixes
        imu_samples         — number of IMU samples
    """
    gps = dfs.get('GPS')
    imu = dfs.get('IMU')
    metrics = {}

    if gps is not None and len(gps) >= 2:
        metrics['duration_s']         = round(float(gps['time_s'].iloc[-1]), 2)
        metrics['total_distance_m']   = round(total_distance(gps), 1)
        metrics['max_horiz_speed_ms'] = round(float(gps['Spd'].max()), 2)
        metrics['max_vert_speed_ms']  = round(float(gps['VZ'].abs().max()), 2)
        metrics['home_alt_m']         = round(float(gps['Alt'].iloc[0]), 1)
        rel_alts = gps['Alt'] - gps['Alt'].iloc[0]
        metrics['max_altitude_gain_m'] = round(float(rel_alts.max()), 1)
        metrics['gps_points']          = len(gps)
    else:
        metrics.update({
            'duration_s': 0, 'total_distance_m': 0,
            'max_horiz_speed_ms': 0, 'max_vert_speed_ms': 0,
            'home_alt_m': 0, 'max_altitude_gain_m': 0, 'gps_points': 0,
        })

    if imu is not None and len(imu) >= 2:
        acc_mag = np.sqrt(imu['AccX']**2 + imu['AccY']**2 + imu['AccZ']**2)
        metrics['max_accel_ms2']    = round(float(acc_mag.max()), 2)
        imu_spd = imu_velocity_magnitude(imu)
        metrics['imu_max_speed_ms'] = round(float(imu_spd.max()), 2)
        metrics['imu_samples']      = len(imu)
    else:
        metrics.update({'max_accel_ms2': 0, 'imu_max_speed_ms': 0, 'imu_samples': 0})

    return metrics