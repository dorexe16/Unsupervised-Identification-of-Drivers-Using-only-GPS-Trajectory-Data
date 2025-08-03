# -*- coding: utf-8 -*-
"""
Vectorised helpers to derive extra motion features from each trip DF.

Expected columns already present:
    longitude, latitude, orig_time (datetime64[ns]),
    speed [km/h], acceleration_est_1 [m/s²], angular_acc [deg/s]

Adds:
    d_speed   = first diff of speed  (km/h per second)
    jerk      = first diff of acceleration_est_1  (m/s³)
"""
import numpy as np, pandas as pd

def add_motion_features_inplace(df):
    """Add d_speed and jerk columns *in place* if missing."""
    if {"d_speed", "jerk"}.issubset(df.columns):
        return df                         # already patched

    df.sort_values("orig_time", inplace=True)
    dt = df["orig_time"].diff().dt.total_seconds().fillna(1.0)

    df["d_speed"] = df["speed"].diff() / dt
    accel = df["acceleration_est_1"]
    df["jerk"] = accel.diff() / dt

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[["d_speed", "jerk"]] = df[["d_speed", "jerk"]].fillna(0.0)
    return df