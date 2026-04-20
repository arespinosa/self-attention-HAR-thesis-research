import pandas as pd
import numpy as np


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_rotation_df(df):
    rotation_df = df[[
        "Timestamp", "Rotation X", "Rotation Y", "Rotation Z", "Rotation W"
    ]].copy()

    rotation_df["Timestamp"] = pd.to_datetime(rotation_df["Timestamp"])
    rotation_df = rotation_df.sort_values(by="Timestamp").reset_index(drop=True)

    return rotation_df


def estimate_sampling_rate(df):
    time_diffs = df["Timestamp"].diff().dropna().dt.total_seconds()
    mean_dt = time_diffs.mean()
    return 1 / mean_dt


def resample_to_30hz(df):
    df = df.set_index("Timestamp")

    df_resampled = (
        df.resample("33ms")
        .mean()
        .interpolate(method="linear")
    )

    return df_resampled.reset_index()

# Windowing
def create_windows(data, window_size=30, step_size=15):
    return np.array([
        data[i:i+window_size]
        for i in range(0, len(data) - window_size, step_size)
    ])


