import os
import numpy as np
import pandas as pd
from children_exp.children_preprocessing import *

# Standard Imports 
import tensorflow as tf
# Imports from the model folder 
from model.attentive_pooling import AttentionWithContext
from model.sensor_attention import SensorAttention
from model.self_attention.encoder import EncoderLayer
from model.self_attention.positional_encoding import PositionalEncoding

# Model path
MODEL_PATH = "saved_model/pamap2"

csvLists = [
    "childrens_data/P009_QM-SS1_0E3E9_p1.csv", 
    "childrens_data/P009_QM-SS1_0E3E9_p2.csv",
    "childrens_data/P009_QM-SS1_11CCD_p1.csv",
    "childrens_data/P009_QM-SS1_11CCD_p2.csv",
    "childrens_data/P009_QM-SS1_14A51_p1.csv",
    "childrens_data/P009_QM-SS1_14A51_p2.csv",
    "childrens_data/P009_QM-SS1_16E17_p1.csv",
    "childrens_data/P009_QM-SS1_16E17_p2.csv",
    "childrens_data/P009_QM-SS1_1503C_p1.csv",
    "childrens_data/P009_QM-SS1_1503C_p2.csv",
    "childrens_data/P009_QM-SS1_12144_p1.csv",
    "childrens_data/P009_QM-SS1_12144_p2.csv"
]

custom_objects = {
    "AttentionWithContext": AttentionWithContext,
    "SensorAttention": SensorAttention,
    "EncoderLayer": EncoderLayer,
    "PositionalEncoding": PositionalEncoding
}

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects
)

# --- Helper Functions (unchanged) ---
def create_windows(data, window_size=30, step_size=15):
    return np.array([
        data[i:i+window_size]
        for i in range(0, len(data) - window_size, step_size)
    ])

def fix_timesteps(X, target_len=33):
    current_len = X.shape[1]

    if current_len > target_len:
        return X[:, :target_len, :]
    
    pad = target_len - current_len
    padding = np.zeros((X.shape[0], pad, X.shape[2]))
    return np.concatenate([X, padding], axis=1)

def fix_features(X, target_features=18):
    current = X.shape[2]

    if current >= target_features:
        return X[:, :, :target_features]

    pad = np.zeros((X.shape[0], X.shape[1], target_features - current))
    return np.concatenate([X, pad], axis=2)

# --- Main Loop ---
for csv_path in csvLists:
    print(f"Processing: {csv_path}")

    # Extract filename without folder + extension
    base_name = os.path.basename(csv_path).replace(".csv", "")

    # Load + preprocess
    df = load_data(csv_path)
    df = prepare_rotation_df(df)
    df = resample_to_30hz(df)

    # Features
    features = df.drop(columns=["Timestamp"]).values

    # Windowing + fixes
    X = create_windows(features)
    X = fix_timesteps(X)
    X = fix_features(X)

    # Dummy labels
    Y = np.zeros(len(X))

    # Save X and Y
    np.save(f"{base_name}_X.npy", X)
    np.save(f"{base_name}_Y.npy", Y)

    print("X shape:", X.shape)

    # Predict embeddings
    embeddings = model.predict(X, batch_size=64)

    # Save embeddings
    np.save(f"{base_name}_predictions.npy", embeddings)

print("All files processed.")