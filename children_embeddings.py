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
# Imports from the preprocess folder 
from preprocess.pamap2.data_loader import get_pamap2_data

# We will first start by loading the pretrained PAMAP2 model 
MODEL_PATH = "saved_model/pamap2"

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


df = load_data("childrens_data/march5.csv")
df = prepare_rotation_df(df)
df = resample_to_30hz(df)

# Features
features = df.drop(columns=["Timestamp"]).values

# Windowing
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


X = create_windows(features)
X = fix_timesteps(X)
X = fix_features(X)
# Dummy labels (for now)
Y = np.zeros(len(X))

# Save
np.save("Children_X.npy", X)
np.save("Children_Y.npy", Y)

print("X shape:", X.shape)

embeddings = model.predict(X, batch_size=64)
np.save("children_predictions.npy", embeddings)


