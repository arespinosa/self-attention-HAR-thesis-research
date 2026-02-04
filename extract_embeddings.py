# Standard Imports 
import tensorflow as tf
import numpy as np
import pandas as pd
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

model.summary()

# From the model summary, the embedding layer we want is dense12. We know this since it's the 2nd to last layer.
# Now we'll create the embedding model 
EMBEDDING_LAYER = "dense_12"

embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer(EMBEDDING_LAYER).output
)

# We are calling the dataloader function specifically for pamap2 and going to be using text_x and test_y since those are the "test_inputs & test_targets"
(train_x, train_y), (val_x, val_y), (test_x, test_y), _ = get_pamap2_data(verbose=True)

embeddings = embedding_model.predict(test_x, batch_size=64)
labels = test_y  

np.save("pamap2_embeddings.npy", embeddings)
np.save("pamap2_labels.npy", labels)

print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels.shape)


embeddings = np.load("pamap2_embeddings.npy")

# labels was in the form of a 1d array since it was using OHE, so I converted to an acceptable shape
labels = np.load("pamap2_labels.npy")
labels_int = np.argmax(labels, axis=1) 

df = pd.DataFrame(embeddings)
df['label'] = labels_int
df.to_csv("pamap2_embeddings.csv", index=False)