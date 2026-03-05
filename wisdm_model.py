import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils.data import get_data
from utils.result import generate_result
from utils.test import test_model
from utils.train import train_model
import yaml





# First step will be to print the shape of the data
X = np.load("wisdm_experiments/wisdm_data/wisdm_30hz_clean/X.npy")
Y = np.load("wisdm_experiments/wisdm_data/wisdm_30hz_clean/Y.npy")

print(X.shape)
print(Y.shape)


label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

print("Classes:", label_encoder.classes_)
print("Encoded shape:", Y_encoded.shape)

# Now one-hot encode
num_classes = len(np.unique(Y_encoded))
Y_onehot = tf.keras.utils.to_categorical(Y_encoded, num_classes=num_classes)

print(Y_onehot.shape)

# Applying the same train, testing, & validation split that this repo does to PAMAP2 
X_train, X_temp, y_train, y_temp = train_test_split(
    X, Y_onehot,
    test_size=0.3,
    stratify=Y,
    random_state=42
)

# Then split temp into val + test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=np.argmax(y_temp, axis=1),
    random_state=42
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# Loading in the .yaml file 
model_config_file = open('configs/model.yaml', mode='r')
model_config = yaml.load(model_config_file, Loader=yaml.FullLoader)

# Training the pre-trained model onto the WISDM dataset 
train_model(
    dataset="wisdm",
    model_config=model_config,
    train_x=X_train,
    train_y=y_train,
    val_x=X_val,
    val_y=y_val,
    epochs=50
)

# Now creating the predictions to see how the model performed 
pred = test_model(
    dataset="wisdm",
    model_config=model_config,
    test_x=X_test
)

print('\n[MODEL INFERENCE]')

generate_result(dataset="wisdm", ground_truth=y_test, prediction=pred)