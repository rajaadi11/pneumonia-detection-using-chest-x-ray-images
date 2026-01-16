import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Silence TensorFlow logs (VERY IMPORTANT on Windows)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_PATH = os.path.join("model", "disease_model.h5")
DATASET_DIR = "dataset/train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load model
model = load_model(MODEL_PATH)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Collect true labels
y_true = np.concatenate([y.numpy().flatten() for _, y in val_ds])

# Predict silently
y_pred_prob = model.predict(val_ds, verbose=0).flatten()

# Apply threshold
THRESHOLD = 0.55
y_pred = (y_pred_prob >= THRESHOLD).astype(int)

# Print evaluation (NO UNICODE)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
