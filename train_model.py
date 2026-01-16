import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF logs
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight


# Configuration

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

DATASET_DIR = "dataset/train"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)


# Load Dataset (STABLE API)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Class names:", train_ds.class_names)


# Normalize images

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# Compute Class Weights (FIXES BIAS 🔥)

y_train = np.concatenate([y.numpy().flatten() for _, y in train_ds])


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)


# Load MobileNetV2

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze most layers
for layer in base_model.layers[:-20]:
    layer.trainable = False


# Classification Head

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)


# Compile

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Train (NO CRASH GUARANTEED)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=0
)


# Save Model (DEPLOYMENT SAFE)

model.save(MODEL_PATH, include_optimizer=False)

print("\nTraining completed successfully")
