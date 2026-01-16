import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths & Parameters

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset/train"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)


# Data Preprocessing

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)


# Load MobileNetV2 Base Model

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze most layers (important for size & stability)
for layer in base_model.layers[:-20]:
    layer.trainable = False


# Custom Classification Head

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)


# Compile Model

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train Model

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save Model 

model.save(MODEL_PATH, include_optimizer=False)

print("\n✅ Model training completed successfully!")

