from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load trained model
model = load_model("model/disease_model.h5")

# Image preprocessing (same as training)
img_size = 224

datagen = ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    'dataset/train',
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# True labels
y_true = val_data.classes

# Predictions
y_pred = model.predict(val_data)
y_pred = (y_pred > 0.5).astype(int).reshape(-1)

# Results
print("Classification Report:\n")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
