"""
Lab 7 - CNN for Plant Disease Detection
- ImageDataGenerator for augmentation
- CNN with Dropout
- Accuracy/loss curves
Dataset: Place images in plant_data/ folder with one subfolder per disease class
         OR use the PlantVillage dataset from Kaggle
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kagglehub

path = kagglehub.dataset_download("emmarex/plantdisease")

print("Path to dataset files:", path)

IMG_SIZE  = (64, 64)
BATCH     = 32
EPOCHS    = 10
DATA_DIR  = path

# ── Data generators with augmentation ────────────────────
gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.1
)

train_data = gen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
    batch_size=BATCH, class_mode='categorical', subset='training')
val_data   = gen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
    batch_size=BATCH, class_mode='categorical', subset='validation')

n_cls = len(train_data.class_indices)
print(f"Classes found: {train_data.class_indices}")

# ── Model ─────────────────────────────────────────────────
m = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_cls, activation='softmax')
])

m.summary()
m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ── Train ─────────────────────────────────────────────────
h = m.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# ── Curves ────────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h.history['accuracy'], label='train')
plt.plot(h.history['val_accuracy'], label='val')
plt.title('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(h.history['loss'], label='train')
plt.plot(h.history['val_loss'], label='val')
plt.title('Loss'); plt.legend()
plt.tight_layout()
plt.savefig('plant_training_curves.png')
plt.show()

m.save('plant_disease_model.h5')
print("Model saved.")
