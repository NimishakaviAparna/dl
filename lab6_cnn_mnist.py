"""
Lab 6 - CNN Multiclass Classifier on MNIST
- Data preprocessing
- CNN model definition & training
- Confusion matrix evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix

# ── Data ──────────────────────────────────────────────────
(X_tr, y_tr), (X_te, y_te) = mnist.load_data()
X_tr = X_tr.reshape(-1, 28, 28, 1) / 255.0
X_te = X_te.reshape(-1, 28, 28, 1) / 255.0

# ── Model ─────────────────────────────────────────────────
m = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

m.summary()
m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ── Train ─────────────────────────────────────────────────
h = m.fit(X_tr, y_tr, epochs=5, batch_size=64, validation_split=0.1)

# ── Evaluate ──────────────────────────────────────────────
loss, acc = m.evaluate(X_te, y_te)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# ── Confusion Matrix ──────────────────────────────────────
y_pred = np.argmax(m.predict(X_te), axis=1)
cm = confusion_matrix(y_te, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - MNIST CNN')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ── Training curves ───────────────────────────────────────
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
plt.savefig('training_curves.png')
plt.show()
