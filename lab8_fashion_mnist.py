"""
Lab 8 - CNN Classifier on Fashion MNIST
- 10 clothing categories
- BatchNormalization + Dropout
- Confusion matrix + sample predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix

NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ── Data ──────────────────────────────────────────────────
(X_tr, y_tr), (X_te, y_te) = fashion_mnist.load_data()
X_tr = X_tr[..., np.newaxis] / 255.0
X_te = X_te[..., np.newaxis] / 255.0

# ── Model ─────────────────────────────────────────────────
m = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

m.summary()
m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ── Train ─────────────────────────────────────────────────
h = m.fit(X_tr, y_tr, epochs=10, batch_size=64, validation_split=0.1)

# ── Evaluate ──────────────────────────────────────────────
loss, acc = m.evaluate(X_te, y_te)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# ── Confusion matrix ──────────────────────────────────────
y_pred = np.argmax(m.predict(X_te), axis=1)
cm = confusion_matrix(y_te, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=NAMES, yticklabels=NAMES)
plt.xticks(rotation=45, ha='right')
plt.title('Confusion Matrix - Fashion MNIST')
plt.tight_layout()
plt.savefig('fashion_cm.png')
plt.show()

# ── Sample predictions ────────────────────────────────────
pred5 = m.predict(X_te[:5])
print("\n── Sample Predictions ──")
for i in range(5):
    print(f"Actual: {NAMES[y_te[i]]:<12} Predicted: {NAMES[np.argmax(pred5[i])]}")
