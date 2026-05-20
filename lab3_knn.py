"""
Lab 3 - K-Nearest Neighbours on Iris Dataset
- Print correct and wrong predictions
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

d = load_iris()
X, y = d.data, d.target
names = d.target_names

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_tr, y_tr)
y_pred = knn.predict(X_te)

print("── Predictions ──")
for i in range(len(y_te)):
    status = "✓ CORRECT" if y_te[i] == y_pred[i] else "✗ WRONG"
    print(f"[{i+1:02d}] Actual: {names[y_te[i]]:<12} Predicted: {names[y_pred[i]]:<12} {status}")

print(f"\nAccuracy: {accuracy_score(y_te, y_pred)*100:.2f}%")
print("\n── Classification Report ──")
print(classification_report(y_te, y_pred, target_names=names))
