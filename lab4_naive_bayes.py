"""
Lab 4 - Naive Bayes Classifier
- Uses Iris dataset (no external CSV needed)
- GaussianNB for continuous features
- Accuracy + classification report
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

d = load_iris()
X, y = d.data, d.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_tr, y_tr)
y_pred = nb.predict(X_te)

print(f"Accuracy: {accuracy_score(y_te, y_pred)*100:.2f}%")
print("\n── Classification Report ──")
print(classification_report(y_te, y_pred, target_names=d.target_names))

# ── If using your own CSV ─────────────────────────────────
# import pandas as pd
# df = pd.read_csv('data.csv')
# df.dropna(inplace=True)
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
# ... rest of code is same
