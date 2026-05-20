"""
Lab 5 - Decision Tree, Pruning, Random Forest, AdaBoost
- Train/test split
- Decision tree with overfitting check
- Cost complexity pruning
- Random Forest
- AdaBoost
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# ── 1. Decision Tree (unpruned) ───────────────────────────
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_tr, y_tr)
print("── Decision Tree (unpruned) ──")
print(f"Train Acc : {accuracy_score(y_tr, dt.predict(X_tr)):.4f}")
print(f"Test  Acc : {accuracy_score(y_te, dt.predict(X_te)):.4f}")

plt.figure(figsize=(12, 5))
plot_tree(dt, filled=True, feature_names=load_iris().feature_names,
          class_names=load_iris().target_names)
plt.title('Decision Tree (Unpruned)')
plt.tight_layout()
plt.savefig('decision_tree.png')
plt.show()

# ── 2. Cost Complexity Pruning ────────────────────────────
dt_p = DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
dt_p.fit(X_tr, y_tr)
print("\n── Decision Tree (pruned, ccp_alpha=0.01) ──")
print(f"Train Acc : {accuracy_score(y_tr, dt_p.predict(X_tr)):.4f}")
print(f"Test  Acc : {accuracy_score(y_te, dt_p.predict(X_te)):.4f}")

# ── 3. Random Forest ──────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_tr, y_tr)
print("\n── Random Forest (100 trees) ──")
print(f"Test  Acc : {accuracy_score(y_te, rf.predict(X_te)):.4f}")

# ── 4. AdaBoost ───────────────────────────────────────────
ab = AdaBoostClassifier(n_estimators=50, random_state=42)
ab.fit(X_tr, y_tr)
print("\n── AdaBoost (50 stumps) ──")
print(f"Test  Acc : {accuracy_score(y_te, ab.predict(X_te)):.4f}")
