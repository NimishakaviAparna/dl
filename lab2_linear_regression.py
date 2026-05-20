"""
Lab 2 - Simple and Multiple Linear Regression
- Handle missing values
- Simple LR (one feature)
- Multiple LR (multiple features)
- MSE, RMSE, R2 evaluation
- Plot regression line
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load dataset (California Housing - no CSV needed)
data = fetch_california_housing(as_frame=True)
df = data.frame
df.fillna(df.mean(), inplace=True)

# ── Simple Linear Regression ──────────────────────────────
X1 = df[['AveRooms']]
y  = df['MedHouseVal']

X_tr, X_te, y_tr, y_te = train_test_split(X1, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_tr, y_tr)
y_pred = lr.predict(X_te)

mse  = mean_squared_error(y_te, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_te, y_pred)
print("── Simple Linear Regression ──")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

plt.figure(figsize=(7, 4))
plt.scatter(X_te, y_te, color='steelblue', alpha=0.4, s=10, label='Actual')
plt.plot(X_te.sort_values('AveRooms'),
         lr.predict(X_te.sort_values('AveRooms')), color='red', label='Predicted')
plt.xlabel('Average Rooms')
plt.ylabel('House Value')
plt.title('Simple Linear Regression')
plt.legend()
plt.tight_layout()
plt.savefig('simple_lr.png')
plt.show()

# ── Multiple Linear Regression ────────────────────────────
X_m = df[['AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_m, y, test_size=0.2, random_state=42)

lr2 = LinearRegression()
lr2.fit(X_tr2, y_tr2)
y_p2 = lr2.predict(X_te2)

print("\n── Multiple Linear Regression ──")
print(f"MSE : {mean_squared_error(y_te2, y_p2):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_te2, y_p2)):.4f}")
print(f"R²  : {r2_score(y_te2, y_p2):.4f}")

plt.figure(figsize=(7, 4))
plt.scatter(range(len(y_te2)), y_te2.values, alpha=0.4, s=8, label='Actual')
plt.scatter(range(len(y_p2)), y_p2, alpha=0.4, s=8, label='Predicted')
plt.title('Multiple LR - Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.savefig('multiple_lr.png')
plt.show()
