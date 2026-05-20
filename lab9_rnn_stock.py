"""
Lab 9 - RNN for Stock Price Prediction
- MinMaxScaler normalization
- 60-step sliding window sequences
- SimpleRNN model
- Inverse transform to real prices
Dataset: Download any stock CSV from Yahoo Finance (e.g., AAPL.csv)
         Column needed: 'Close'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

# ── Generate synthetic data if no CSV available ───────────
# Replace this block with: df = pd.read_csv('AAPL.csv')
np.random.seed(42)
t = np.linspace(0, 100, 2000)
prices = 150 + 50 * np.sin(t / 10) + np.random.randn(2000) * 5
df = pd.DataFrame({'Close': prices})
# ─────────────────────────────────────────────────────────

data = df['Close'].values.reshape(-1, 1)

sc = MinMaxScaler()
data_s = sc.fit_transform(data)

# ── Create sequences ──────────────────────────────────────
def mk_seq(d, steps=60):
    X, y = [], []
    for i in range(steps, len(d)):
        X.append(d[i - steps:i, 0])
        y.append(d[i, 0])
    return np.array(X), np.array(y)

X, y = mk_seq(data_s)
split = int(len(X) * 0.8)
X_tr, X_te = X[:split].reshape(-1, 60, 1), X[split:].reshape(-1, 60, 1)
y_tr, y_te = y[:split], y[split:]

# ── Model ─────────────────────────────────────────────────
m = Sequential([
    SimpleRNN(64, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    SimpleRNN(32),
    Dense(1)
])

m.summary()
m.compile(optimizer='adam', loss='mse')
m.fit(X_tr, y_tr, epochs=20, batch_size=32, validation_split=0.1)

# ── Predict & inverse transform ───────────────────────────
pred   = sc.inverse_transform(m.predict(X_te).reshape(-1, 1))
actual = sc.inverse_transform(y_te.reshape(-1, 1))

plt.figure(figsize=(10, 4))
plt.plot(actual, label='Actual Price', color='steelblue')
plt.plot(pred,   label='Predicted',   color='tomato')
plt.xlabel('Time Step')
plt.ylabel('Stock Price')
plt.title('RNN - Stock Price Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('stock_prediction.png')
plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(actual, pred))
print(f"RMSE: {rmse:.4f}")
