"""
Lab 10 - LSTM for Weather Prediction
- 30-step sliding windows on temperature data
- Stacked LSTM with Dropout
- Predict future temperature
Dataset: Any weather CSV with a 'Temperature' column
         E.g., download from Kaggle weather datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ── Generate synthetic data if no CSV available ───────────
# Replace with: df = pd.read_csv('weather.csv')
np.random.seed(0)
days = np.arange(1000)
temps = 20 + 10 * np.sin(2 * np.pi * days / 365) + np.random.randn(1000) * 2
df = pd.DataFrame({'Temperature': temps})
# ─────────────────────────────────────────────────────────

data = df['Temperature'].values.reshape(-1, 1)

sc = MinMaxScaler()
data_s = sc.fit_transform(data)

# ── Create sequences ──────────────────────────────────────
def mk_seq(d, steps=30):
    X, y = [], []
    for i in range(steps, len(d)):
        X.append(d[i - steps:i, 0])
        y.append(d[i, 0])
    return np.array(X), np.array(y)

X, y = mk_seq(data_s)
split = int(len(X) * 0.8)
X_tr = X[:split].reshape(-1, 30, 1)
X_te = X[split:].reshape(-1, 30, 1)
y_tr, y_te = y[:split], y[split:]

# ── Model ─────────────────────────────────────────────────
m = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

m.summary()
m.compile(optimizer='adam', loss='mse')
h = m.fit(X_tr, y_tr, epochs=20, batch_size=32, validation_split=0.1)

# ── Predict ───────────────────────────────────────────────
pred   = sc.inverse_transform(m.predict(X_te).reshape(-1, 1))
actual = sc.inverse_transform(y_te.reshape(-1, 1))

plt.figure(figsize=(10, 4))
plt.plot(actual, label='Actual Temp', color='steelblue')
plt.plot(pred,   label='Predicted',  color='tomato')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('LSTM - Weather Temperature Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('weather_prediction.png')
plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(actual, pred))
print(f"RMSE: {rmse:.4f}")
