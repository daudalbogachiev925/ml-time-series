"""Прогнозирование временных рядов."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error

# === Данные ===
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=365)
trend = np.linspace(100, 200, 365)
seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 30)
noise = np.random.normal(0, 5, 365)
series = pd.Series(trend + seasonal + noise, index=dates)

# === Декомпозиция ===
decomp = seasonal_decompose(series, model='additive', period=30)
decomp.plot()
plt.savefig("decomposition.png")
plt.show()

# === Train/Test split ===
train = series[:-30]
test = series[-30:]

# === ARIMA ===
model = ARIMA(train, order=(2, 1, 2)).fit()
forecast = model.forecast(steps=30)

mae = mean_absolute_error(test, forecast)
print(f"MAE: {mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', color='green')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.legend()
plt.title("ARIMA прогноз")
plt.savefig("forecast.png")
plt.show()
