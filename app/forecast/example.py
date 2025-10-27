import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('airline_passengers.csv', parse_dates=['Month'], index_col='Month')
series = data['Passengers'].astype(float)

train_size = int(len(series) * 0.8)
train, test = series.iloc[:train_size], series.iloc[train_size:]

model = ARIMA(train, order=(2, 1, 2), enforce_stationarity=False, enforce_invertibility=False)
results = model.fit()

forecast = results.get_forecast(steps=len(test))
predicted_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

residuals = test.to_numpy() - predicted_mean.to_numpy()
mae = float(np.mean(np.abs(residuals)))
rmse = float(np.sqrt(np.mean(np.square(residuals))))

print("Testing Metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print()
print("Confidence interval for last prediction:")
print(conf_int.iloc[-1])
