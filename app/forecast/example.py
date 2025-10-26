import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

# Load the data
df = pd.read_csv('airline_passengers.csv', parse_dates=['Month'], index_col='Month')

# Prepare the data for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        X.append(dataset[i:(i+look_back), 0])
        Y.append(dataset[i+look_back, 0])
    return np.array(X), np.array(Y)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Split into train and test sets
train_size = int(len(scaled_data) * 0.67)
train, test = scaled_data[:train_size], scaled_data[train_size:]

# Reshape into X=t and Y=t+1
look_back = 12  # Use 1 year of historical data for prediction
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(look_back, 1), return_sequences=True),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(
    X_train, Y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_test, Y_test),
    verbose=0
)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train_inv = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test_inv = scaler.inverse_transform([Y_test])

# Calculate metrics
train_mae = mean_absolute_error(Y_train_inv[0], train_predict[:,0])
train_mse = mean_squared_error(Y_train_inv[0], train_predict[:,0])
train_rmse = np.sqrt(train_mse)

test_mae = mean_absolute_error(Y_test_inv[0], test_predict[:,0])
test_mse = mean_squared_error(Y_test_inv[0], test_predict[:,0])
test_rmse = np.sqrt(test_mse)

# Prepare data for comprehensive plotting
# Adjusted date ranges
train_dates = df.index[look_back:train_size]
train_actual = df.iloc[look_back:train_size]['Passengers']
train_predicted = pd.Series(train_predict[:,0], index=train_dates)

test_dates = df.index[train_size+look_back:]
test_actual = df.iloc[train_size+look_back:]['Passengers']
test_predicted = pd.Series(test_predict[:,0], index=test_dates)

# Future forecasting
def forecast_future(model, last_known_data, look_back, scaler, steps=12):
    current_batch = last_known_data[-look_back:]
    predictions = []
    
    for _ in range(steps):
        # Reshape the current batch
        current_input = current_batch.reshape((1, look_back, 1))
        
        # Predict next value
        pred = model.predict(current_input)
        
        # Inverse transform the prediction
        pred_actual = scaler.inverse_transform(pred)[0][0]
        predictions.append(pred_actual)
        
        # Update current batch
        current_batch = np.append(current_batch[1:], pred)
    
    return predictions

# Perform future forecasting
last_known_data = scaled_data[-look_back:]
future_forecast = forecast_future(model, last_known_data, look_back, scaler)

# Create future dates
last_date = pd.to_datetime(df.index[-1])
future_dates = pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(days=1), periods=12, freq='M')
future_forecast_series = pd.Series(future_forecast, index=future_dates)

# Plotting
plt.figure(figsize=(15,6))
# Plot training actual and predicted
plt.plot(train_dates, train_actual, label='Training Actual', color='blue')
plt.plot(train_dates, train_predicted, label='Training Predicted', color='lightblue', linestyle='--')

# Plot testing actual and predicted
plt.plot(test_dates, test_actual, label='Testing Actual', color='green')
plt.plot(test_dates, test_predicted, label='Testing Predicted', color='lightgreen', linestyle='--')

# Plot future forecast
plt.plot(future_dates, future_forecast_series, label='Future Forecast', color='red', linestyle=':')

plt.title('Airline Passengers: Actual vs Predicted with Future Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.legend()
plt.tight_layout()
plt.savefig('lstm_predictions.png')

# Print evaluation metrics
print("Training Metrics:")
print(f"MAE: {train_mae:.2f}")
print(f"MSE: {train_mse:.2f}")
print(f"RMSE: {train_rmse:.2f}")

print("\nTesting Metrics:")
print(f"MAE: {test_mae:.2f}")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")