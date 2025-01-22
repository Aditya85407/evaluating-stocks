# evaluating-stocks
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Function to generate synthetic stock data (for example purposes)
def generate_synthetic_stock_data(num_samples=1000):
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=num_samples)
    prices = np.cumsum(np.random.randn(num_samples)) + 100  # Random walk
    return pd.DataFrame({"Date": dates, "Close": prices})

# Generate synthetic stock data
data = generate_synthetic_stock_data()
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

# Preparing the data for LSTM
sequence_length = 60  # Use 60 previous days to predict the next day
x_data, y_data = [], []
for i in range(sequence_length, len(data_scaled)):
    x_data.append(data_scaled[i - sequence_length:i, 0])
    y_data.append(data_scaled[i, 0])

x_data, y_data = np.array(x_data), np.array(y_data)
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Building the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)  # Prediction of the next price
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
epochs = 50
batch_size = 32
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# Making predictions
y_pred = model.predict(x_test)

# Rescaling predictions back to original scale
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Saving the model
model.save("lstm_stock_prediction_model.h5")

# Example visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, color='blue', label='Actual Prices')
plt.plot(y_pred_rescaled, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
