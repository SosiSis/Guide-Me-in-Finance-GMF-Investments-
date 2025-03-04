import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def prepare_lstm_data(data, sequence_length=60):
    """
    Prepare data for LSTM by scaling and creating sequences.

    Parameters:
    - data: DataFrame containing the time series data.
    - sequence_length: Length of the input sequences for LSTM.

    Returns:
    - X_train, X_test: Training and testing input sequences.
    - y_train, y_test: Training and testing target values.
    - scaler: Fitted scaler for inverse transform.
    """
    # Scale the data
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    # Create sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(data, sequence_length)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test

def build_lstm_model(input_shape):
    """
    Build an LSTM model.

    Parameters:
    - input_shape: Shape of the input data.

    Returns:
    - model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_lstm_model(model, X_train, y_train, epochs=100, batch_size=32):
    """
    Train the LSTM model.

    Parameters:
    - model: Compiled LSTM model.
    - X_train, y_train: Training data.
    - epochs: Number of training epochs.
    - batch_size: Batch size for training.

    Returns:
    - history: Training history.
    """

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    return history



def forecast_lstm(model, X_test):
    """
    Make predictions using the LSTM model.

    Parameters:
    - model: Trained LSTM model.
    - X_test: Testing input sequences.
    - scaler: Fitted scaler for inverse transform.

    Returns:
    - predictions: Forecasted values.
    """
    predictions = model.predict(X_test)
    
    return predictions

def evaluate_lstm_model(y_true, y_pred):
    """
    Evaluate the LSTM model using MAE, RMSE, and MAPE.

    Parameters:
    - y_true: True values.
    - y_pred: Predicted values.

    Returns:
    - mae, rmse, mape: Evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


import matplotlib.pyplot as plt

def plot_lstm_forecast(y_true, y_pred):
    """
    Plot the forecasted values against the actual values.

    Parameters:
    - y_true: True values.
    - y_pred: Predicted values.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predicted Values', color='red')
    plt.title('LSTM Stock Price Forecast')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()




def plot_loss(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
