import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Load historical data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
def preprocess_data(data, target_column, sequence_length):
    # Extract the target column (e.g., 'Close' prices)
    target_data = data[target_column].values.reshape(-1, 1)
    
   
   
    
    # Create sequences for LSTM input
    X = []
    for i in range(sequence_length, len(target_data)):
        X.append(target_data[i-sequence_length:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X

def generate_forecast(model, X,  forecast_steps, sequence_length):
    """
    Generate future forecasts using the trained LSTM model.

    Args:
        model: Trained LSTM model.
        X: Input data (sequences) for the model.
        scaler: Scaler used to normalize the data.
        forecast_steps: Number of steps to forecast into the future.
        sequence_length: Length of the input sequences used by the model.

    Returns:
        forecast: Predicted future values.
    """
    # Predict future values

    
    # Generate future forecasts
    last_sequence = X[-1]  # Use the last sequence from the input data
    forecast = []
    for _ in range(forecast_steps):
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
        forecast.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred)  # Update the sequence
    
   
    return forecast


def plot_all_forecasts(historical_datasets, forecasts, confidence_interval, company_names):
    """
    Plot multiple historical stock data and their forecasts in one graph.

    Args:
        historical_datasets: List of DataFrames containing historical stock prices.
        forecasts: List of arrays containing forecasted values.
        confidence_interval: Scalar value representing the confidence interval range.
        company_names: List of company names corresponding to each dataset.
    """
    plt.figure(figsize=(12, 6))

    colors = ['blue', 'orange', 'green']  # Define colors for each stock
    
    for i, (historical_data, forecast, company) in enumerate(zip(historical_datasets, forecasts, company_names)):
        forecast = np.array(forecast)  # Ensure forecast is a NumPy array

        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=historical_data['Date'].iloc[-1],  # Start from last historical date
            periods=len(forecast) + 1,
            inclusive='right'
        )

        # Plot historical data
        plt.plot(historical_data['Date'], historical_data['Close'], label=f'{company} Historical', color=colors[i], linestyle='dashed')

        # Plot forecast
        plt.plot(forecast_dates, forecast, label=f'{company} Forecast', color=colors[i])

        # Plot confidence interval
        plt.fill_between(
            forecast_dates,
            forecast - confidence_interval,
            forecast + confidence_interval,
            color=colors[i],
            alpha=0.2
        )

    # Set labels, title, and legend
    plt.title("Stock Price Forecasts")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
def analyze_forecast(forecast, confidence_interval):
    # Trend Analysis
    trend = 'Upward' if forecast[-1] > forecast[0] else 'Downward' if forecast[-1] < forecast[0] else 'Stable'
    print(f"Trend Analysis: {trend} trend observed.")
    
    # Volatility and Risk
    volatility = np.std(forecast)
    print(f"Volatility: {volatility:.2f}")
    print(f"Confidence Interval: ±{confidence_interval}")
    
    # Market Opportunities and Risks
    if trend == 'Upward':
        print("Market Opportunity: Potential for price appreciation.")
    elif trend == 'Downward':
        print("Market Risk: Potential for price decline.")
    else:
        print("Market Outlook: Stable with moderate risk.")
