import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def download_historical_data(tickers, start_date, end_date):
    """
    Download historical financial data for the given tickers and date range.

    Parameters:
    - tickers: List of ticker symbols (e.g., ['TSLA', 'BND', 'SPY'])
    - start_date: Start date for the data (e.g., '2020-01-01')
    - end_date: End date for the data (e.g., '2023-12-31')

    Returns:
    - Pandas DataFrame containing the historical data
    """
    return yf.download(tickers, start=start_date, end=end_date)

def separate_data(historical_data, tickers):
    """
    Separate the historical data into individual DataFrames for each ticker.

    Parameters:
    - historical_data: DataFrame containing the historical data
    - tickers: List of ticker symbols

    Returns:
    - Dictionary of DataFrames, one for each ticker
    """
    data_frames = {}
    for ticker in tickers:
        data = historical_data['Close'][ticker].reset_index()
        data.columns = ['Date', 'Close']
        data['Open'] = historical_data['Open'][ticker].values
        data['High'] = historical_data['High'][ticker].values
        data['Low'] = historical_data['Low'][ticker].values
        data['Volume'] = historical_data['Volume'][ticker].values
        data_frames[ticker] = data
    return data_frames

def check_basic_statistics(data_frames):
    """
    Print basic statistics for each ticker's data.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        print(f"Basic Statistics for {ticker}:")
        print(data.describe(), "\n")

def ensure_data_types(data_frames):
    """
    Ensure all columns have appropriate data types.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        print(f"Data Types for {ticker}:")
        print(data.dtypes, "\n")

def standardize_data_types(data_frames):
    """
    Standardize the data types of numerical columns to float64.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker

    Returns:
    - Dictionary of DataFrames with standardized data types
    """
    for ticker, data in data_frames.items():
        # Convert numerical columns to float64
        data_frames[ticker] = data.astype({
            'Close': 'float64',
            'Open': 'float64',
            'High': 'float64',
            'Low': 'float64',
            'Volume': 'float64'
        })
    return data_frames
def check_missing_values(data_frames):
    """
    Check for missing values in each ticker's data.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        print(f"Missing Values for {ticker}:")
        print(data.isnull().sum(), "\n")

def handle_missing_values(data_frames, method='ffill'):
    """
    Handle missing values by filling, interpolating, or removing them.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    - method: Method to handle missing values ('ffill', 'bfill', 'interpolate', 'drop')
    """
    for ticker, data in data_frames.items():
        if method == 'drop':
            data_frames[ticker] = data.dropna()
        else:
            data_frames[ticker] = data.fillna(method=method)
    return data_frames

def normalize_data(data_frames):
    """
    Normalize or scale the data using MinMaxScaler.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    scaler = MinMaxScaler()
    for ticker, data in data_frames.items():
        data_frames[ticker][['Close', 'Open', 'High', 'Low', 'Volume']] = scaler.fit_transform(
            data[['Close', 'Open', 'High', 'Low', 'Volume']]
        )
    return data_frames

def display_cleaned_data(data_frames):
    """
    Display the first few rows of each DataFrame after cleaning and scaling.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        print(f"Cleaned and Scaled Data for {ticker}:")
        print(data.head(), "\n")

def plot_closing_prices(data_frames):
    """
    Visualize the closing price over time for each ticker.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close'], label='Close Price')
        plt.title(f'{ticker} Closing Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid()
        plt.show()

def calculate_daily_returns(data_frames):
    """
    Calculate and plot the daily percentage change (returns) for each ticker.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    """
    for ticker, data in data_frames.items():
        data['Daily Return'] = data['Close'].pct_change() * 100
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Daily Return'], label='Daily Return', color='orange')
        plt.title(f'{ticker} Daily Percentage Change (Returns)')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.legend()
        plt.grid()
        plt.show()

def analyze_volatility(data_frames, window=30):
    """
    Analyze volatility by calculating rolling means and standard deviations.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    - window: Rolling window size (default: 30 days)
    """
    for ticker, data in data_frames.items():
        data['Rolling Mean'] = data['Close'].rolling(window=window).mean()
        data['Rolling Std'] = data['Close'].rolling(window=window).std()

        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        plt.plot(data['Date'], data['Rolling Mean'], label=f'{window}-Day Rolling Mean', color='green')
        plt.plot(data['Date'], data['Rolling Std'], label=f'{window}-Day Rolling Std', color='red')
        plt.title(f'{ticker} Volatility Analysis (Rolling Mean & Std)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()






def decompose_time_series_all_in_one(data_frames, period=252):
    """
    Decompose the time series into trend, seasonal, and residual components for all assets.
    Visualize each component (trend, seasonal, residual) in a single plot for all assets.
    
    Parameters:
        data_frames (dict): A dictionary where keys are ticker symbols (e.g., 'TSLA') and values are DataFrames.
        period (int): The period for seasonal decomposition (default is 252 for annual seasonality in trading days).
    """
    # Initialize dictionaries to store decomposed components
    trends = {}
    seasonals = {}
    residuals = {}

    # Perform decomposition for each asset
    for ticker, data in data_frames.items():
        # Ensure the data is sorted by date
        data = data.sort_values('Date')
        
        # Set the date column as the index
        data.set_index('Date', inplace=True)
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(data['Close'], model='additive', period=period)
        
        # Store the components
        trends[ticker] = decomposition.trend
        seasonals[ticker] = decomposition.seasonal
        residuals[ticker] = decomposition.resid

    # Plot Trend Components
    plt.figure(figsize=(14, 7))
    for ticker, trend in trends.items():
        plt.plot(trend.index, trend, label=f'{ticker} Trend')
    plt.title('Trend Components for TSLA, BND, and SPY')
    plt.xlabel('Date')
    plt.ylabel('Trend')
    plt.legend()
    plt.show()

    # Plot Seasonal Components
    plt.figure(figsize=(14, 7))
    for ticker, seasonal in seasonals.items():
        plt.plot(seasonal.index, seasonal, label=f'{ticker} Seasonal')
    plt.title('Seasonal Components for TSLA, BND, and SPY')
    plt.xlabel('Date')
    plt.ylabel('Seasonal')
    plt.legend()
    plt.show()

    # Plot Residual Components
    plt.figure(figsize=(14, 7))
    for ticker, residual in residuals.items():
        plt.plot(residual.index, residual, label=f'{ticker} Residual')
    plt.title('Residual Components for TSLA, BND, and SPY')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.legend()
    plt.show()

def visualize_All_in_one(data_frames):
    # Plot Closing Price Over Time
    plt.figure(figsize=(14, 7))
    for ticker, data in data_frames.items():
        plt.plot(data['Date'], data['Close'], label=ticker)
    plt.title('Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    # Plot Daily Percentage Change
    plt.figure(figsize=(14, 7))
    for ticker, data in data_frames.items():
        data['Daily Return'] = data['Close'].pct_change()
        plt.plot(data['Date'], data['Daily Return'], label=ticker)
    plt.title('Daily Percentage Change')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.show()

    # Plot Rolling Mean (Volatility Analysis)
    plt.figure(figsize=(14, 7))
    for ticker, data in data_frames.items():
        data['Rolling Mean'] = data['Close'].rolling(window=20).mean()
        plt.plot(data['Date'], data['Rolling Mean'], label=f'{ticker} Rolling Mean')
    plt.title('Volatility Analysis (Rolling Mean)')
    plt.xlabel('Date')
    plt.ylabel('Rolling Mean')
    plt.legend()
    plt.show()

    # Plot Rolling Standard Deviation (Volatility Analysis)
    plt.figure(figsize=(14, 7))
    for ticker, data in data_frames.items():
        data['Rolling Std'] = data['Close'].rolling(window=20).std()
        plt.plot(data['Date'], data['Rolling Std'], label=f'{ticker} Rolling Std')
    plt.title('Volatility Analysis (Rolling Standard Deviation)')
    plt.xlabel('Date')
    plt.ylabel('Rolling Std')
    plt.legend()
    plt.show()


def detect_outliers(data_frames, threshold=3):
    """
    Detect outliers using the Z-score method.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    - threshold: Z-score threshold for outlier detection (default: 3)
    """
    from scipy.stats import zscore

    for ticker, data in data_frames.items():
        data['Z-Score'] = zscore(data['Close'])
        outliers = data[abs(data['Z-Score']) > threshold]

        print(f"Outliers for {ticker}:")
        print(outliers[['Date', 'Close', 'Z-Score']], "\n")

def analyze_unusual_returns(data_frames, threshold=2):
    """
    Analyze days with unusually high or low returns.

    Parameters:
    - data_frames: Dictionary of DataFrames, one for each ticker
    - threshold: Threshold for unusual returns (default: 2%)
    """
    for ticker, data in data_frames.items():
        if 'Daily Return' not in data.columns:
            data['Daily Return'] = data['Close'].pct_change() * 100

        unusual_returns = data[(data['Daily Return'] > threshold) | (data['Daily Return'] < -threshold)]

        print(f"Unusual Returns for {ticker}:")
        print(unusual_returns[['Date', 'Close', 'Daily Return']], "\n")

def save_ticker_data(data_frames, output_directory):
    for ticker, data in data_frames.items():
        # Define the file path
        file_path = f"{output_directory}/{ticker}_data.csv"

        # Save the DataFrame to a CSV file
        data.to_csv(file_path, index=False)
        print(f"Data for {ticker} saved to {file_path}")