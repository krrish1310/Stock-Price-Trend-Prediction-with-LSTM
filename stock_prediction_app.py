# Bonus: Create Streamlit Dashboard
# --------------------------------
# Save this code to a separate file named 'stock_prediction_app.py'

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import talib

# Load the model
@st.cache_resource
def load_lstm_model():
    return load_model('my_model.keras')

# Function to fetch stock data
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Add technical indicators function
def add_technical_indicators(df):
    # Calculate RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    
    # Calculate Simple Moving Averages
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    
    # Calculate Exponential Moving Averages
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    
    # Calculate MACD
    macd, macd_signal, macd_hist = talib.MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    
    # Calculate Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

# Prepare data for prediction
def prepare_prediction_data(df, feature_columns, sequence_length=60):
    data_features = df[feature_columns].values
    
    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_features)
    
    # Create the most recent sequence
    X_recent = []
    X_recent.append(data_scaled[-sequence_length:])
    X_recent = np.array(X_recent)
    
    return X_recent, scaler

# Streamlit app
st.title('Stock Price Prediction Dashboard')

# Sidebar inputs
st.sidebar.header('User Input Parameters')
ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')
start_date = st.sidebar.date_input('Start Date', dt.date(2018, 1, 1))
end_date = st.sidebar.date_input('End Date', dt.date.today())
prediction_days = st.sidebar.slider('Future Prediction Days', 1, 90, 30)

# Load data
try:
    df = fetch_stock_data(ticker, start_date, end_date)
    
    # Display raw data
    st.subheader('Raw Stock Data')
    st.write(df.tail())
    
    # Plot close price
    st.subheader('Close Price History')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.grid(True)
    st.pyplot(fig)
    
    # Add technical indicators
    df_with_indicators = add_technical_indicators(df.copy())
    
    # Display technical indicators
    st.subheader('Technical Indicators')
    cols_to_plot = ['Close', 'SMA_20', 'SMA_50', 'EMA_20']
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot price and moving averages
    ax1.plot(df_with_indicators['Close'], label='Close')
    ax1.plot(df_with_indicators['SMA_20'], label='SMA 20')
    ax1.plot(df_with_indicators['SMA_50'], label='SMA 50')
    ax1.plot(df_with_indicators['EMA_20'], label='EMA 20')
    ax1.set_title('Price and Moving Averages')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot RSI
    ax2.plot(df_with_indicators['RSI'], color='purple')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_title('RSI')
    ax2.set_ylabel('RSI Value')
    ax2.grid(True)
    
    # Plot MACD
    ax3.plot(df_with_indicators['MACD'], label='MACD')
    ax3.plot(df_with_indicators['MACD_signal'], label='Signal')
    ax3.set_title('MACD')
    ax3.set_ylabel('MACD Value')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Make prediction when button is clicked
    if st.button('Predict Future Prices'):
        # Load the model
        model = load_lstm_model()
        
        # Prepare recent data for prediction
        feature_columns = ['Close', 'RSI', 'SMA_20', 'EMA_20', 'MACD', 'BB_upper', 'BB_lower']
        X_recent, scaler = prepare_prediction_data(df_with_indicators, feature_columns)
        
        # Predict future prices (simple approach)
        last_price = df['Close'].iloc[-1]
        future_prices = [last_price]
        
        current_batch = X_recent.copy()
        for i in range(prediction_days):
            # Predict next price
            pred = model.predict(current_batch)
            
            # Scale the prediction back
            prediction_array = np.zeros((1, len(feature_columns)))
            prediction_array[0, 0] = pred[0, 0]  # Set the predicted closing price
            
            # For simplicity, we're keeping other features constant
            pred_price = scaler.inverse_transform(prediction_array)[0, 0]
            future_prices.append(pred_price)
            
            # Update the batch for next prediction (simplified)
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0, -1, 0] = pred[0, 0]  # Replace the last element with our prediction
        
        # Plot the prediction
        future_dates = pd.date_range(start=df.index[-1], periods=prediction_days+1)[1:]
        historical_dates = df.index[-60:]  # Last 60 days
        
        st.subheader('Future Price Prediction')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical_dates, df['Close'][-60:], label='Historical Prices')
        ax.plot(future_dates, future_prices[1:], color='red', label='Predicted Prices')
        ax.axvline(x=df.index[-1], color='green', linestyle='--', label='Today')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price')
        ax.set_title(f'{ticker} Stock Price Prediction')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Display prediction summary
        st.subheader('Prediction Summary')
        prediction_summary = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_prices[1:]
        })
        st.write(prediction_summary)
        
        # Calculate and display potential return
        current_price = df['Close'].iloc[-1]
        predicted_price = future_prices[-1]
        potential_return = (predicted_price / current_price - 1) * 100
        
        st.subheader('Investment Potential')
        col1, col2, col3 = st.columns(3)
        col1.metric('Current Price', f'${current_price:.2f}')
        col2.metric('Predicted Price', f'${predicted_price:.2f}')
        col3.metric('Potential Return', f'{potential_return:.2f}%')
        
except Exception as e:
    st.error(f"Error: {e}")
    st.write("Please make sure you've entered a valid stock ticker and date range.")

