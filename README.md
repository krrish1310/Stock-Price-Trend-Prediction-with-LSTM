
# 📈 Stock Price Trend Prediction with LSTM

## 🔍 Objective  
Predict future stock prices based on historical data using an LSTM (Long Short-Term Memory) neural network.

---

## 🛠️ Tools & Technologies
- Python 3.x
- Keras / TensorFlow
- Pandas, NumPy
- Matplotlib
- yfinance (Yahoo Finance API)!

- TA-Lib (for technical indicators)
- Streamlit (optional dashboard deployment)

---

## 📘 Project Structure
stock-lstm-predictor/
├── LSTM_Stock_Predictor.ipynb # Main Jupyter notebook
├── model_weights.h5 # Trained LSTM model weights
├── requirements.txt # Project dependencies
├── streamlit_app.py # Streamlit dashboard (optional)
├── README.md # Project documentation

---

## 🧭 Mini-Guide

### 1. Fetch Historical Stock Data  
Use yfinance to download stock data:
```python
import yfinance as yf
data = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
2. Preprocess & Normalize Data
Select the Close prices
Scale data between 0 and 1 with MinMaxScaler
Prepare time-series sequences for LSTM input
3. Build LSTM Model
Example architecture using Keras:
4. Train & Validate Model
Train with training data and validate on test data:![WhatsApp Image 2025-05-18 at 10 41 41 (2)](https://github.com/user-attachments/assets/623c411f-dbb0-48ff-872c-6c096e2379f9)

5. Plot Predictions vs Actual

6. Add Technical Indicators (Moving Average, RSI)
Using TA-Lib:
7. (Optional) Deploy Dashboard with Streamlit
Run the dashboard with:
