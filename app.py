import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import load_model
import streamlit as st

# App Title and Description
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title("📈 Stock Trend Prediction")
st.write("Analyze and predict stock trends using historical data and machine learning models.")

# Sidebar for User Input
st.sidebar.header("Stock Selection")
user_input = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2024-07-01"))

# Download Data
df = yf.download(user_input, start=start, end=end)
st.sidebar.write(f"Data Range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

# Data Summary
st.subheader("Data Summary")
st.dataframe(df.describe())

# Plots Layout
st.subheader("Stock Price Analysis")


# Plot 1: Closing Price vs Time
st.write("### Closing Price vs Time")
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df.Close, color='blue', label='Closing Price')
ax1.set_xlabel("Time (Year)")
ax1.set_ylabel("Closing Price")
ax1.legend(loc="upper left")
st.pyplot(fig1)

# Plot 2: Closing Price with 100/200-Day Moving Average
st.write("### Closing Price with 100/200-Day Moving Averages")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df.Close, color='blue', label='Closing Price')
ax2.plot(ma100, color='orange', label='100-Day MA')
ax2.plot(ma200, color='green', label='200-Day MA')
ax2.set_xlabel("Time (Year)")
ax2.set_ylabel("Closing Price")
ax2.legend(loc="upper left")
st.pyplot(fig2)

# Data Preparation
train_size = int(len(df) * 0.7)
train_data = df[:train_size]
test_data = df[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Creating training and testing datasets
window_size = 150
X_train, y_train = [], []
X_test, y_test = [], []

for i in range(window_size, len(train_data)):
    X_train.append(train_data[i - window_size:i, 0])
    y_train.append(train_data[i, 0])

for i in range(window_size, len(test_data)):
    X_test.append(test_data[i - window_size:i, 0])
    y_test.append(test_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Load Model and Predict
model = load_model("LSTM_Model.h5")
y_pred = model.predict(X_test)

close_column_index = 3
min_value = scaler.data_min_[close_column_index]
max_value = scaler.data_max_[close_column_index]
value_range = max_value - min_value

y_pred_no_scale = (y_pred * value_range) + min_value
y_test_no_scale = (y_test * value_range) + min_value

# Plot 3: Predicted vs Actual Prices
st.subheader("Predicted vs Actual Closing Prices")
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(y_test_no_scale, color='blue', label='Actual Price')
ax3.plot(y_pred_no_scale, color='red', label='Predicted Price')
ax3.set_xlabel("Time (Year)")
ax3.set_ylabel("Closing Price")
ax3.legend(loc="upper left")
st.pyplot(fig3)

# Custom Footer
st.markdown("<br><br><hr>", unsafe_allow_html=True)