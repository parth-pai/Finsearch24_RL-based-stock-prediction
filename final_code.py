import numpy as np
import pandas as pd
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
# Load data
nifty100 = yf.download('^NSEI', start='2023-06-01', end='2023-07-15')
prices = nifty100['Close'].values
# Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
# Reinforcement Learning (DQN) Model
class TradingEnvironment(gym.Env):
def __init__(self, data):
self.data = data
self.current_step = 0
self.total_steps = len(data) - 1
self.position = 0 # -1: short, 0: neutral, 1: long
self.cash = 10000 # initial cash
self.portfolio_value = 10000
def step(self, action):
prev_value = self.portfolio_value
# Take action: Buy (1), Sell (-1), or Hold (0)
if action == 1:
self.position = 1
elif action == -1:
self.position = -1
self.current_step += 1
price = self.data[self.current_step]
# Calculate new portfolio value
self.portfolio_value = self.cash + self.position * price
reward = self.portfolio_value - prev_value
done = self.current_step == self.total_steps
return np.array([price]), reward, done, {}
def reset(self):
self.current_step = 0
self.position = 0
self.portfolio_value = 10000
return np.array([self.data[self.current_step]])
# Define LSTM Model
def create_lstm_model(input_shape):
model = Sequential()
model.add(LSTM(50, return_sequences=True,
input_shape=input_shape))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
return model
# ARIMA Model
arima_model = ARIMA(prices, order=(5, 1, 0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=10)
# Continue with training, backtesting, and performance evaluation...
