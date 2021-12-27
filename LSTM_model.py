import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model
import streamlit as st

'''
 RUN THIS LSTM MODEL ONCE SO IT CREATES THE keras_model.h5 FILE THAT app.py CAN LOAD
'''

start = '2010-01-01'
end = '2021-12-20'
yf.pdr_override()            # needed for the yahoo finance API

# Web scrap the data from yahoo for Apple stock and create a DataFrame
df = pdr.get_data_yahoo('AAPL', start="2010-01-01", end="2021-12-01")

# Check if the data was imported correctly
# print(df.head())
# print(df.tail())
# since data's index is defaulted to the date, we can reset the index and index properly moving date to column 1 (column indexing 0)
df = df.reset_index()
# then drop date and adj close columns
df.drop(['Date', 'Adj Close'], axis=1)
# check again
# print(df.head())
# plot the closing values
# plt.plot(df.Close)

# Visualization
st.subheader('Closing Price vs. Time')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)

# Moving avg 100 days
ma100 = df.Close.rolling(100).mean()
st.subheader('Closing Price vs. Time with 100 MA ')
plt.figure(figsize=(12, 6))
plt.xlabel('years')
plt.ylabel('price')
plt.plot(df.Close)
plt.plot(ma100, 'r')


# Moving avg 200 days
ma200 = df.Close.rolling(200).mean()
st.subheader('Closing Price vs. Time with 100MA & 200MA')
plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')


# Split the data into Test and Training sets
data_train = pd.DataFrame(df['Close'][:int(len(df)*0.7)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.7):])

# Convert the data --- Scaled down the data to be between 0 and 1 ... that's how the LSTM model receives the data
scaler = MinMaxScaler(feature_range=(0, 1))

data_train_array = scaler.fit_transform(data_train)

X_train = []
y_train = []
# Time series analogy of the upcoming data being dependent on a certain # of days before the predicted day
for i in range(100, data_train_array.shape[0]):
    X_train.append(data_train_array[i-100:i])  # previous 100 days of data will be the training data
    y_train.append(data_train_array[i, 0])     # y_train will have the data that's going to be predicted for the 101th day

# Convert the above to Numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Machine Learning Model (LSTM)
model = Sequential()
# Layer 1
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Layer 2
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
# Layer 3
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
# Layer 4
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
# Combine all layers
model.add(Dense(units=1)) # as dense as 1 column (which is our Close)

# print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50)  # let the model train for 50 epochs
model.save('keras_model.h5')  # save the data

past_100_days = data_train.tail(100)
final_df = past_100_days.append(data_test, ignore_index=True)

# Scale down again
input_data = scaler.fit_transform(final_df)

# Define the tests  ---- Similarly as above
X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Prediction
predicted = model.predict(X_test)

# Find out the factor by which the data was scaled down
# print(scaler.scale_)

# Divide the unscaled data by scaled factor
scale_factor = 1/0.02099517
predicted = predicted * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()



