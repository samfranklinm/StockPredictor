import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import datetime, date

# '''
#  THIS APP RUNS THE LSTM MODEL TO PREDICT THE STOCK TREND
# '''
yf.pdr_override()  # needed for the yahoo finance API

# ''' Setup Streamlit '''
st.title('Stock Trend Predictor Using ML')
ticker = st.text_input('Enter Stock Ticker', autocomplete='on', placeholder='TICKER')  # Take a ticker input from user
try:
    if ticker == '':
        st.write('Blank ticker is no ticker!')
    # Web scrap the data from yahoo for Apple stock and create a DataFrame
    # start = "2010-01-01"
    # end = "2021-12-25"
    start = str(st.date_input("Select a start date"))
    end = str(st.date_input("Select an end date"))
    df = pdr.get_data_yahoo(ticker, start, end)
    if df.empty:
        st.write('Hm, either you entered an invalid stock ticker or the stock hasn\'t gone public yet!')
    else:
        # preprocess the start and end date to display the dates more legibly
        start = start.replace('-', '')
        end = end.replace('-', '')
        preprocess_dates = [start, end]
        res = []
        for i in preprocess_dates:
            date_object = i
            year = int(date_object[:4])
            month = int(date_object[4:6])
            day = int(date_object[6:])
            date = datetime(year=year, month=month, day=day)
            new_date = date.strftime("%b %d, %Y")
            res.append(new_date)
        start_date = res[0]
        end_date = res[1]

        # Describe data for the user
        st.subheader(ticker + ' from ' + start_date + ' to ' + end_date)
        st.write(df.describe())

        # ''' Visualization '''

        # Check if the range (number) of days is greater than 100
        # need to do similar processing as above to get the range (number of days)
        diff = []
        for i in preprocess_dates:
            date_object = i
            year = int(date_object[:4])
            month = int(date_object[4:6])
            day = int(date_object[6:])
            date = datetime(year=year, month=month, day=day)
            diff.append(date)
        start_day = diff[0]
        end_day = diff[1]
        num_days = end_day - start_day

        if num_days.days >= 100:
            # Moving avg 100 days
            st.subheader('Closing Price vs. Time with 100 MA ')
            fig100 = plt.figure(figsize=(12, 6))
            ma100 = df.Close.rolling(100).mean()
            plt.xlabel('Years')
            plt.ylabel('Price')
            plt.plot(df.Close)
            plt.plot(ma100)
            st.pyplot(fig100)

            # Moving avg 200 days
            ma200 = df.Close.rolling(200).mean()
            st.subheader('Closing Price vs. Time with 100MA & 200MA')
            fig200 = plt.figure(figsize=(12, 6))
            plt.xlabel('Years')
            plt.ylabel('Price')
            plt.plot(df.Close)
            plt.plot(ma100)
            plt.plot(ma200)
            st.pyplot(fig200)

            # ''' Split Data into test and train set '''
            data_train = pd.DataFrame(df['Close'][:int(len(df) * 0.7)])
            data_test = pd.DataFrame(df['Close'][int(len(df) * 0.7):])

            # Convert the data --- Scaled down the data to be between 0 and 1 ... that's how the LSTM model receives the data
            scaler = MinMaxScaler(feature_range=(0, 1))

            data_train_array = scaler.fit_transform(data_train)

            # Load the model that's already created (keras_model.h5 file) -- Ensure this is in the same working folder
            model = load_model('keras_model.h5')

            # Test set
            past_100_days = data_train.tail(100)
            final_df = past_100_days.append(data_test, ignore_index=True)

            # Scale down the test set
            input_data = scaler.fit_transform(final_df)

            # Define the tests  ---- Similarly as above
            X_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                X_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])

            # Convert to numpy array
            X_test, y_test = np.array(X_test), np.array(y_test)

            # ''' Prediction Set '''
            predicted = model.predict(X_test)

            # Find out the factor by which the data was scaled down
            scaler = scaler.scale_  # this produces an array
            # Divide the unscaled data by scaled factor
            scale_factor = 1 / scaler[0]
            predicted = predicted * scale_factor
            y_test = y_test * scale_factor

            # Original Price vs. Predicted Price
            st.subheader('Original Price vs. Predicted Price')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(predicted, 'r', label='Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig2)
        else:
            st.write('The stock predictor only works if more than 100 days are entered, the machine has to learn ya know... but here\'s the closing price for the selected range :)')
            # Closing Price
            st.subheader('Closing Prices')
            fig = plt.figure(figsize=(12, 6))
            plt.xlabel('Years')
            plt.ylabel('Price')
            plt.plot(df.Close)
            st.pyplot(fig)

except: pass
