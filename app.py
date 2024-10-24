import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler



model = load_model('D:\Stock_market\Stock Prediction Model.keras')

st.header('Stock Market Predictor')
stock=st.text_input('Enter the stock Symbol')
start  = '2014-01-01'
end  = '2024-12-31'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test],ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Prics vs MovingAverage_50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Prics vs MovingAverage_50 vs MovingAverage_100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'b')
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Prics vs MovingAverage_50 vs MovingAverage_100 vs MovingAverage_200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'b')
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'y')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x ,y = np.array(x) , np.array(y)

predict  = model.predict(x)
scale = 1/scaler.scale_
predict = predict*scale
y = y*scale

st.subheader('Original Prics vs Predicted Price')

fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r',label = 'Original Prics')
plt.plot(y,'g',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
