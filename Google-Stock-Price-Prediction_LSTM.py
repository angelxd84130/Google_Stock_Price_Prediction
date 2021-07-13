import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# load dataset
df = pd.read_csv('Google_Stock_Price_Train.csv')
# get open value from the dataset
df = df.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
df = sc.fit_transform(df)

# set training period (days)
peroid = 60
# split data to make a new training data (use past [period]days to predict the next day price)
Xtrain = []
Ytrain = []
for i in range(60, len(df)-1):
    Xtrain.append(df[i-60: i, 0])
    Ytrain.append(df[i, 0])

# transform input from 2D to 3D (the input for LSTM model must be 3D)
Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xtrain.reshape((len(Xtrain[0], len(Xtrain[1], 1))))

# Build model
model = Sequential()
# first layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(len(Xtrain[1], 1))))
model.add(Dropout(0.2))
# second layer
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout)
# output layer
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(Xtrain, Ytrain, epochs=100, batch_size=32)

# prepare testing data


# predict and check the accuracy

