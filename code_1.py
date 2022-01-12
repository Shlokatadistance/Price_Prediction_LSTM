import pandas as pd
import numpy as np
import requests
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
key = '0D6HAHFDZPJLAL0E'

#using Forex pairs now

ticker = 'USDEUR'
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='+ticker+'&outputsize=full&apikey='+key
urlData = requests.get(url).json()['Time Series (Daily)']
Price_df = pd.DataFrame(urlData).T
Price_df = Price_df.drop(columns=['7. dividend amount', '8. split coefficient'])
Price_df = Price_df.astype(float)
Price_df = Price_df.iloc[::-1]
Price_df.reset_index(inplace=True)
Price_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
Price_df['Date'] = pd.to_datetime(Price_df['Date'])

df = Price_df
print('Number of rows and columns:', df.shape)


# it is recommended that the model gets more recent datapoints to learn current trends of price
#train_datapoints = 1700 ##sets the number of training datapoints; the more datapoints, the more data that the model #get to train
datapoints_input = 60 ##number of datapoints in one input data

training_set = df.iloc[:train_datapoints, 4:5].values ##800 values
test_set = df.iloc[train_datapoints:, 4:5].values

print(training_set.shape)
print(test_set.shape)


sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set) ##Transform the training set

# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []

for i in range(datapoints_input, train_datapoints): ##append 60 x datapoints and 1 y datapoint
    X_train.append(training_set_scaled[i-datapoints_input:i, 0]) ##D-60 to D-0 data
    y_train.append(training_set_scaled[i, 0]) ##resulting D+1 price data

X_train, y_train = np.array(X_train), np.array(y_train) ##X becomes set of arrays and Y becomes set of resulting values
print(X_train.shape) ##row - number of day series, column - number of datapoints (60)
print(X_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) ##reshape X data into 3d manner; 60 lists of one value times the number of day series
print(X_train)
print(X_train.shape)




model = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) ## (60, 1)
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1,activation='relu'))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32) 
#X train is the list of (60, 1) datasets, while y_train is one array containing the results



dataset_train = df.iloc[:train_datapoints, 4:5] ##period for train dataset; 800
dataset_test = df.iloc[train_datapoints:, 4:5] ##period after train dataset; 1806
print(dataset_train.shape)
print(dataset_test.shape)

dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
print(dataset_total.shape)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - datapoints_input:].values ##2606 - 606 - 60 = 1940; includes train data

print(inputs)
print(inputs.shape)

print('then...')

inputs = inputs.reshape(-1,1)

print(inputs)
print(inputs.shape)


inputs = sc.transform(inputs) ##transform inputs with minmax scale

X_test = []

for i in range(datapoints_input, len(dataset_test) + datapoints_input): ##compile input data; 
    X_test.append(inputs[i-datapoints_input:i, 0])

X_test = np.array(X_test) ##make as an array

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) ##reshape input data in the same way we did when fitting the model

print(X_test.shape)


predicted_stock_price = model.predict(X_test) ##predict y values based on x values acquired above

predicted_stock_price = sc.inverse_transform(predicted_stock_price) ##decode minmax scaled predicted FX price

print(predicted_stock_price)

print(predicted_stock_price.shape) ##print shape of predicted fx price
print(dataset_test.values.shape) ##print shapes of actual fx price

updown_prediction = pd.DataFrame({"Predicted":predicted_stock_price.reshape(1,len(dataset_test))[0]})
print(updown_prediction)

updown_actual = pd.DataFrame({"Actual":dataset_test.values.reshape(1,len(dataset_test))[0]})
print(updown_actual)

results = []

for i in range(0, len(dataset_test)-1):
  predicted_change = (updown_prediction['Predicted'][i+1] / updown_actual['Actual'][i]) - 1.0
  actual_change = (updown_actual['Actual'][i+1] / updown_actual['Actual'][i]) - 1.0

  if np.sign(predicted_change) == np.sign(actual_change):
    results.append(True)
  else:
    results.append(False)

results.count(True)
results.count(False)

print(results)
print('accuracy: ' + str(results.count(True)/len(results)))