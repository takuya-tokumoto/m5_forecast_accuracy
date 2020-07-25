import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import argparse
import json
import numpy as np

from utils import load_datasets, load_target
from logs.logger import log_best
from models.lgbm import train_and_predict


timesteps = 20
startDay = 0
TrTestWin = 56


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now))

feats = config['features']
logging.debug(feats)

# target_name = config['target_name']


feats_train, feats_test = load_datasets(feats)
y_train_all = load_target()


lr_Train = pd.concat([y_train_all, feats_train], axis = 1)

lr_Train.head()

sc = MinMaxScaler(feature_range = (0, 1))

lr_Train_scaled = sc.fit_transform(lr_Train)


X_Train = []
y_Train = []


for i in range(timesteps, 1913 - startDay):
    X_Train.append(lr_Train_scaled[i-timesteps:i]) #i = 14の場合、[0:14], i = 15の場合、[1:15]
    y_Train.append(lr_Train_scaled[i][0:30490])    #i = 14の場合、[14][0:30490], i = 15の場合、[15][0:30490]

    
X_Train = np.array(X_Train)
y_Train = np.array(y_Train)
    
    
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
layer_1_units=40
regressor.add(LSTM(units = layer_1_units, return_sequences = True, input_shape = (X_Train.shape[1], X_Train.shape[2])))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
layer_2_units=300
regressor.add(LSTM(units = layer_2_units, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
layer_3_units=300
regressor.add(LSTM(units = layer_3_units))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 30490))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
epoch_no=32
batch_size_RNN=44
regressor.fit(X_Train, y_Train, epochs = epoch_no, batch_size = batch_size_RNN)


inputs = lr_Train[-timesteps:]
inputs = sc.transform(inputs)

FeatureTest = pd.concat([daysBeforeEventTest, DateFlagTest], axis = 1)


X_Test = []
X_Test.append(inputs[0:timesteps])
X_Test = np.array(X_Test)
predictions = []

# for j in range(timesteps,timesteps + 28):  # range(14,42)
for j in range(timesteps,timesteps + 28):  # range(14,42)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_Test[0,j - timesteps:j].reshape(1, timesteps, X_Train.shape[2]))
#   .reshape(1, timesteps, ~)) ~はX_trainの行と一致させること。特徴量増やしたら変更必要。
    
    testInput = np.column_stack((np.array(predicted_stock_price), np.array(FeatureTest)[j - timesteps].reshape(1, X_Train.shape[2] - 30490))) #特徴量変更後注意
#   j = 14の場合、..(X_test[0,0:14].reshape(1, 14, 30494)) j = 15の場合、..(X_test[0,1:15].reshape(1, 14, 30494)) 
#   testInput = np.column_stack((np.array(testInput), pd.get_dummies(DateFlagTest[category_col].astype("category"), drop_first = True)[1913 + j - timesteps]))
    
    X_Test = np.append(X_Test, testInput).reshape(1,j + 1,X_Train.shape[2]) 
    # j = 14の場合、..reshape(1, 15, 30538)) j = 15の場合、..reshape(1, 16, 30538)) 
    
    predicted_stock_price = sc.inverse_transform(testInput)[:,0:30490] # 正規化していたのを戻している
    predictions.append(predicted_stock_price)