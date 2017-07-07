# from keras.models import Sequential
# from keras.layers import Dense, Activation
#
# model = Sequential([
#     Dense(32, input_dim=784),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax'),
# ])


import datetime
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from NN_model import Model1,Model2
from class_load_data import load_data,load_date_separate,load_data_separate_dekidaka,LSTM_load_data_separate_dekidaka

# config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7))
# session = tf.Session(config=config)
# tensorflow_backend.set_session(session)

data_place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
data_place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15daysvwap'
days = 15
start = 0
end = 3
code = ['8306-T','8316-T','8411-T']

# stock_data = load_date_separate(data_place,days,start,end,code)
# stock_data = load_data_separate_dekidaka(data_place,days,start,end,code)
stock_data = LSTM_load_data_separate_dekidaka(data_place,days,start,end,code)
x, y = stock_data.read_stock_data()