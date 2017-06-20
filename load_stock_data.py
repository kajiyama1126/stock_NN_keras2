import copy
import datetime
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.layers import Dense, Dropout, Activation, normalization
from keras.models import Sequential
from keras.optimizers import Adam
from scipy.stats import zscore
from class_load_data import load_data,load_date_separate

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

data_place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
days = 15
start = 0
end = 100000
code = '6503-T'

stock_data = load_date_separate(data_place,days,start,end,code)
# def read_stock_data(days):
#     i = 0
#     basename = ['start', 'high', 'low', 'last']
#     column_name = copy.copy(basename)
#     for i in range(days - 1):
#         for j in basename:
#             column_name.append(j + str(i + 1))
#
#     column_name.pop(0)
#     teacher_name = ['teacher_only_up_down']
#
#     place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
#     directory = os.listdir(place)
#
#     stock_data = np.empty((0, 4 * days - 1), float)
#     teacher_data = np.empty((0, 1), int)
#     for name in directory:
#         print(name)
#         os.chdir(place)
#         data = pd.read_csv(name, encoding='cp932', index_col=0)
#         index_name = data.index
#         stock_data_onedata = np.empty((0, 4 * days - 1), float)
#         teacher_data_onedata = np.empty((0, 1), int)
#         for name1 in index_name:
#             stock_data_tmp = np.array([data.ix[name1, column_name]])
#             teacher_data_tmp_tmp = data.ix[name1, teacher_name]
#             if teacher_data_tmp_tmp[0] > 0:
#                 tmp = int(1)
#             else:
#                 tmp = int(0)
#
#             teacher_data_tmp = np.array([[tmp]])
#             # print(stock_data_tmp)
#             stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
#             teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)
#
#         if name == '2010-06-30column_renamefor15day_reformconnect.csv':
#             break
#
#         stock_data = np.append(stock_data, stock_data_onedata, axis=0)
#         teacher_data = np.append(teacher_data, teacher_data_onedata, axis=0)
#         # print(stock_data)
#
#     np.savetxt('x_train_test.csv', stock_data, delimiter=',')
#     np.savetxt('y_train_test.csv', teacher_data, delimiter=',')
#
#     return stock_data, teacher_data
#
#
# def read_stock_data_normalization(days):
#     i=0
#     basename = ['start', 'high', 'low', 'last']
#     column_name = copy.copy(basename)
#     for i in range(days - 1):
#         for j in basename:
#             column_name.append(j + str(i + 1))
#
#     teacher_name = ['teacher_only_up_down']
#
#     place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
#     directory = os.listdir(place)
#
#     stock_data = np.empty((0, 4 * days), float)
#     teacher_data = np.empty((0, 1), int)
#
#     for name in directory:
#         i += 1
#         if i<300:
#             pass
#         else:
#             print(name)
#             os.chdir(place)
#             data = pd.read_csv(name, encoding='cp932', index_col=0)
#             index_name = data.index
#             stock_data_onedata = np.empty((0, 4 * days), float)
#             teacher_data_onedata = np.empty((0, 1), int)
#             for name1 in index_name:
#                 stock_data_normalization = zscore(data.ix[name1, column_name])
#                 stock_data_tmp = np.array([stock_data_normalization])
#                 teacher_data_tmp_tmp = data.ix[name1, teacher_name]
#                 if teacher_data_tmp_tmp[0] > 0:
#                     tmp = int(1)
#                 else:
#                     tmp = int(0)
#
#                 teacher_data_tmp = np.array([[tmp]])
#                 # print(stock_data_tmp)
#                 stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
#                 teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)
#
#             # if name == '2010-06-30column_renamefor15day_reformconnect.csv':
#             #     break
#             if i>= 400:
#                 break
#             #
#             stock_data = np.append(stock_data, stock_data_onedata, axis=0)
#             teacher_data = np.append(teacher_data, teacher_data_onedata, axis=0)
#             # print(stock_data)
#     today = datetime.datetime.today()
#     todayname = str(today.strftime("%Y%m%d_%H%M%S"))
#     np.savetxt('x_train_test_norm' + todayname +'.csv', stock_data, delimiter=',')
#     np.savetxt('y_train_test_norm' + todayname +'.csv', teacher_data, delimiter=',')
#
#     return stock_data, teacher_data
#
#
# def load_stock_data():
#     place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
#     os.chdir(place)
#     stock_data = np.loadtxt('x_train_test.csv', delimiter=',')
#     teacher_data = np.loadtxt('y_train_test.csv', delimiter=',')
#
#     return stock_data, teacher_data
#
#
# def load_stock_data_normalization():
#     place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
#     os.chdir(place)
#     stock_data = np.loadtxt('x_train_test_norm.csv', delimiter=',')
#     teacher_data = np.loadtxt('y_train_test_norm.csv', delimiter=',')
#
#     return stock_data, teacher_data
#
#
# def test_data(file_name, days):
#     basename = ['start', 'high', 'low', 'last']
#     column_name = copy.copy(basename)
#     for i in range(days - 1):
#         for j in basename:
#             column_name.append(j + str(i + 1))
#
#     column_name.pop(0)
#     place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
#     os.chdir(place)
#     directory = os.listdir(place)
#     if os.path.isfile(file_name):
#         data = pd.read_csv(file_name, encoding='cp932', index_col=0)
#     else:
#         return None
#
#     teacher_name = ['teacher_only_up_down']
#     stock_data_onedata = np.empty((0, 4 * days - 1), float)
#     teacher_onedata = np.empty((0, 1), int)
#     index_name = data.index
#     for name1 in index_name:
#         stock_data_tmp = np.array([data.ix[name1, column_name]])
#         teacher_data_tmp_tmp = data.ix[name1, teacher_name]
#         if teacher_data_tmp_tmp[0] > 0:
#             tmp = int(1)
#         else:
#             tmp = int(0)
#
#         teacher_data_tmp = np.array([[tmp]])
#         stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
#         teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)
#
#     return stock_data_onedata, teacher_data_onedata


today = datetime.datetime.today()
todayname = str(today.strftime("%Y%m%d_%H%M%S"))

days = 15
x,y = stock_data.read_stock_data()
# x,y= read_stock_data(days)
# x,y = read_stock_data_normalization(days)
# x, y = load_stock_data()
# x, y = load_stock_data_normalization()
# x_train, x_test = np.split(x, [int(len(x)* 0.8)])
# print(x.size)
x_train, y_train = x, y
# y_train, y_test = np.split(y, [int(len(y) * 0.8)])

# print(x_test,y_test)

model = Sequential()
model.add(Dense(2000, input_dim=4 * days))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2000))
model.add(Activation('relu'))
model.add(normalization.BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(normalization.BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(500))
# model.add(normalization.BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, shuffle=True, verbose=2,
          batch_size=16)

os.chdir('D:\Pycharm Project\stock_NN_keras2')
model.save('testmodel_norm' + todayname + '.h5')

# y_pre = model.predict(x_test,batch_size=1)
# score = model.evaluate(x_test, y_test)
# print(score[0],score[1])
