import copy
import os

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from scipy.stats import zscore

import tensorflow as tf
from keras.backend import tensorflow_backend
from NN_model import Model_GRU,Model_GRU_const
from class_load_data import LSTM_load_data_separate_dekidaka2

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.7))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

def make_topix_core30():
    code = []
    topix = [2914, 3382, 4063, 4502, 4503, 6501, 6752, 6758, 6861, 6902, 6954, 6981, 7201, 7203, 7267, 7751, 8031, 8058, 8306, 8316, 8411, 8766, 8801, 8802, 9020, 9022, 9432, 9433, 9437, 9984]
    for i in topix:
        code.append(str(i)+'-T')
    return code

data_place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for30daysvwap'
days = 30
start = 1200
end = 1500
code = ['8306-T', '6501-T','7203-T']
code_topix_core30 = make_topix_core30()
code = code_topix_core30
model = Model_GRU_const(6,days)
os.chdir('D:\Pycharm Project\stock_NN_keras2')
os.chdir('20170707_110132')
model.load('testmodel_norm20170707_112825.h5')
correct = [0,0]
for i in range(start,end):
    stock_data = LSTM_load_data_separate_dekidaka2(data_place, days,i, i+1, code)
    x, y = stock_data.read_stock_data()
    pre=model.predict(x)
    # print(model.predict(x))
    print(y[np.argmax(pre[0])],pre[0][np.argmax(pre[0])],np.max(pre[0]))
    if pre[0][np.argmax(pre[0])]>0:
        correct[0] += y[np.argmax(pre[0])][0]
        correct[1] += 1
print(correct)