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

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.7))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

def test_data(file_name,days):
    basename = ['start', 'high', 'low', 'last']
    column_name = copy.copy(basename)
    for i in range(days - 1):
        for j in basename:
            column_name.append(j + str(i + 1))

    column_name.pop(0)
    place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
    os.chdir(place)
    directory = os.listdir(place)
    if os.path.isfile(file_name):
        data = pd.read_csv(file_name, encoding='cp932', index_col=0)
    else:
        return None,None

    teacher_name = ['teacher_only_up_down']
    stock_data_onedata = np.empty((0, 4 * days-1 ), float)
    teacher_data_onedata = np.empty((0, 1), int)
    index_name = data.index

    for name1 in index_name:
        stock_data_tmp = np.array([data.ix[name1, column_name]])
        teacher_data_tmp_tmp = data.ix[name1, teacher_name]
        if teacher_data_tmp_tmp[0] > 0:
            tmp = int(1)
        else:
            tmp = int(0)

        teacher_data_tmp = np.array([[tmp]])
        stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
        teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)

    return stock_data_onedata,teacher_data_onedata

def test_data_separate(file_name,days,code):
    basename = ['start', 'high', 'low', 'last']
    column_name = copy.copy(basename)
    for i in range(days - 1):
        for j in basename:
            column_name.append(j + str(i + 1))

    # column_name.pop(0)
    place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
    os.chdir(place)
    directory = os.listdir(place)
    if os.path.isfile(file_name):
        data = pd.read_csv(file_name, encoding='cp932', index_col=0)
    else:
        return None,None

    teacher_name = ['teacher_only_up_down']
    stock_data_onedata = np.empty((0, 4 * days), float)
    teacher_data_onedata = np.empty((0, 1), int)
    index_name = data.index

    # for name1 in index_name:
    for stock_name in code:
        stock_data_tmp = np.array([zscore(data.ix[stock_name, column_name])])
        teacher_data_tmp_tmp = data.ix[stock_name, teacher_name]
        if teacher_data_tmp_tmp[0] > 0:
            tmp = int(1)
        else:
            tmp = int(0)

        teacher_data_tmp = np.array([[tmp]])
        stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
        teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)

    return stock_data_onedata,teacher_data_onedata

model = load_model('testmodel_norm20170628_102907.h5')
correct = 0
count = 0
code = ['8306-T','8316-T','8411-T']
for i1 in range(200):
    # os.chdir('D:\Pycharm Project\stock_NN\stock_data\connect_for15days')
    filename = os.listdir('D:\Pycharm Project\stock_NN\stock_data\connect_for15days')
    # filename = '2015-08-2'+str(i+1)+'column_renamefor15day_reformconnect.csv'
    try:
        x_test ,y_test = test_data_separate(filename[i1+1500],15,['8306-T','8316-T','8411-T'])
        os.chdir('D:\Pycharm Project\stock_NN_keras2')

        # y_pre = model.predict_classes(x_test, batch_size=1)
        # print(x_test)
        y_pre = model.predict_proba(x_test,batch_size= 1,verbose=0)
        # if y_pre[0][0] == y_test[0][0]:
        for j in range(len(code)):
            count += 1
            if abs(y_pre[0][j] - y_test[0][j])<=0.5:
                correct += 1
        print(y_pre,y_test)

        # print(np.max(y_pre), y_test[np.argmax(y_pre)],np.argmax(y_pre))
        # np.savetxt('y_pre.csv', y_pre, delimiter=',')
    except:
        pass

print(correct,correct/count)
