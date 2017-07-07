import datetime
import os

import tensorflow as tf
from keras.backend import tensorflow_backend
import numpy as np
from NN_model import Model2,Model_lstm,Model_GRU,Model_GRU_const
from class_load_data import LSTM_load_data_separate_dekidaka,LSTM_load_data_separate_dekidaka2

def make_topix_core30():
    code = []
    topix = [2914, 3382, 4063, 4502, 4503, 6501, 6752, 6758, 6861, 6902, 6954, 6981, 7201, 7203, 7267, 7751, 8031, 8058, 8306, 8316, 8411, 8766, 8801, 8802, 9020, 9022, 9432, 9433, 9437, 9984]
    for i in topix:
        code.append(str(i)+'-T')
    return code

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

data_place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
data_place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15daysvwap'
data_place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for30daysvwap'
days = 30
start = 0
end = 1200
code = ['8306-T', '6501-T','7203-T']
code_topix_core30 = make_topix_core30()
code = code_topix_core30

# stock_data = load_date_separate(data_place,days,start,end,code)
# stock_data = load_data_separate_dekidaka(data_place,days,start,end,code)
stock_data = LSTM_load_data_separate_dekidaka2(data_place, days, start, end, code)
# stock_data = load_data(data_place, days, start, end)
# stock_data = load_data_add_dekidaka(data_place,days,start,end)


today = datetime.datetime.today()
todayname = str(today.strftime("%Y%m%d_%H%M%S"))

os.chdir(data_place)
# x, y = stock_data.read_stock_data()
# np.savez('x_core30_for1200'+'.npz',x)
# np.savez('y_core30_for1200'+'.npz',y)
x= np.load('x_core30_for1200'+'.npz')['arr_0']
y= np.load('y_core30_for1200'+'.npz')['arr_0']
# x,y= read_stock_data(days)
# x,y = read_stock_data_normalization(days)
# x, y = load_stock_data()
# x, y = load_stock_data_normalization()

# print(x.size)
batch_size = 128
# x_train, y_train = x[len(x) % batch_size:], y[ len(y) % batch_size: ]
x_train, x_val = np.split(x, [int(len(x)* 0.9)])
y_train, y_val = np.split(y, [int(len(y) * 0.9)])

x_train = x_train[len(x_train) % batch_size:]
y_train = y_train[len(y_train) % batch_size:]
x_val = x_val[len(x_val) % batch_size:]
y_val = y_val[len(y_val) % batch_size:]
# print(x_test,y_test)

model = Model_GRU_const(6,days)
os.chdir('D:\Pycharm Project\stock_NN_keras2')
# os.chdir('20170707_003614')
# model.load('testmodel_norm20170707_011717.h5')
os.chdir('D:\Pycharm Project\stock_NN_keras2')
os.mkdir(todayname)
for i in range(10):
    print(str(i)+'回目')
    hist = model.fit(x_train, y_train, epochs=10,batch_size=batch_size,validation=(x_val,y_val))
    os.chdir('D:\Pycharm Project\stock_NN_keras2')
    os.chdir(todayname)
    now = datetime.datetime.today()
    now_name = str(now.strftime("%Y%m%d_%H%M%S"))
    model.save('testmodel_norm' + now_name + '.h5')

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
#
# epochs = len(loss)
# plt.plot(range(epochs), loss, marker='.', label='acc')
# plt.plot(range(epochs), val_loss, marker='.', label='val_acc')
# plt.legend(loc='best')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.show()

# y_pre = model.predict(x_test,batch_size=1)

# score = model.evaluate(x_test, y_test)
# print(score[0],score[1])
