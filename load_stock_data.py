import datetime
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from NN_model import Model1,Model2
from class_load_data import load_data,load_date_separate,load_data_separate_dekidaka

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

data_place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15days'
data_place = 'D:\Pycharm Project\stock_NN\stock_data\connect_for15daysvwap'
days = 15
start = 0
end = 1000
code = ['8306-T','8316-T','8411-T']

# stock_data = load_date_separate(data_place,days,start,end,code)
stock_data = load_data_separate_dekidaka(data_place,days,start,end,code)
# stock_data = load_data(data_place, days, start, end)
# stock_data = load_data_add_dekidaka(data_place,days,start,end)


today = datetime.datetime.today()
todayname = str(today.strftime("%Y%m%d_%H%M%S"))


x, y = stock_data.read_stock_data()
# x,y= read_stock_data(days)
# x,y = read_stock_data_normalization(days)
# x, y = load_stock_data()
# x, y = load_stock_data_normalization()
# x_train, x_test = np.split(x, [int(len(x)* 0.8)])
# print(x.size)
x_train, y_train = x, y


# y_train, y_test = np.split(y, [int(len(y) * 0.8)])

# print(x_test,y_test)



model = Model2(days)
hist = model.fit(x_train,y_train,epochs=200)

os.chdir('D:\Pycharm Project\stock_NN_keras2')
model.save('testmodel_norm' + todayname + '.h5')

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
