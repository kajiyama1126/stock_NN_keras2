import copy
import os

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam


def read_stock_data(days):
    basename = ['start', 'high', 'low', 'last']
    column_name = copy.copy(basename)
    for i in range(days - 1):
        for j in basename:
            column_name.append(j + str(i + 1))

    column_name.pop(0)
    teacher_name = ['teacher_only_up_down']

    place = '/Users/kajiyama/PycharmProjects/stock_NN/stock_data/connect'
    directory = os.listdir(place)

    stock_data = np.empty((0, 4 * days - 1), float)
    teacher_data = np.empty((0, 1), int)
    for name in directory:
        print(name)
        os.chdir(place)
        data = pd.read_csv(name, encoding='cp932', index_col=0)
        index_name = data.index
        stock_data_onedata = np.empty((0, 4 * days - 1), float)
        teacher_data_onedata = np.empty((0, 1), int)
        for name1 in index_name:
            stock_data_tmp = np.array([data.ix[name1, column_name]])
            teacher_data_tmp_tmp = data.ix[name1, teacher_name]
            if teacher_data_tmp_tmp[0] > 0:
                tmp = int(1)
            else:
                tmp = int(0)

            teacher_data_tmp = np.array([[tmp]])
            # print(stock_data_tmp)
            stock_data_onedata = np.append(stock_data_onedata, stock_data_tmp, axis=0)
            teacher_data_onedata = np.append(teacher_data_onedata, teacher_data_tmp, axis=0)

        stock_data = np.append(stock_data, stock_data_onedata, axis=0)
        teacher_data = np.append(teacher_data, teacher_data_onedata, axis=0)
        # print(stock_data)
    np.savetxt('x_train_test.csv', stock_data, delimiter=',')
    np.savetxt('y_train_test.csv', teacher_data, delimiter=',')

    return stock_data, teacher_data


def load_stock_data():
    place = '/Users/kajiyama/PycharmProjects/stock_NN/stock_data/connect'
    os.chdir(place)
    stock_data = np.loadtxt('x_train_test.csv', delimiter=',')
    teacher_data = np.loadtxt('y_train_test.csv', delimiter=',')

    return stock_data, teacher_data


days = 10
# x_train,y_train = read_stock_data(days)
x, y = load_stock_data()
x_train, x_test = np.split(x, [int(x.size * 0.8)])
y_train, y_test = np.split(y, [int(x.size * 0.8)])

model = Sequential()
model.add(Dense(100, input_dim=4 * days - 1))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.3))
# model.add(Dense(2))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(2))
# model.add(Activation('relu'))
# model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=5, shuffle=True,
          batch_size=16)
score = model.evaluate(x_test, y_test, batch_size=16)
print(score)
