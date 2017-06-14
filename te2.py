from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255
X_test = X_test / 255

# print(X_train[0])
# print(X_test[0])
nb_classes = 10
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
# print(y_train[0])
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(1000, input_dim=784))
model.add(Activation('relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1000))
# model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(X_train, y_train,nb_epoch = 3,shuffle=True,
           batch_size = 16)
score = model.evaluate(X_test, y_test, batch_size=16),
print(score)
