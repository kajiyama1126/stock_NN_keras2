from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam


class Model1(object):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.mymodel = self.model_make()
        self.model_compile()


    def mdoel_make(self):
        model = Sequential()
        model.add(Dense(1000, input_dim=4 * self.input_dim))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # model.add(normalization.BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # model.add(normalization.BatchNormalization())
        model.add(Dropout(0.5))
        # model.add(Dense(500))
        # model.add(Activation('relu'))
        # model.add(normalization.BatchNormalization())
        # model.add(Dropout(0.5))
        model.add(Dense(500))
        # model.add(normalization.BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        return model

    def fit(self, x_train, y_train, epochs):
        hist = self.mymodel.fit(x_train, y_train, epochs=epochs, shuffle=True, validation_split=0.1, verbose=1,
                                batch_size=16)
        return hist

    def save(self, name):
        self.mymodel.save(name)

    def model_compile(self):
        self.mymodel.compile(loss='binary_crossentropy',
                             optimizer=Adam(),
                             metrics=['accuracy'])


class Model2(Model1):
    def model_make(self):
        model = Sequential()
        model.add(Dense(128, input_dim=6 * self.input_dim))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64))
        model.add(Activation('relu'))
        # model.add(normalization.BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(32))
        model.add(Activation('relu'))
        # model.add(normalization.BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(16))
        model.add(Activation('relu'))
        # model.add(normalization.BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        return model

    def model_compile(self):
        self.mymodel.compile(loss='binary_crossentropy',
                             optimizer=Adam(),
                             metrics=['accuracy'])
