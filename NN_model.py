from keras.layers import Dense, Dropout, Activation,LSTM,GRU
from keras.models import Sequential,load_model
from keras.optimizers import Adam,RMSprop,Adadelta,Adagrad
from keras.constraints import non_neg,max_norm
import keras.backend as K
import keras.models

class Model1(object):
    def __init__(self, input_dim,days):
        self.input_dim = input_dim
        self.days = days

        self.mymodel = self.model_make()
        self.model_compile()



    def model_make(self):
        model = Sequential()
        model.add(Dense(1000, input_dim=self.input_dim*self.days))
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

    def load(self,name):
        self.mymodel = load_model(name)
    # def precision

    def model_compile(self):
        self.mymodel.compile(loss='binary_crossentropy',
                             optimizer=Adam(),
                             metrics=['accuracy',])
    def predict(self,x):
        return self.mymodel.predict(x,batch_size=1)



class Model2(Model1):
    def model_make(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.input_dim*self.days))
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

class Model_lstm(Model1):
    def model_make(self):
        Hidden_size = 32

        model = Sequential()
        model.add(LSTM(32, input_shape=(self.days,self.input_dim), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16))
        model.add(Dropout(0.2))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(64))
        # model.add(Activation('relu'))
        # # model.add(normalization.BatchNormalization())
        # model.add(Dropout(0.5))
        # model.add(Dense(32))
        # model.add(Activation('relu'))
        # # model.add(normalization.BatchNormalization())
        # model.add(Dropout(0.5))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # # model.add(normalization.BatchNormalization())
        # model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        return model

    def model_compile(self):
        self.mymodel.compile(loss='binary_crossentropy',
                             optimizer = RMSprop(),
                             metrics=['accuracy'])

    def fit(self, x_train, y_train, epochs,batch_size,validation=None):
        hist = self.mymodel.fit(x_train, y_train, epochs=epochs, validation_data = validation, verbose=1,
                                batch_size=batch_size,shuffle = True)
        return hist

class Model_GRU(Model_lstm):
    def model_make(self):

        model = Sequential()
        model.add(LSTM(128, input_shape=(self.days,self.input_dim), return_sequences=False,dropout=0.0))
        # model.add(Dropout(0.2))
        # model.add(GRU(64,return_sequences=True))
        # model.add(GRU(16))
        # model.add(Dropout(0.2))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(64))
        # model.add(Activation('relu'))
        # # model.add(normalization.BatchNormalization())
        # model.add(Dropout(0.5))
        # model.add(Dense(32))
        # model.add(Activation('relu'))
        # # model.add(normalization.BatchNormalization())
        # model.add(Dropout(0.5))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # # model.add(normalization.BatchNormalization())
        # model.add(Dropout(0.5))
        model.add(Dense(64))
        # model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(64))
        # model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(32))
        # model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        return model

    def model_compile(self):
        self.mymodel.compile(loss='binary_crossentropy',
                             optimizer = Adam(),
                             metrics=['accuracy'])

class Model_GRU_const(Model_GRU):
    def model_make(self):

        model = Sequential()
        model.add(GRU(128, input_shape=(self.days,self.input_dim), return_sequences=True,dropout=0.2,implementation=2))
        model.add(Activation('relu'))
        model.add(GRU(128, return_sequences=False,implementation=2))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        # model.add(Dense(128))
        # model.add(Dropout(0.5))
        model.add(Dense(1,use_bias=False))
        model.add(Activation('sigmoid'))
        model.summary()
        return model
