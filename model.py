import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import MaxPooling1D, Conv1D
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.optimizers import Adam, SGD
from keras.layers import Flatten, RepeatVector
from achieve_params import Flags

def base_model(trainX, trainY, input_dim=1, output_dim=7, type='easy'):
    """
    output = activation(BN(Wx+b))
    :param trainX:
    :param trainY:
    :param input_dim:
    :param output_dim:
    :param type:
    :return:
    """
    keras.backend.clear_session()
    if type == 'easy':
        model = Sequential()
        model.add(LSTM(units=Flags.unit, input_dim=input_dim))
        model.add(BatchNormalization(momentum=0.99))
        model.add(Activation('linear'))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer=Adam(lr=0.002, beta_1=0.9))
        history = model.fit(trainX, trainY, nb_epoch=Flags.epoch, batch_size=Flags.batch_size,
                            verbose=1, validation_split=Flags.validation_split)
    elif type == 'hard':
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=1, activation='linear', input_shape=(1, 7)))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        model.add(RepeatVector(1))
        model.add(LSTM(Flags.unit, activation='linear', return_sequences=True))
        model.add(LSTM(50, activation='linear', return_sequences=False))
        model.add(Dropout(Flags.dropout))
        model.add(Flatten())
        model.add(Dense(output_dim))
        model.add(BatchNormalization(momentum=0.99))
        model.compile(loss='mae', optimizer=Adam(lr=0.002, beta_1=0.9))
        history = model.fit(trainX, trainY, nb_epoch=Flags.epoch, batch_size=Flags.batch_size,
                            verbose=1, validation_split=Flags.validation_split)
    return model, history
