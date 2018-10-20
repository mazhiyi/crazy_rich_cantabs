import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras import optimizers

def demo_simple_dense(dims=None):
    assert len(dims)==2, 'only give input and output dimensions'
    optimizer = optimizers.SGD(lr=0.1, clipnorm=1.)

    model = Sequential()
    model.add(Dense(units=dims[1], activation='linear', input_dim=dims[0]))
    model.compile(optimizer=optimizer, loss='mse')
    return model

def event_random_forest_model(dims=None):
    pass

def event_conv_model(dims=None):
    optimizer = optimizers.SGD(lr=0.1)#, clipnorm=1.)

    model = Sequential()
    # model.add(Conv1D(18, 32, activation='relu', input_dim=dims[0]))
    model.add(Conv1D(18, 2, activation='relu', input_shape=(2, 1)))
    # model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(units=dims[1], activation='linear', input_dim=18))

    model.compile(optimizer=optimizer, loss='mse')

    return model