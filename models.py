import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

def demo_simple_dense(dims=None):
    assert len(dims)==2, 'only give input and output dimensions'
    optimizer = optimizers.SGD(lr=0.1, clipnorm=1.)

    model = Sequential()
    model.add(Dense(units=dims[1], activation='linear', input_dim=dims[0]))
    model.compile(optimizer=optimizer, loss='mse')
    return model
