import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

def demo_simple_dense(train_X, train_y):
    # layer_dims: number of nodes in input, hidden and output layers (excluding input layer)
    # activations: activation functions for hidden and output layers

    optimizer = optimizers.SGD(lr=0.1, clipnorm=1.)

    model = Sequential()
    model.add(Dense(units=1, activation='linear', input_dim=train_X.shape[1]))
    model.compile(optimizer=optimizer, loss='mse')

    print('training')
    model.fit(train_X, train_y, epochs=100)
    return model
