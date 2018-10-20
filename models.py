import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

def demo_simple_dense(layer_dims=[1, 1], activations=['linear'], optimizer='sgd'):
    # layer_dims: number of nodes in input, hidden and output layers (excluding input layer)
    # activations: activation functions for hidden and output layers
    assert len(layer_dims)-1 == len(activations), 'Length of input_dims shoud match that of activations'
    model = Sequential()
    model.add(Dense(units=layer_dims[1], activation=activations[0], input_dim=layer_dims[0]))
    model.compile(optimizer=optimizer, loss='mse')
    return model

