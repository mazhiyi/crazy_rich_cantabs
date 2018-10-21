import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras import optimizers

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def demo_simple_dense(dims=None):
    assert len(dims)==2, 'only give input and output dimensions'
    optimizer = optimizers.SGD(lr=0.1, clipnorm=1.)

    model = Sequential()
    model.add(Dense(units=dims[1], activation='linear', input_dim=dims[0]))
    model.compile(optimizer=optimizer, loss='mse')
    return model

# def event_random_forest_model(dims=None):
#     X, y = make_classification(n_samples=1000, n_features=4,
#                                n_informative=2, n_redundant=0,
#                                random_state=0, shuffle=False)
#     model = RandomForestClassifier(n_estimators=100, max_depth=2,
#                                  random_state=0)

#     return model 

def event_conv_model(dims=None):
    optimizer = optimizers.SGD(lr=0.01, clipnorm=1.)

    model = Sequential()
    model.add(Conv1D(17, 5, activation='relu', input_shape=(dims[0], 1)))
    model.add(Flatten())
    model.add(Dense(units=dims[1], activation='linear'))

    model.compile(optimizer=optimizer, loss='mse')
    print(model.summary())

    return model