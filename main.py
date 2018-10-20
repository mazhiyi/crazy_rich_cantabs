import datasets
import models

import numpy as np
from sklearn.model_selection import train_test_split

from utils import expand_for_conv

# An example training
# 

# Define dataset and model architecture
train_X, train_y, test_X, test_y = datasets.get_demo_simple_dense_data()
# model = models.demo_simple_dense([2, 1])
# 
train_X = expand_for_conv(train_X.iloc[0:1])
test_X = expand_for_conv(test_X.iloc[0:1])
train_y = train_y.iloc[0:1]
test_y = test_y.iloc[0:1]
model = models.event_conv_model([2, 1])
# Train the model 
print('training')
model.fit(train_X, train_y, epochs=50)

# Evaluate the model 
print('evaluation loss')
print(model.evaluate(test_X, test_y))
print('predicted y:', model.predict(test_X))
print('actual y: ', test_y)

# print('let\'s test geting event X')
# print(datasets.get_event_data('', 2017, 1))
