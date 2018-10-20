import datasets
import models

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

from utils import expand_for_conv

# An example training
# 

# Define dataset and model architecture
# train_X, train_y, test_X, test_y = datasets.get_demo_simple_dense_data()
# model = models.demo_simple_dense([2, 1])
def main():
    print('process training and testing data')
    X, y = datasets.get_event_data('MRQ', 2018, 1)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

    train_X = expand_for_conv(train_X)
    test_X = expand_for_conv(test_X)
    model = models.event_conv_model([train_X.shape[1], 1])

    # Train the model 
    print('training')    
    model.fit(train_X, train_y, epochs=3)

    # Evaluate the model 
    print('evaluation loss')
    print(model.evaluate(test_X, test_y))
    predicted_y = model.predict(test_X)

    print('saving model')
    model.save('model.h5')
    
    plt.figure()
    plt.plot(test_y.values)
    plt.plot(predicted_y)
    plt.legend(['test', 'predicted'])
    plt.show()

if __name__ == "__main__":
    main()
