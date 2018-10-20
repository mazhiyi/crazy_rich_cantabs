from keras import optimizers

import dataset
import models

# An example training
train_X, train_y = dataset.get_demo_simple_dense_data()
optimizer = optimizers.SGD(lr=0.1, clipnorm=1.)
model = models.demo_simple_dense(layer_dims=[train_X.shape[1], 1], activations=['linear'], optimizer=optimizer)
print('training')
model.fit(train_X, train_y, epochs=100)
print('evaluation loss')
print(model.evaluate(train_X, train_y))
