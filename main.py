import dataset
import models

# An example training
# 

# Only need to change the functions for the following two lines 
# to train a different model using different dataset
train_X, train_y, test_X, test_y = dataset.get_demo_simple_dense_data()
model = models.demo_simple_dense(train_X, train_y)


print('evaluation loss')
print(model.evaluate(test_X, test_y))
