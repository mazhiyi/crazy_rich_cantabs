import datasets
import models
from sklearn.model_selection import train_test_split

# An example training
# 

# Define dataset and model architecture
train_X, train_y, test_X, test_y = datasets.get_demo_simple_dense_data()
model = models.demo_simple_dense([2, 1])

# Train the model 
print('training')
model.fit(train_X, train_y, epochs=100)

# Evaluate the model 
print('evaluation loss')
print(model.evaluate(test_X, test_y))

print('let\'s test geting event X')
print(datasets.get_event_data('', 2017, 1))
