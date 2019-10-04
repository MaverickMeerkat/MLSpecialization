import numpy as np
import matplotlib.pyplot as plt

from Course1.Week4.DNN import dnn_model as dnn
from Course1.Week4.DNN import accuracy
from Course1.Week4.dnn_utils import *


# This is a variation on the exercise - although I use numpy and modular functions, the functions are different...
# Main difference is an architecture parameter carrying a list of layers, each containing a tuple of # units and
# activation

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
plt.show()
print ("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

architecture = [(20, relu), (7, relu), (5, relu), (1, sigmoid)]

# This will load pre-trained weights. If you wish to train the model yourself, delete the weights.h5 file
try:
    params = load_weights()
except IOError:
    params, losses = dnn(train_x, train_y, architecture, iter=2500)
    save_weights(params)  # Save weights

    # Plot learning curve for train set
    plt.plot(losses)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.show()

train_acc = accuracy(train_x, architecture, params, train_y)
test_acc = accuracy(test_x, architecture, params, test_y)
print(f'Train acc: {train_acc}  |  Test acc: {test_acc}')

