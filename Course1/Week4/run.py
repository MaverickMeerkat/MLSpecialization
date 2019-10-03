import numpy as np
import matplotlib.pyplot as plt

from Course1.Week4.DNN import dnn_model as dnn
from Course1.Week4.dnn_utils import *


# This is a variation on the exercise - although I use numpy and modular functions, the functions are different...
# Main difference is an architecture parameter carrying a list of layers, each containing a tuple of # units and
# activation


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

architecture = [(7, relu), (1, sigmoid)]

dnn(train_x, train_y, architecture)