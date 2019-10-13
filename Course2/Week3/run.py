import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

from Course2.Week3.DNNClass3 import *
from Course2.Week1.Architecture import Architecture
from Course2.Week1.activations import *


# The exercise focuses on TensorFlow
# Here I will try to implement hyper-parameter search, and batch normalization


def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=500, noise=.25)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y


train_X, train_Y = load_dataset()

# Visualize the data
plt.scatter(train_X.T[:, 0], train_X.T[:, 1], c=train_Y.T.ravel(), s=40, cmap=plt.cm.Spectral)
plt.show()

# Hyper-Parameters
lr = 0.03
dropout = 1.0
epochs = 10000
lambd = 0

# DNN architecture
list = [(5, relu, 1.0), (0, None, dropout), (3, relu, dropout), (1, sigmoid, dropout)]
architecture = Architecture.from_list(list)

# Instantiate and train
dnn = DNNModel3(train_X.shape[0], architecture)
costs = dnn.train(train_X, train_Y, lr_decay=0, lambd=lambd, epochs=epochs, learning_rate=lr)
