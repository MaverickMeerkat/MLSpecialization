import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

from Course2.Week1.Architecture import Architecture
from Course2.Week1.utils import plot_decision_boundary
from Course2.Week2.DNNClass2 import DNNModel2
from Course2.Week1.activations import *


def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)
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
list = [(5, relu, 1.0), (3, relu, dropout), (1, sigmoid, dropout)]
architecture = Architecture.from_list(list)

# Instantiate and train
dnn = DNNModel2(train_X.shape[0], architecture)
costs = dnn.train(train_X, train_Y, lr_decay=0, lambd=lambd, epochs=epochs, learning_rate=lr)

print(f"Accuracy on the training set: {dnn.accuracy(train_X, train_Y)}")

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: dnn.predict(x.T), train_X, train_Y)
