import numpy as np
import scipy.io
import sklearn.datasets
import matplotlib.pyplot as plt

from Course2.Week1.DNNClass import *
from Course2.Week1.utils import plot_decision_boundary
from Course2.Week3.DNNClass3 import *
from Course2.Week1.Architecture import Architecture
from Course2.Week1.activations import *


# The exercise focuses on TensorFlow
# Here I will try to implement hyper-parameter search, and batch normalization

# Hyper Parameters search
def load_2D_dataset():
    data = scipy.io.loadmat('../Week1/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    return train_X, train_Y, test_X, test_Y


# Load data
train_X, train_Y, test_X, test_Y = load_2D_dataset()

# Scatter
plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral)
plt.show()

# Hyper-Parameters
epochs = 10000
lambd = 0

def get_random_lr():
    # From 1 to 1/(10^5)
    np.random.seed()
    x = np.random.rand()
    return 10**(-5*x)

def get_random_dropout():
    # From 0.5 to 1.0
    np.random.seed()
    x = np.random.rand()
    return 0.5*(1+x)


for i in range(10):
    lr = get_random_lr()
    dropout = get_random_dropout()
    list = [(20, relu, 1.0), (3, relu, dropout), (1, sigmoid, dropout)]
    architecture = Architecture.from_list(list)
    dnn = DNNModel(train_X.shape[0], architecture)
    costs = dnn.train(train_X, train_Y, lr_decay=0, lambd=lambd, epochs=epochs, learning_rate=lr)
    plt.plot(costs, label=f'lr={lr}; do={dropout}')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('iterations (x1,000)')
plt.show()


# Batch Normalization

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
print(f"Accuracy on the training set: {dnn.accuracy(train_X, train_Y)}")

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: dnn.predict(x.T), train_X, train_Y)
