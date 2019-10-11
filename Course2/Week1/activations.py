import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def sigmoid_back(da, z):
    a = sigmoid(z)
    return da * a * (1 - a)


def tanh_back(da, z):
    return da * (1 - tanh(z)**2)


def relu_back(da, z):
    dz = np.copy(da)
    dz[z < 0] = 0
    return dz


MAPPING = {sigmoid: sigmoid_back,
           relu: relu_back,
           tanh: tanh_back}

