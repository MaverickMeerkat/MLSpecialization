import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def softmax(z):
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    return e/s


def sigmoid_back(da, z):
    a = sigmoid(z)
    return da * a * (1 - a)


def tanh_back(da, z):
    return da * (1 - tanh(z)**2)


def relu_back(da, z):
    dz = np.copy(da)
    dz[z < 0] = 0
    return dz


def softmax_back(da, z):
    # z, da shapes - (m, n)
    m, n = z.shape
    p = softmax(z)
    tensor1 = np.einsum('ij,ik->ijk', p, p)  # (m, n, n)
    tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))  # (m, n, n)
    dSoftmax = tensor2 - tensor1
    dz = np.einsum('ijk,ik->ij', dSoftmax, da)  # (m, n)
    return dz


MAPPING = {sigmoid: sigmoid_back,
           relu: relu_back,
           tanh: tanh_back,
           softmax: softmax_back}

