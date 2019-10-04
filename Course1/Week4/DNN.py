import numpy as np

import Course1.Week4.dnn_utils as utils


def layer_forward(a, W, b, activation):
    z = W @ a + b
    next_a = activation(z)
    return next_a, z


def layer_backward(da, a, W, z, m, back):
    dz = back(da, z)
    dW = (1/m) * dz @ a.T
    db = (1/m) * np.sum(dz, axis=1, keepdims=True)
    da = W.T @ dz
    return dW, db, da


def forward_pass(X, params, architecture):
    '''
    Make a forward pass in a NN
    :param X: n_x by m matrix
    :param architecture: list of tuples, each containing number of units in the layer, and the activation
    :return:
    '''
    a = X
    caches = []
    for i in range(len(architecture)):
        _, activation = architecture[i]
        W = params[i]['W']
        b = params[i]['b']
        next_a, z = layer_forward(a, W, b, activation)
        back = utils.MAPPING[activation]
        cache = {'params': params[i], 'z': z, 'a': a, 'back': back}
        caches.append(cache)
        a = next_a
    return a, caches


def cost(h, y):
    '''
    Cross entropy cost / log loss, w/o regularization
    In practice this is not used in the backprop algorithm, as we use a heuristic for its gradient
    :param h: hypothesis output, y-hat, final activation a
    :param y: true values
    :return:
    '''
    m = y.shape[1]
    cost = (-1 / m) * (y @ np.log(h).T + (1 - y) @ np.log(1 - h).T)
    return np.squeeze(cost)


def backward_pass(da, caches):
    m = da.shape[1]
    graients = []
    for i in reversed(range(len(caches))):
        cache = caches[i]
        W = cache['params']['W']
        a = cache['a']
        z = cache['z']
        back = cache['back']
        dW, db, da = layer_backward(da, a, W, z, m, back)
        graients.insert(0, {'dW': dW, 'db': db})  # insert in reverse, to match parameters list
    return graients


def update_weights(params, gradients, lr=0.01):
    for i in range(len(params)):
        params[i]['W'] -= lr * gradients[i]['dW']
        params[i]['b'] -= lr * gradients[i]['db']


def initalize_parameters(n_x, architecture):
    np.random.seed(1)  # this really seems to have huge affect on the convergence of the cost function
    params = []
    n_l_0 = n_x
    for i in range(len(architecture)):
        n_l_1, activation = architecture[i]
        W = np.random.randn(n_l_1, n_l_0) * np.sqrt(1 / n_l_0)  # Xaviar initialization
        b = np.zeros((n_l_1, 1))
        params.append({'W': W, 'b': b})
        n_l_0 = n_l_1
    return params


def dnn_model(X, y, architecture, iter=1000, learning_rate=0.0075, decay=0.001):
    params = initalize_parameters(X.shape[0], architecture)
    losses = []
    for i in range(iter):
        a, caches = forward_pass(X, params, architecture)
        l = cost(a, y)
        da = - (np.divide(y, a) - np.divide(1 - y, 1 - a))  # Heuristic for log loss
        gradients = backward_pass(da, caches)
        lr = learning_rate/(1+decay*i)  # learning rate decay
        update_weights(params, gradients, lr=lr)
        if i % 100 == 0:
            acc = accuracy(X, architecture, params, y)
            print(f'iteration - {i}  -  loss: {l} | accuracy {acc}')
            losses.append(l)
    return params, losses


def accuracy(X, architecture, params, y):
    h, _ = forward_pass(X, params, architecture)
    pred = h > 0.5
    acc = np.mean(y == pred)
    return acc
