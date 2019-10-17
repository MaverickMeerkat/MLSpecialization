from Course2.Week1.activations import *
from Course4.Week1.BackProp import *
from Course4.Week1.ForwardProp import conv_forward, pool_forward
from Course4.Week1.Helpers import *


class ConvNet(object):
    EPS = 1e-12

    def __init__(self, input_shape, architecture):
        self.input_shape = input_shape
        self.architecture = architecture
        self.params = self.initialize_parameters()

    def initialize_parameters(self):
        np.random.seed(1)  # this really seems to have huge affect on the convergence of the cost function
        params = []
        shape = self.input_shape
        m = shape[0]
        for layer in self.architecture:
            if layer[0] == "Dense":
                n_l_1 = layer[1]
                self.xaviar_initialization(shape[1], n_l_1, params)
                shape = (m, n_l_1)
            elif layer[0] == "Conv":
                self.conv_initialization(layer[1], params)
                shape = (shape[0], shape[1], shape[2], layer[1][3])
                # conv is assumed to be same shape
                # todo: add support for non same layers, and update shape
            elif layer[0] == "Pool":
                s1 = (shape[1] - layer[1][0])//layer[1][2] + 1
                s2 = (shape[2] - layer[1][1])//layer[1][2] + 1
                shape = (shape[0], s1, s2, shape[3])
                params.append(None)
            elif layer[0] == "Flatten":
                n = shape[1] * shape[2] * shape[3]
                shape = (m, n)
                params.append(None)
        return params

    def xaviar_initialization(self, n_l_0, n_l_1, params):
        W = np.random.randn(n_l_0, n_l_1) * np.sqrt(1 / n_l_0)  # Xaviar initialization
        b = np.zeros((1, n_l_1))
        params.append({'W': W, 'b': b})

    def conv_initialization(self, shape, params):
        W = np.random.randn(shape[0], shape[1], shape[2], shape[3]) * np.sqrt(1 / (shape[0]*shape[1]*shape[2]))
        b = np.zeros((1, 1, 1, shape[3]))
        params.append({'W': W, 'b': b})

    def train(self, X, y, epochs=1000, learning_rate=0.1, lr_decay=0.001, lambd=0, print_every=10):
        losses = []
        for i in range(epochs):
            a, caches = self.forward_pass(X)
            da = np.divide(-y, a + self.EPS)  # Heuristic for softmax loss
            l = self.cost(a, y, lambd)
            gradients = self.backward_pass(da, caches, lambd)
            lr = learning_rate/(1+lr_decay*i)  # learning rate decay
            self.update_weights(gradients, lr=lr)
            if i % print_every == 0:
                acc = self.accuracy(X, y)
                print(f'iteration - {i}  -  loss: {l} | accuracy {acc}')
                losses.append(l)
        return losses

    def cost(self, h, y, lambd):
        '''
        Softmax loss, w/ regularization
        In practice this is not used in the backprop algorithm, as we use a heuristic for its derivative
        :param h: hypothesis output, y-hat, final activation a
        :param y: true values
        :return:
        '''
        m = y.shape[0]

        cost_w = 0
        for dict in self.params:
            if dict is None:
                continue
            w = dict['W']
            cost_w += (lambd / (2 * m)) * np.sum(w ** 2)
        loss = (1 / m) * np.sum(-np.log(h[np.arange(m), np.argmax(y, axis=1)]))
        return np.squeeze(loss)

    def backward_pass(self, da, caches, lambd):
        m = da.shape[0]
        gradients = []
        layers = self.architecture
        for i in reversed(range(len(caches))):
            if layers[i][0] == "Conv":
                cache = caches[i]
                activation = layers[i][4]
                back = MAPPING[activation]
                z = cache[0]
                dz = back(da, z)
                da, dW, db = conv_backward(dz, cache)
                gradients.insert(0, {'dW': dW, 'db': db})  # insert in reverse, to match parameters list
            elif layers[i][0] == "Pool":
                cache = caches[i]
                da = pool_backward(da, cache, mode=layers[i][2])
                gradients.insert(0, None)
            elif layers[i][0] == "Flatten":
                shape = caches[i]
                da = da.reshape(shape)
                gradients.insert(0, None)
            elif layers[i][0] == "Dense":
                cache = caches[i]
                W = cache['params']['W']
                a = cache['a']
                z = cache['z']
                back = cache['back']
                dW, db, da = self.layer_backward(da, a, W, z, m, back, lambd)
                gradients.insert(0, {'dW': dW, 'db': db})  # insert in reverse, to match parameters list
        return gradients

    @staticmethod
    def layer_backward(da, a, W, z, m, back, lambd):
        dz = back(da, z)
        dW = (1 / m) * a.T @ dz + (lambd / m) * W
        db = (1 / m) * np.sum(dz, axis=0, keepdims=True)
        da = dz @ W.T
        return dW, db, da

    def forward_pass(self, X):
        '''
        Make a forward pass in a NN
        :param X: (m, n_W, n_H, n_C) matrix
        :return:
        '''
        np.random.seed(1)

        a = X
        caches = []
        layers = self.architecture
        for i in range(len(layers)):
            if layers[i][0] == "Conv":
                a, cache = self.conv(a, i, layers)
                caches.append(cache)
            elif layers[i][0] == "Pool":
                hparameters = {'f': layers[i][1][0], 'stride': layers[i][1][2]}  # todo: add 2nd f
                a, cache = pool_forward(a, hparameters, mode=layers[i][2])
                caches.append(cache)
            elif layers[i][0] == "Flatten":
                a, old_shape = self.flatten(a)
                caches.append(old_shape)
            elif layers[i][0] == "Dense":
                a, cache = self.dense_forward(a, i)
                caches.append(cache)
        return a, caches

    def conv(self, a, i, layers):
        W = self.params[i]['W']
        b = self.params[i]['b']
        hparameters = {'stride': layers[i][2], 'pads': layers[i][3]}
        Z, cache = conv_forward(a, W, b, hparameters)
        a = layers[i][4](Z)
        return a, cache

    def dense_forward(self, a, i):
        activation = self.architecture[i][2]
        W = self.params[i]['W']
        b = self.params[i]['b']
        next_a, z = self.layer_forward(a, W, b, activation)
        back = MAPPING[activation]
        cache = {'params': self.params[i], 'z': z, 'a': a, 'back': back}
        return next_a, cache
    
    @staticmethod
    def layer_forward(a, W, b, activation):
        z = a @ W + b
        next_a = activation(z)
        return next_a, z

    @staticmethod
    def flatten(a):
        m = a.shape[0]
        old_shape = a.shape
        return a.reshape(m, -1), old_shape

    def update_weights(self, gradients, lr=0.01):
        for i in range(len(self.params)):
            if self.params[i] is None:
                continue
            self.params[i]['W'] -= lr * gradients[i]['dW']
            self.params[i]['b'] -= lr * gradients[i]['db']

    def accuracy(self, X, y):
        h = self.predict(X)
        pred = np.argmax(h, axis=1)
        y_int = np.argmax(y, axis=1)
        acc = np.mean(y_int == pred)
        return acc

    def reset_params(self):
        self.params = self.initialize_parameters()

    def predict(self, X):
        h, _ = self.forward_pass(X)
        return h
