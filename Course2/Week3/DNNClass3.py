import numpy as np

from Course2.Week2.DNNClass2 import DNNModel2
from Course2.Week1.activations import *
from Course2.Week3.BatchNorm import *

class DNNModel3(DNNModel2):
    """
    DNN Model 2 with batch normalization.
    """
    def __init__(self, input_len, architecture):
        super().__init__(input_len, architecture)

    def initialize_parameters(self, n_x):
        np.random.seed(1)  # this really seems to have huge affect on the convergence of the cost function
        params = []
        n_l_0 = n_x
        for layer in self.architecture.get_layers():
            n_l_1 = layer.units
            if n_l_1 == 0:
                self.batchnorm_init(n_l_0, params)
            else:
                self.xaviar_initialization(n_l_0, n_l_1, params)
                n_l_0 = n_l_1
        return params

    def batchnorm_init(self, n_l_0, params):
        gamma = np.random.randn(n_l_0, 1) * np.sqrt(1 / n_l_0)
        beta = np.zeros((n_l_0, 1))
        params.append({'gamma': gamma, 'beta': beta})

    def forward_pass(self, X, train=True):
        '''
        Make a forward pass in a NN
        :param X: n_x by m matrix
        :param train: should you use dropout or not
        :return:
        '''
        np.random.seed(1)

        a = X
        caches = []
        layers = self.architecture.get_layers()
        for i in range(len(layers)):
            units = layers[i].units
            activation = layers[i].activation
            keep_prob = layers[i].keep_prob
            if units != 0:  # regular layer
                d = np.ones(a.shape)
                if train:  # dropout
                    d = np.random.rand(a.shape[0], a.shape[1])
                    d = d < keep_prob
                    # shut down some neurons of a, & scale the value of neurons that haven't been shut down
                    a = a * d / keep_prob
                W = self.params[i]['W']
                b = self.params[i]['b']
                next_a, z = self.layer_forward(a, W, b, activation)
                back = MAPPING[activation]
                cache = {'params': self.params[i], 'z': z, 'd': d, 'a': a, 'back': back}
                caches.append(cache)
                a = next_a
            else:  # batchnorm layer
                gamma = self.params[i]['gamma']
                beta = self.params[i]['beta']
                next_a, cache = batchnorm_forward(a, gamma, beta, self.EPS)
                caches.append(cache)
                a = next_a
        return a, caches

    def cost(self, h, y, lambd):
        '''
        Cross entropy cost / log loss, w/ regularization
        In practice this is not used in the backprop algorithm, as we use a heuristic for its gradient
        :param h: hypothesis output, y-hat, final activation a
        :param y: true values
        :return:
        '''
        m = y.shape[1]

        cost_w = 0
        for dict in self.params:
            if 'W' in dict:
                w = dict['W']
            else:  # batchnorm layer
                w = dict['gamma']
            cost_w += (lambd / (2 * m)) * np.sum(w ** 2)
        cost = (-1 / m) * (y @ np.log(h + self.EPS).T + (1 - y) @ np.log(1 - h + self.EPS).T) + cost_w
        return np.squeeze(cost)

    def backward_pass(self, da, caches, layers, lambd):
        m = da.shape[1]
        graients = []
        for i in reversed(range(len(caches))):
            cache = caches[i]
            if type(cache) is dict:  # regular layer
                W = cache['params']['W']
                a = cache['a']
                z = cache['z']
                back = cache['back']
                dW, db, da = self.layer_backward(da, a, W, z, m, back, lambd)
                d = cache['d']  # dropout shutdown units
                keep_prob = layers[i].keep_prob
                da = da * d / keep_prob
                graients.insert(0, {'dW': dW, 'db': db})  # insert in reverse, to match parameters list
            else:  # batchnorm layer
                da, dgamma, dbeta = batchnorm_backward(da, cache)
                graients.insert(0, {'dgamma': dgamma, 'dbeta': dbeta})
        return graients

    def initialize_optimizer(self):
        momentum = {}
        rms_prop = {}
        for i in range(len(self.params)):
            param = self.params[i]
            if 'W' in param:
                momentum['dW' + str(i + 1)] = np.zeros_like(param['W'])
                momentum['db' + str(i + 1)] = np.zeros_like(param['b'])
                rms_prop['dW' + str(i + 1)] = np.zeros_like(param['W'])
                rms_prop['db' + str(i + 1)] = np.zeros_like(param['b'])
            elif 'gamma' in param:
                momentum['dgamma' + str(i + 1)] = np.zeros_like(param['gamma'])
                momentum['dbeta' + str(i + 1)] = np.zeros_like(param['beta'])
                rms_prop['dgamma' + str(i + 1)] = np.zeros_like(param['gamma'])
                rms_prop['dbeta' + str(i + 1)] = np.zeros_like(param['beta'])
        return momentum, rms_prop

    def update_weights(self, gradients, lr, t, v, s, beta1, beta2):
        v_corrected = {}
        s_corrected = {}
        for i in range(len(self.params)):
            param = self.params[i]
            if 'W' in param:
                weight_key = 'W'
                bias_key = 'b'
                dweight_key = 'dW' + str(i + 1)
                dbias_key = 'db' + str(i + 1)
                dweight = gradients[i]['dW']
                dbias = gradients[i]['db']
            else:
                weight_key = 'gamma'
                bias_key = 'beta'
                dweight_key = 'dgamma' + str(i + 1)
                dbias_key = 'dbeta' + str(i + 1)
                dweight = gradients[i]['dgamma']
                dbias = gradients[i]['dbeta']

            v[dbias_key] = beta1 * v[dbias_key] + (1 - beta1) * dbias
            v[dweight_key] = beta1 * v[dweight_key] + (1 - beta1) * dweight
            v_corrected[dweight_key] = v[dweight_key] / (1 - beta1 ** t)
            v_corrected[dbias_key] = v[dbias_key] / (1 - beta1 ** t)

            s[dweight_key] = beta2 * s[dweight_key] + (1 - beta2) * (dweight ** 2)
            s[dbias_key] = beta2 * s[dbias_key] + (1 - beta2) * (dbias ** 2)
            s_corrected[dweight_key] = s[dweight_key] / (1 - beta2 ** t)
            s_corrected[dbias_key] = s[dbias_key] / (1 - beta2 ** t)

            param[weight_key] -= lr * (
                    v_corrected[dweight_key] / (np.sqrt(s_corrected[dweight_key]) + self.EPS))
            param[bias_key] -= lr * (
                    v_corrected[dbias_key] / (np.sqrt(s_corrected[dbias_key]) + self.EPS))


