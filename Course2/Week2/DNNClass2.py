import numpy as np

from Course2.Week1.DNNClass import DNNModel
from Course2.Week1.activations import *

class DNNModel2(DNNModel):
    """
    DNN Model with mini batches, momentum and RMS prop (aka Adam)
    """
    def __init__(self, input_len, architecture):
        super().__init__(input_len, architecture)

    def train(self, X, y, batch_size=64, epochs=1000, learning_rate=0.0075, lr_decay=0.001,
              lambd=0, print_every=1000, beta1=0.9, beta2=0.999):
        """
        :param X: X
        :param y: y
        :param batch_size: number of examples per batch: 1 for SGD, m for batch GD
        :param epochs: number of epochs
        :param learning_rate: learning rate
        :param lr_decay: learning rate decay
        :param lambd: L2 regularization
        :param print_every: after how many epochs to print
        :param beta1: parameter for momentum
        :param beta2: parameter for RMS prop
        :return:
        """
        v, s = self.initialize_optimizer()
        losses = []
        t = 1  # bias correction for mom/rms
        for i in range(epochs):
            mini_batches = self.random_mini_batches(X, y, batch_size)
            for mini_batch in mini_batches:
                (minibatch_X, minibatch_Y) = mini_batch
                a, caches = self.forward_pass(minibatch_X)
                layers = self.architecture.get_layers()
                da = np.divide(a - minibatch_Y, a*(1-a) + self.EPS)  # Heuristic for log loss
                l = self.cost(a, minibatch_Y, lambd)
                gradients = self.backward_pass(da, caches, layers, lambd)
                lr = learning_rate/(1+lr_decay*i)  # learning rate decay
                self.update_weights(gradients, lr, t, v, s, beta1, beta2)
                t += 1
            if i % print_every == 0:
                acc = self.accuracy(X, y)
                print(f'iteration - {i}  -  loss: {l} | accuracy {acc}')
                losses.append(l)
        return losses

    def initialize_optimizer(self):
        momentum = {}
        rms_prop = {}
        for i in range(len(self.params)):
            momentum['dW' + str(i+1)] = np.zeros_like(self.params[i]['W'])
            momentum['db' + str(i+1)] = np.zeros_like(self.params[i]['b'])
            rms_prop['dW' + str(i+1)] = np.zeros_like(self.params[i]['W'])
            rms_prop['db' + str(i+1)] = np.zeros_like(self.params[i]['b'])
        return momentum, rms_prop

    def update_weights(self, gradients, lr, t, v, s, beta1, beta2):
        v_corrected = {}
        s_corrected = {}
        for i in range(len(self.params)):
            v['dW' + str(i+1)] = beta1 * v['dW' + str(i+1)] + (1 - beta1) * gradients[i]['dW']
            v['db' + str(i+1)] = beta1 * v['db' + str(i+1)] + (1 - beta1) * gradients[i]['db']
            v_corrected['dW' + str(i+1)] = v['dW' + str(i+1)] / (1 - beta1**t)
            v_corrected['db' + str(i+1)] = v['db' + str(i+1)] / (1 - beta1**t)

            s['dW' + str(i+1)] = beta2 * s['dW' + str(i+1)] + (1 - beta2) * (gradients[i]['dW']**2)
            s['db' + str(i+1)] = beta2 * s['db' + str(i+1)] + (1 - beta2) * (gradients[i]['db']**2)
            s_corrected['dW' + str(i+1)] = s['dW' + str(i+1)] / (1 - beta2**t)
            s_corrected['db' + str(i+1)] = s['db' + str(i+1)] / (1 - beta2**t)

            self.params[i]['W'] -= lr * (v_corrected['dW' + str(i+1)] / (np.sqrt(s_corrected['dW' + str(i+1)]) + self.EPS))
            self.params[i]['b'] -= lr * (v_corrected['db' + str(i + 1)] / (np.sqrt(s_corrected['db' + str(i + 1)]) + self.EPS))

    @staticmethod
    def random_mini_batches(X, y, batch_size):
        m = X.shape[1]
        # Shuffle X and Y
        perms = list(np.random.permutation(m))
        X_shuffled = X[:, perms]
        y_shuffled = y[:, perms].reshape(1, m)
        mini_batches = []
        num = m // batch_size
        for i in range(num):
            mini_batch_X = X_shuffled[:, i*batch_size:(i+1)*batch_size]
            mini_batch_y = y_shuffled[:, i*batch_size:(i+1)*batch_size]
            mini_batches.append((mini_batch_X, mini_batch_y))
        if m % batch_size != 0:
            mini_batch_X = X_shuffled[:, num * batch_size:]
            mini_batch_y = y_shuffled[:, num * batch_size:]
            mini_batches.append((mini_batch_X, mini_batch_y))
        return mini_batches
