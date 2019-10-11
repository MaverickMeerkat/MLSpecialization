from Course2.Week1.activations import *


class DNNModel(object):
    EPS = 1e-12

    def __init__(self, input_len, architecture):
        self.input_len = input_len
        self.architecture = architecture
        self.params = self.initialize_parameters(input_len)

    def initialize_parameters(self, n_x):
        np.random.seed(1)  # this really seems to have huge affect on the convergence of the cost function
        params = []
        n_l_0 = n_x
        for layer in self.architecture.get_layers():
            n_l_1 = layer.units
            self.xaviar_initialization(n_l_0, n_l_1, params)
            n_l_0 = n_l_1
        return params

    def train(self, X, y, epochs=1000, learning_rate=0.0075, lr_decay=0.001, lambd=0, print_every=1000):
        losses = []
        for i in range(epochs):
            a, caches = self.forward_pass(X)
            layers = self.architecture.get_layers()
            da = np.divide(a - y, a*(1-a) + self.EPS)  # Heuristic for log loss
            l = self.cost(a, y, lambd)
            gradients = self.backward_pass(da, caches, layers, lambd)
            lr = learning_rate/(1+lr_decay*i)  # learning rate decay
            self.update_weights(gradients, lr=lr)
            if i % print_every == 0:
                acc = self.accuracy(X, y)
                print(f'iteration - {i}  -  loss: {l} | accuracy {acc}')
                losses.append(l)
        return losses

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
            activation = layers[i].activation
            keep_prob = layers[i].keep_prob
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
        return a, caches

    @staticmethod
    def xaviar_initialization(n_l_0, n_l_1, params):
        W = np.random.randn(n_l_1, n_l_0) * np.sqrt(1 / n_l_0)  # Xaviar initialization
        b = np.zeros((n_l_1, 1))
        params.append({'W': W, 'b': b})

    @staticmethod
    def layer_forward(a, W, b, activation):
        z = W @ a + b
        next_a = activation(z)
        return next_a, z

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
            w = dict['W']
            cost_w += (lambd / (2 * m)) * np.sum(w ** 2)
        cost = (-1 / m) * (y @ np.log(h + self.EPS).T + (1 - y) @ np.log(1 - h + self.EPS).T) + cost_w
        return np.squeeze(cost)

    @staticmethod
    def layer_backward(da, a, W, z, m, back, lambd):
        dz = back(da, z)
        dW = (1 / m) * dz @ a.T + (lambd / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        da = W.T @ dz
        return dW, db, da

    def backward_pass(self, da, caches, layers, lambd):
        m = da.shape[1]
        graients = []
        for i in reversed(range(len(caches))):
            cache = caches[i]
            W = cache['params']['W']
            a = cache['a']
            z = cache['z']
            back = cache['back']
            dW, db, da = self.layer_backward(da, a, W, z, m, back, lambd)
            d = cache['d']  # dropout shutdown units
            keep_prob = layers[i].keep_prob
            da = da * d / keep_prob
            graients.insert(0, {'dW': dW, 'db': db})  # insert in reverse, to match parameters list
        return graients

    def update_weights(self, gradients, lr=0.01):
        for i in range(len(self.params)):
            self.params[i]['W'] -= lr * gradients[i]['dW']
            self.params[i]['b'] -= lr * gradients[i]['db']

    def accuracy(self, X, y):
        h = self.predict(X)
        pred = h > 0.5
        acc = np.mean(y == pred)
        return acc

    def reset_params(self):
        self.params = self.initalize_parameters(self.input_len)

    def predict(self, X):
        h, _ = self.forward_pass(X, False)
        return h
