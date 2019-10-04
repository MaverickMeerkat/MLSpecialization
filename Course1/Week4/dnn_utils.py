import numpy as np
import h5py



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


def load_data():
    train_dataset = h5py.File('./train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('./test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def save_weights(weights):
    hf = h5py.File('./weights.h5', 'w')
    i = 0
    for w in weights:
        hf.create_dataset(f'weights_W_{i}', data=w['W'])
        hf.create_dataset(f'weights_b_{i}', data=w['b'])
        i += 1
    hf.close()



def load_weights():
    hf = h5py.File('./weights.h5', 'r')
    weights = []
    for i in range(len(hf.keys())//2):
        W = np.array(hf.get(f'weights_W_{i}'))
        b = np.array(hf.get(f'weights_b_{i}'))
        weights.append({'W': W, 'b': b})
    return weights
