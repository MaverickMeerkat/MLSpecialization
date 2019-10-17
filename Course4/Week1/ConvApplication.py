import h5py
import numpy as np

from Course2.Week1.activations import *
from Course4.Week1.BackProp import *
from Course4.Week1.Helpers import *
from Course4.Week1.ConvNet import ConvNet


# Exercise uses TF - we will use the implementations in the first exercise
# Note this is unbelievably slow...

# Ex1 (Numpy) Implementation

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# Hyper-Parameters
lr = 0.05
epochs = 100
lambd = 0

arch = [('Conv', [4, 4, 3, 8], 1, [2, 1], relu),  # type, weight-shape, stride, pads, activation
        ('Pool', [8, 8, 8], "max"),  # type, W-H-Stride, mode - max/average
        ('Conv', [2, 2, 8, 16], 1, [1, 0], relu),
        ('Pool', [4, 4, 4], "max"),
        ("Flatten",),
        ("Dense", 6, softmax)]


conv_net = ConvNet(X_train.shape, arch)
costs = conv_net.train(X_train[:100], Y_train[:100], lr_decay=0, lambd=lambd, epochs=epochs, learning_rate=lr, print_every=1)

print(f"Accuracy on the training set: {conv_net.accuracy(X_train, Y_train)}")
print(f"Accuracy on the training set: {conv_net.accuracy(X_test, Y_test)}")
