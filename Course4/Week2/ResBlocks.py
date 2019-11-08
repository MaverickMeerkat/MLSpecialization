import numpy as np
import tensorflow as tf


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    shape = tf.shape(X)

    # First component of main path
    X = tf.layers.Conv2D(F1, (1, 1), strides=(1, 1), name=conv_name_base + '2a', padding='VALID')(X)
    X = tf.layers.batch_normalization(X, name=bn_name_base + '2a')
    X = tf.nn.relu(X)

    # Second component of main path
    X = tf.layers.Conv2D(F2, (f, f), strides=(1, 1), padding='SAME', name=conv_name_base + '2b')(X)
    X = tf.layers.batch_normalization(X, name=bn_name_base + '2b')
    X = tf.nn.relu(X)

    # Third component of main path
    X = tf.layers.Conv2D(F3, (1, 1), strides=(1, 1), padding='VALID', name=conv_name_base + '2c')(X)
    X = tf.layers.batch_normalization(X, name=bn_name_base + '2c')

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.add(X, X_shortcut)
    X = tf.nn.relu(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = tf.layers.Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', padding='VALID')(X)
    X = tf.layers.batch_normalization(X, name=bn_name_base + '2a')
    X = tf.nn.relu(X)

    # Second component of main path
    X = tf.layers.Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='SAME')(X)
    X = tf.layers.batch_normalization(X, name=bn_name_base + '2b')
    X = tf.nn.relu(X)

    # Third component of main path
    X = tf.layers.Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='VALID')(X)
    X = tf.layers.batch_normalization(X, name=bn_name_base + '2c')

    X_shortcut = tf.layers.Conv2D(F3, (1,1), strides=(s, s),
                              name=conv_name_base + '1', padding='VALID')(X_shortcut)
    X_shortcut = tf.layers.batch_normalization(X_shortcut, name=bn_name_base + '1')

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = tf.add(X_shortcut, X)
    X = tf.nn.relu(X)

    return X