import tensorflow as tf

from Course4.Week2.Helpers import load_dataset, convert_to_one_hot
from Course4.Week2.ResBlocks import convolutional_block, identity_block

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

def ResNet50(X_data, Y_data, lr=0.1, epochs=10, print_cost=False):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    """
    tf.reset_default_graph()  # this makes sure the variables are reset

    shape = (None, X_data.shape[1], X_data.shape[2], X_data.shape[3])
    classes = Y_data.shape[1]
    tf_x = tf.placeholder(tf.float32, shape)  # input x
    tf_y = tf.placeholder(tf.int32, (None, classes))  # input y

    # # Zero-Padding
    # paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
    # X = tf.pad(tf_x, paddings)

    # Stage 1
    X = tf.layers.Conv2D(64, (7, 7), strides=2, padding='SAME', name='conv1')(tf_x)
    X = tf.layers.batch_normalization(X, axis=-1, name='bn_conv1')
    X = tf.nn.relu(X)
    X = tf.nn.max_pool2d(X, (3, 3), strides=(2, 2), padding='VALID')

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = tf.nn.avg_pool2d(X, (2, 2), strides=(1, 1), padding='VALID')

    # output layer
    X = tf.layers.flatten(X)
    output = tf.layers.dense(X, classes, activation='softmax', name='fc' + str(classes),
                             kernel_initializer=tf.initializers.glorot_uniform(seed=0))

    # loss, optimizer & metric
    loss = tf.losses.softmax_cross_entropy(tf_y, output)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)
    pred = tf.argmax(output, axis=1)
    y_h = tf.argmax(tf_y, axis=1)
    accuracy, update_op = tf.metrics.accuracy(labels=y_h, predictions=pred)

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    feed_dict = {tf_x: X_data, tf_y: Y_data}
    with tf.Session() as sess:
        sess.run(init_global)
        for step in range(epochs):
            # train and net output
            sess.run(init_local)
            sess.run(train_op, feed_dict=feed_dict)
            cur_loss = sess.run(loss, feed_dict=feed_dict)

            sess.run(update_op, feed_dict=feed_dict)
            acc = sess.run(accuracy, feed_dict=feed_dict)
            if print_cost:  # step % 100 == 0 and
                print(f'step {step} - loss: {cur_loss} | acc: {acc}')

        print('Optimization finished')

    return weights


ResNet50(X_train, Y_train)
