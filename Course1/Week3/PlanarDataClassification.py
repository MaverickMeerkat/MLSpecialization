import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

from Course1.Week3.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_blobs_dataset, \
    load_circle_dataset, load_moons_datasets, load_gaussian_quantiles_dataset, load_no_structure_dataset


np.random.seed(1)

# Load data
X, Y = load_gaussian_quantiles_dataset()
# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=20, cmap=plt.cm.Spectral)
plt.show()

# Simple Logistic Regression
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T.ravel())

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()

pred = clf.predict(X.T)
print(f'accuracy: {np.mean(Y.T == pred)*100}%')


# The exercise build a single-layer NN from the grounds up, using only numpy
# We will use Tensorflow
def nn_model(X, Y, n_h, epochs=1000, print_cost=False):
    tf.reset_default_graph()  # this makes sure the variables are reset

    X = X.T
    Y = Y.T

    # actual model
    tf_x = tf.placeholder(tf.float32, (None, X.shape[1]))  # input x
    tf_y = tf.placeholder(tf.int32, (None, Y.shape[1]))  # input y
    hidden = tf.layers.dense(tf_x, n_h, tf.tanh)
    output = tf.layers.dense(hidden, 1, tf.sigmoid)

    # loss, optimizer & metric
    loss = tf.losses.log_loss(tf_y, output)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss)
    accuracy, update_op = tf.metrics.accuracy(labels=tf_y, predictions=(output > 0.5))

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    feed_dict = {tf_x: X, tf_y: Y}
    with tf.Session() as sess:
        sess.run(init_global)
        for step in range(epochs):
            # train and net output
            sess.run(init_local)
            sess.run(train_op, feed_dict=feed_dict)
            cur_loss = sess.run(loss, feed_dict=feed_dict)

            sess.run(update_op, feed_dict=feed_dict)
            acc = sess.run(accuracy, feed_dict=feed_dict)
            if step % 100 == 0 and print_cost:
                print(f'step {step} - loss: {cur_loss} | acc: {acc}')

        print('Optimization finished')
        weights = sess.run(tf.trainable_variables())

    return weights


def predict(weights, x):
    w1 = weights[0]
    b1 = weights[1]
    w2 = weights[2]
    b2 = weights[3]
    z1 = x @ w1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    return a2 > 0.5


plt.figure(figsize=(12, 6))
hidden_layer_sizes = [2, 4, 6]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(1, 3, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    weights = nn_model(X, Y, n_h, print_cost=True)
    plot_decision_boundary(lambda x: predict(weights, x), X, Y)
plt.show()
