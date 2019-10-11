import scipy.io
import matplotlib.pyplot as plt

from Course2.Week1.DNNClass import DNNModel
from Course2.Week1.Architecture import Architecture
from Course2.Week1.activations import *
from Course2.Week1.utils import plot_decision_boundary

# A variation of the exercise.
# Uses OOP design, which allows for both L2 regularization and Dropout
# Specify lambda other than 0 for L2 regularization, and dropout < 1 for dropout

def load_2D_dataset():
    data = scipy.io.loadmat('./data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    return train_X, train_Y, test_X, test_Y


# Load data
train_X, train_Y, test_X, test_Y = load_2D_dataset()

# Scatter
plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral)
plt.show()

# Hyper-Parameters
lr = 0.3
dropout = 0.86
epochs = 30000
lambd = 0

# architecture - not quite needed as a Class, but ok...
list = [(20, relu, 1.0), (3, relu, dropout), (1, sigmoid, dropout)]
architecture = Architecture.from_list(list)

# model - a class that saves the state of the params, and you can run multiple times
dnn = DNNModel(train_X.shape[0], architecture)

# train, in 2 parts
costs1 = dnn.train(train_X, train_Y, lr_decay=0.01, lambd=lambd, epochs=epochs//2, learning_rate=lr)
costs2 = dnn.train(train_X, train_Y, lr_decay=0.01, lambd=lambd, epochs=epochs//2, learning_rate=lr)
costs = costs1 + costs2

print(f"Accuracy on the training set: {dnn.accuracy(train_X, train_Y)}")
print(f"Accuracy on the test set: {dnn.accuracy(test_X, test_Y)}")

# plot the cost
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (x1,000)')
plt.title(f"Learning rate = {lr}")
plt.show()

# visualize decision boundary
plt.title("Visualize Decision Boundary")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: dnn.predict(x.T), train_X, train_Y)