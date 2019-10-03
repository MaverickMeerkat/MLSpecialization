import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers, Input, Model

from Course1.Week2.lr_utils import load_dataset, process_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
plt.show()
print(f"y = {train_set_y[:, index]}, it's a '{classes[np.squeeze(train_set_y[:, index])].decode('utf-8')}' picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x, test_set_x = process_dataset(train_set_x_orig, test_set_x_orig)
train_set_y = train_set_y.T
test_set_y = test_set_y.T

# The original exercise is creating logistic regression from the grounds (using only numpy)
# Here I will use Keras sequential and functional API

# There is no regular gradient descent in Keras, so I will use Adam
adam = optimizers.Adam(lr=0.009)

# Keras Sequential
def sequential(input_shape):
    model = Sequential([
        Dense(1, input_shape=input_shape, activation='sigmoid'),
    ])
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = sequential((train_set_x.shape[1],))
model.fit(train_set_x, train_set_y, epochs=200, verbose=2)
Y_prediction_train = (model.predict(train_set_x) > 0.5) * 1
Y_prediction_test = (model.predict(test_set_x) > 0.5) * 1
_, train_score = model.evaluate(train_set_x, train_set_y)
_, test_score = model.evaluate(test_set_x, test_set_y)
print(f'train accuracy: {train_score}; test accuracy: {test_score}')


# Keras Functional
def functional(shape):
    inputs = Input(shape=shape)
    output = Dense(1, activation='sigmoid')(inputs)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = functional((train_set_x.shape[1],))
history = model.fit(train_set_x, train_set_y, epochs=100, verbose=2)  # batch_size = m_train
Y_prediction_train = (model.predict(train_set_x) > 0.5) * 1  # *1 to convert boolean to int
Y_prediction_test = (model.predict(test_set_x) > 0.5) * 1
_, train_score = model.evaluate(train_set_x, train_set_y)
_, test_score = model.evaluate(test_set_x, test_set_y)
print(f'train accuracy: {train_score}; test accuracy: {test_score}')


# Example of a picture that was wrongly classified.
index = 5
plt.imshow(test_set_x[index, :].reshape((num_px, num_px, 3)))
plt.show()
print(f"y = {train_set_y[index, :]}, it's a '{classes[np.squeeze(train_set_y[index, :])].decode('utf-8')}' picture.")
print(f"y_hat = {Y_prediction_test[index]}, it's a '{classes[np.squeeze(Y_prediction_test[index])].decode('utf-8')}' picture.")

# plot cost function over iterations
plt.plot(history.history['loss'])
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title(f"Learning rate = 0.009")
plt.show()
