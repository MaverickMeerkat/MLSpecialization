# MLSpecialization

Coursera Machine Learning Specialization (Andrew Ng) course exercises

https://www.coursera.org/specializations/deep-learning

Note that due to the explicit request in the Code of Honor not to post solutions on GitHub, I will not (as of now)
post the solutions for the graded exercises - but will post rather related/adapted stuff, or solutions for the ungraded
optional parts.

The graded exercises themselves are quite easy, so it shouldn't be much of a problem - and you can use the forums or
discussion boards to ask for help.

## Course 1 - Neural Networks and Deep Learning

Week 2 - used both Keras sequential and functional API to implement logistic regression

Week 3 - used TensorFlow to implement a regular NN

Week 4 - used a variation of the DNN shown in the exercise

## Course 2 - Improving Deep Neural Networks
###  Hyperparameter tuning, Regularization and Optimization

Week 1 - Uses OOP design, which allows for both L2 regularization and Dropout; Unlike exercise, both are supported
simultaneously

Week 2 - Extended the DNNClass in week 1 to support mini batches, momentum and RMS prop (aka Adam = Momentum + RMS)

Week 3 - In the first part I used Hyperparameter search; in the 2nd part I extended DNNClass to support batch
normalization

## Course 3 - Structuring Machine Learning Projects

TODO: think of possible ways to implement the learning material

## Course 4 - Convolutional Neural Networks (CNN's)

Week 1 - the 1st exercise implements conv-nets forward prop and (optional) back prop, and then the 2nd exercise uses
tensorflow to actually run the network. Instead - I implemented the entire network just with numpy. It runs really slow,
but the logic should be correct.

A particular point of interest might be the softmax gradient function (softmax_back). Most implementations out there I
saw jump immediately to the Loss gradient after the softmax, i.e. to dZ, which is `a - y`. This is a heuristic. Here I
calculated the entire derivative. The main challange is that the derivative is a matrix per row, i.e. we are moving from
matrices to tensors.

The general design can probably be highly improved, but this is just a learning exercise.

Week 2 - TODO

Week 3 - TODO

Week 4 - TODO

## Course 5 - Sequence Models (RNN's)

Week 1 - TODO

Week 2 - TODO

Week 3 - TODO