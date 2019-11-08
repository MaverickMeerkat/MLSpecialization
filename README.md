# MLSpecialization

Coursera Machine Learning Specialization (Andrew Ng) course exercises

https://www.coursera.org/specializations/deep-learning

Note that due to the explicit request in the Code of Honor not to post solutions on GitHub, I will not (as of now)
post the solutions for the graded exercises - but will post rather related/adapted stuff, or solutions for the ungraded
optional parts.

The graded exercises themselves are quite easy, so it shouldn't be much of a problem - and you can use the forums or
discussion boards to ask for help.

### Conventions
Every week will consist of a main "run.py" file which runs the code on some data. In addition there might be other
modules with classes or functions that are needed to run the file.

## Course 1 - Neural Networks and Deep Learning

Week 2 - Uses both Keras sequential and functional API to implement logistic regression.

Week 3 - Uses TensorFlow to implement a regular NN.

Week 4 - Uses a variation of the DNN shown in the exercise.

## Course 2 - Improving Deep Neural Networks
###  Hyperparameter tuning, Regularization and Optimization

Week 1 - Uses OOP design, which allows for both L2 regularization and Dropout; Unlike exercise, both are supported
simultaneously.

Week 2 - Extended the DNNClass in week 1 to support mini batches, momentum and RMS prop (aka Adam = Momentum + RMS).

Week 3 - In the first part I used Hyperparameter search; in the 2nd part I extended DNNClass to support batch
normalization.

## Course 3 - Structuring Machine Learning Projects

TODO: think of possible ways to implement the learning material

## Course 4 - Convolutional Neural Networks (CNN's)

Week 1 - The original exercise implements conv-nets forward prop and (optional) back prop, and then uses tensorflow to
actually run the network. Instead - I implemented the entire network just with numpy. It runs really slow, but the logic
should be correct.

A particular point of interest might be the softmax gradient function (softmax_back). Most implementations out there I
saw jump immediately to the Loss gradient after the softmax, i.e. to dZ, which is `a - y`. This is a heuristic. Here I
calculated the entire derivative. The main challange is that the derivative is a matrix per row, i.e. we are moving from
matrices to tensors.

The general design can probably be highly improved, but this is just a learning exercise.

Week 2 - Implements ResNets in TensorFlow instead of Keras.

Note an important confusion: specifying Batch Normalization axis in Keras, is opposite to how it works in NumPy. The
axis you specify in Keras is actually the axis which is not in the calculations!

Work in Progress...

Week 3 - WIP

Download yolo weights from https://pjreddie.com/darknet/yolo/

Week 4 - combined NST with FaceNet. Download the pre-trained model:
https://github.com/nhbond/facenet-resources/tree/master/models

Note that FaceNet is used to create embedding (some latent space representation) of faces. What I tried to do is combine
this with Neural-Style-Transfer, and called it Ultrame (the idea is to turn all faces more into me).

Ultrame1 is a bit of a messy notebook, but Ultrame2 is more detailed.

So far results have been quite disappointing. But this was just a few hours of work.

## Course 5 - Sequence Models (RNN's)

Week 1 - TODO

Week 2 - TODO

Week 3 - TODO