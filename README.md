# neural-network
Build various neural network models writing all the layers, gradients of weights from scratch.

Main aim is to learn Tensorflow and Convolutional Neural Networks by implementing forward and backward propagation in CNN using Tensorflow from scratch.

I will implement various layers of CNN and then check the effects of the presence or absence of various layers over the accuracy on different datasets.

Datasets I have used are CALTECH 101, MNIST.

## About Code

Normal Neural Network is ready. You can add as many fully connected layers as required. I have also implemented a loss layer with softmax cross entropy loss.

The files in this folder are for implementation of Convolutional Neural Network using Tensorflow without using Automatic Differentiation

The entire code is written using **Python 3.6**

The code using Automatic Differntiation of Tensorflow is in **auto_diff** folder.

## How To Run

To see how to use my implementation of neural network, see the **classify_iris.py** file which provides the illustration to form and connect various layers, read dataset, and use my implementation of gradient descent.

The score shown at the end of the file is the accuracy score

Run classify_iris.py file using `python classify_iris.py`