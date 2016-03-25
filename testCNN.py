import cPickle
import theano
import numpy as np
import theano.tensor as T

import network3
from network3 import sigmoid, tanh, ReLU, Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

mini_batch_size = 16
training_data,validation_data,test_data = network3.load_data_shared_notMNIST("data/notMNIST_10000.pickle")

def udacity_convolutions(n=3, epochs=60):
    for j in range(n):
        print "udacity convolutions"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                          filter_shape=(16, 1, 2, 2),
                          subsample=(2, 2),
                          poolsize=(1, 1),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 16, 14, 14),
                          filter_shape=(16, 16, 2, 2),
                          subsample=(2, 2),
                          poolsize=(1, 1),
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=16*7*7, n_out=64, activation_fn=ReLU),
            SoftmaxLayer(n_in=64, n_out=10)], mini_batch_size)
        net.SGD(training_data, epochs, mini_batch_size, 0.05, validation_data, test_data=test_data)

def udacity_convolutions_problem1(n=3, epochs=60):
    for j in range(n):
        print "udacity convolutions problem1"
        net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                          filter_shape=(16, 1, 1, 1),
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 14, 14),
                          filter_shape=(16, 16, 1, 1),
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            FullyConnectedLayer(n_in=16*7*7, n_out=64, activation_fn=ReLU),
            SoftmaxLayer(n_in=64, n_out=10)], mini_batch_size)
        net.SGD(training_data, epochs, mini_batch_size, 0.05, validation_data, test_data=test_data)


udacity_convolutions(1, 10)






