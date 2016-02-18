# Standard library
import cPickle
#import reconstructImageData
# Third-party libraries
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

#Noise Model 2: Multiplicative as well as additive Noise
def add_noise2_loader():
    train_dataset, train_labels,\
           test_dataset, test_labels, \
           valid_dataset, valid_labels = load_full_data()
    training = [np.reshape(x, (784, 1)) for x in train_dataset]
    test = [np.reshape(np.array(x), (784, 1)) for x in test_dataset]
    noisy_data = np.copy(training)
    noisy_test_data = np.copy(test)
    for i in xrange(len(noisy_data)):
        for j in xrange(len(noisy_data[i])):
            if noisy_data[i][j] >= 0.3:
                noisy_data[i][j] = noisy_data[i][j] * np.random.normal(0.75, 0.15)
            else:
                noisy_data[i][j] = noisy_data[i][j] + np.random.uniform(0,0.6)
    train = zip(noisy_data, training)
    for i in xrange(len(noisy_test_data)):
        for j in xrange(len(noisy_test_data[i])):
            if noisy_test_data[i][j] >= 0.3:
                noisy_test_data[i][j] = noisy_test_data[i][j] * np.random.normal(0.75, 0.15)
            else:
                noisy_test_data[i][j] = noisy_test_data[i][j] + np.random.uniform(0,0.6)
    test = zip(test, test)
    return train,test
