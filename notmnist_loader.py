"""
notmnist_loader
~~~~~~~~~~~~~~~
A library to load the notMNIST image data, which is .pickle file
"""

#### Libraries
# Standard library
import cPickle

# Third-party libraries
import numpy as np

def load_new_test():
    f = open("data/notMNIST_10000_test.pickle", "rb")
    data = cPickle.load(f)
    f.close()
    test_dataset = data.get("test_dataset")
    test_labels = data.get("test_labels")
    return test_dataset + 0.5, test_labels

def load_new_dataset():
    test_dataset,test_labels = load_new_test()

    test_dataset = test_dataset
    train_inputs = test_dataset[:9000]
    test_inputs = test_dataset[9000:]
    new_train_inputs = [np.reshape(x, (784, 1)) for x in train_inputs]
    new_test_inputs = [np.reshape(x, (784, 1)) for x in test_inputs]
    train_data = zip(new_train_inputs, new_train_inputs)
    test_data = zip(new_test_inputs, new_test_inputs)

    new_train_labels = test_labels[:9000]
    new_test_labels = test_labels[9000:]

    return train_data, test_data, new_train_labels, new_test_labels

def load_full_data():
    f = open("data/notMNIST_10000.pickle", "rb")
    data = cPickle.load(f)
    f.close()
    train_dataset = data.get("train_dataset")
    train_labels = data.get("train_labels")
    test_dataset = data.get("test_dataset")
    test_labels = data.get("test_labels")
    valid_dataset = data.get("valid_dataset")
    valid_labels = data.get("valid_labels")
    return train_dataset + 0.5, train_labels,\
           test_dataset + 0.5, test_labels, \
           valid_dataset + 0.5, valid_labels

def load_data():
    f1 = open("data/B.pickle", "rb")
    f2 = open("data/test/B.pickle", "rb")
    training_data = cPickle.load(f1)
    test_data = cPickle.load(f2)
    f1.close()
    f2.close()
    return training_data + 0.5,test_data + 0.5
    # return training_data,test_data

def load_data_for_classifier_noisy():
    train_dataset, train_labels,\
           test_dataset, test_labels, \
           valid_dataset, valid_labels = load_full_data()
    train_inputs = [np.reshape(x, (784, 1)) for x in train_dataset]
    test_inputs = [np.reshape(x, (784, 1)) for x in test_dataset]
    valid_inputs = [np.reshape(x, (784, 1)) for x in valid_dataset]
    train_results = [vectorized_result(y) for y in train_labels]

    for i in xrange(len(train_inputs)):
        for j in xrange(len(train_inputs[i])):
            #print training_data[i][j]
            if train_inputs[i][j] >= 0.3:
                train_inputs[i][j] = train_inputs[i][j] * np.random.normal(0.75, 0.15)
            else:
                train_inputs[i][j] = train_inputs[i][j] + np.random.uniform(0,0.6)
    train = zip(train_inputs, train_results)
    valid = zip(valid_inputs, valid_labels)
    test = zip(test_inputs, test_labels)

    return train, valid, test

def load_data_wrapper():
    train_dataset, train_labels,\
           test_dataset, test_labels, \
           valid_dataset, valid_labels = load_full_data()
    training = [np.reshape(x, (784, 1)) for x in train_dataset]
    test = [np.reshape(np.array(x), (784, 1)) for x in test_dataset]
    training_data = zip(training, training)
    test_data = zip(test, test)
    return training_data,test_data, train_labels, test_labels

def add_noise_loader():
    """
    Add noises to original training data
    :return:
    """
    training_data,test_data = load_data()
    training = [np.reshape(x, (784, 1)) for x in training_data]
    test = [np.reshape(np.array(x), (784, 1)) for x in test_data]
    noise_data = [np.multiply(x, np.random.random_integers(7, 10, (784,1))/10.0) for x in training]
    noise_test = [np.multiply(x, np.random.random_integers(7, 10, (784,1))/10.0) for x in test]
    # training_inputs = np.concatenate((training, noise_data))
    # training_outputs = np.concatenate((training, training))
    training_data = zip(noise_data, training)
    test_data = zip(noise_test, test)
    return training_data,test_data

def add_noise2_loader():
    train_dataset, train_labels,\
           test_dataset, test_labels, \
           valid_dataset, valid_labels = load_full_data()
    # training_data,test_data = load_data()
    training = [np.reshape(x, (784, 1)) for x in train_dataset]
    valid = [np.reshape(x, (784, 1)) for x in valid_dataset]
    test = [np.reshape(x, (784, 1)) for x in test_dataset]

    noisy_data = np.copy(training)
    for i in xrange(len(noisy_data)):
        for j in xrange(len(noisy_data[i])):
            #print training_data[i][j]
            if noisy_data[i][j] >= 0.3:
                noisy_data[i][j] = noisy_data[i][j] * np.random.normal(0.75, 0.15)
            else:
                noisy_data[i][j] = noisy_data[i][j] + np.random.uniform(0,0.6)

    train_data = zip(noisy_data, training)
    valid_data = zip(valid, valid)
    test_data = zip(test, test)
    return train_data, valid_data, test_data, train_labels,valid_labels, test_labels

def load_test_data():
    train_dataset, train_labels,\
           test_dataset, test_labels, \
           valid_dataset, valid_labels = load_full_data()
    test_inputs = [np.reshape(x, (784, 1)) for x in test_dataset]

    return zip(test_inputs, test_labels)

def load_SDE_data(isNoisy = False):
    train_dataset, train_labels,\
           test_dataset, test_labels, \
           valid_dataset, valid_labels = load_full_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in train_dataset]
    test_inputs = [np.reshape(x, (784, 1)) for x in test_dataset]
    valid_inputs = [np.reshape(x, (784, 1)) for x in valid_dataset]

    if isNoisy:
        training_inputs_noisy = np.copy(training_inputs)
        for i in xrange(len(training_inputs_noisy)):
            for j in xrange(len(training_inputs_noisy[i])):
                #print training_data[i][j]
                if training_inputs_noisy[i][j] >= 0.3:
                    training_inputs_noisy[i][j] = training_inputs_noisy[i][j] * np.random.normal(0.75, 0.15)
                else:
                    training_inputs_noisy[i][j] = training_inputs_noisy[i][j] + np.random.uniform(0,0.6)

        training_inputs = [training_inputs, training_inputs_noisy]

    return training_inputs, train_labels, \
                test_inputs, test_labels, \
                valid_inputs, valid_labels




def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# training_data,test_data = add_noise_loader()
#
# print np.shape(test_data)

# train_dataset, train_labels,\
#            test_dataset, test_labels, \
#            valid_dataset, valid_labels = load_full_data()
#
# print np.shape(train_dataset)

# train_data, test_data, train_labels, test_labels = add_noise2_loader()
#
# print train_data[0][0][:10], train_data[0][1][:10]

# train_data, test_data, train_labels, test_labels = load_new_dataset()
#
# print len(test_labels), len(train_labels)


