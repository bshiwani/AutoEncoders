import network2
import notmnist_loader
import numpy as np

net = network2.load("models/DenoiseModelAJ")

print np.shape(net.weights[0]), np.shape(net.weights[1]), np.shape(net.biases[0])

train_data, test_data, train_labels, test_labels = notmnist_loader.add_noise2_loader()

def getOutputOfHiddenUnits(data):
    output = []
    for i in xrange(len(data)):
        x = data[i][0]
        y = data[i][1]
        output.append(np.dot(net.weights[0], x) + net.biases[0])
    return output

def getNewDataset(train_data, test_data, train_labels, test_labels):
    train_outputOfHiddenUnits = getOutputOfHiddenUnits(train_data)
    test_outputOfHiddenUnits = getOutputOfHiddenUnits(test_data)

    train_results = [network2.vectorized_result(y) for y in train_labels]

    training = zip(train_outputOfHiddenUnits, train_results)
    testing = zip(test_outputOfHiddenUnits, test_labels)

    return training, testing

newNet = network2.Network([500, 100, 10])
new_train_data, new_test_data = getNewDataset(train_data, test_data, train_labels, test_labels)

evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy = newNet.SGD(training_data=new_train_data, epochs=10,
                                                          mini_batch_size=50, eta=0.1,lmbda=0.0,
                                                          evaluation_data=new_test_data,
                                                          monitor_evaluation_accuracy=True,
                                                          monitor_evaluation_cost=True,
                                                          monitor_training_accuracy=True,
                                                          monitor_training_cost=True)

print "training error is {0}\n test error is {1}".format(training_cost[-1], evaluation_cost[-1])

