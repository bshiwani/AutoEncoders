import notmnist_loader
import network2
import numpy as np
from sklearn import cross_validation

def getOutputOfHiddenUnits(data, net):
    output = []
    for x,y in data:
        output.append(network2.sigmoid(np.dot(net.weights[0], x) + net.biases[0]))
    return output

def getIntermediateDataset(train_data, test_data, net):
    """
    get dataset for intermediate layer
    :param train_data:
    :param test_data:
    :param net:
    :return:
    """
    train_outputOfHiddenUnits = getOutputOfHiddenUnits(train_data, net)
    test_outputOfHiddenUnits = getOutputOfHiddenUnits(test_data, net)

    # train_results = [network2.vectorized_result(y) for y in train_labels]

    training = zip(train_outputOfHiddenUnits, train_outputOfHiddenUnits)
    testing = zip(test_outputOfHiddenUnits, test_outputOfHiddenUnits)

    return training, testing

def getLastDataset(train_data, test_data, train_labels, test_labels, net):
    """
    get dataset for last layer
    :param train_data:
    :param test_data:
    :param train_labels:
    :param test_labels:
    :param net:
    :return:
    """
    train_outputOfHiddenUnits = getOutputOfHiddenUnits(train_data, net)
    test_outputOfHiddenUnits = getOutputOfHiddenUnits(test_data, net)

    train_results = [network2.vectorized_result(y) for y in train_labels]

    training = zip(train_outputOfHiddenUnits, train_results)
    test = zip(test_outputOfHiddenUnits, test_labels)
    return training, test


def crossValidation(training_data):
    """
    cross validation
    :param training_data:
    :return:
    """
    cv = cross_validation.KFold(len(training_data), n_folds=10, shuffle=True)
    cv_total_cost = []
    for trainCV, testCV in cv:
        print "new fold"
        validation_training = map(training_data.__getitem__, trainCV)
        validation_test = map(training_data.__getitem__, testCV)

        net = network2.Network([784, 300, 784], cost=network2.QuadraticCost)
        evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy = net.SGD(training_data=validation_training, epochs=5,mini_batch_size=50,
                eta=0.1, lmbda = 0.0, evaluation_data = validation_test,monitor_evaluation_cost=True)
        cv_total_cost.append(evaluation_cost[-1])

    print "CV_MSE = {0}".format(np.mean(cv_total_cost))

def testAutoEncoder(training_data, test_data, save_fileName, sizes, epochs, mini_batch_size, eta, lmbda=0.0):
    """
    Test autoEncoder
    :return:
    """
    net = network2.Network(sizes, cost=network2.QuadraticCost)

    evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy = net.SGD(training_data=training_data, epochs=20,mini_batch_size=10,
        eta=0.1, lmbda = 0.0,evaluation_data=test_data, monitor_training_cost=True, monitor_evaluation_cost=True, isAutoEncoder=True)


    #The following line save the settings of the net to be a json file,
    #we can load it to reconstruct our net
    net.save(save_fileName)
    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

# training_data,test_data = notmnist_loader.load_data_wrapper()

### load original data
train_data,valid_data, test_data, \
train_labels,valid_labels, test_labels = notmnist_loader.add_noise2_loader()

### the first step. after you finished training this step, you can comment the folliwing line out
testAutoEncoder(train_data, valid_data,save_fileName= "models/SAE_step1", sizes= [784, 500, 784],
                epochs=20, mini_batch_size=10, eta=0.1, lmbda=0.0)


net1 = network2.load("models/SAE_step1")
train_data1, valid_data1 = getIntermediateDataset(train_data[:1000], valid_data[:1000], net1)

### the second step, after you finished training this step, you can comment the folliwing line out
testAutoEncoder(train_data1, valid_data1,save_fileName= "models/SAE_step2", sizes= [500, 100, 500],
                epochs=20, mini_batch_size=10, eta=0.1, lmbda=0.0)

net2 = network2.load("models/SAE_step2")
train_data2, valid_data2 = getLastDataset(train_data1, valid_data1, train_labels, valid_labels, net2)

### classification model
net3 = network2.Network([100, 10],cost=network2.CrossEntropyCost)
net3.SGD(train_data2, epochs=20, mini_batch_size=10, eta=0.1, lmbda=0.0,
         evaluation_data=valid_data2,
         monitor_training_cost=True, monitor_evaluation_cost=True,
         monitor_training_accuracy=True, monitor_evaluation_accuracy=True)
net3.save("models/SAE_step3")




