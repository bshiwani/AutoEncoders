import notmnist_loader
import network2
import numpy as np
from sklearn import cross_validation

def crossValidation(training_data):
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

def testAutoEncoder():
    """
    Test autoEncoder
    :return:
    """
    net = network2.Network([784, 100, 784], cost=network2.QuadraticCost)

    evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy = net.SGD(training_data=training_data, epochs=20,mini_batch_size=10,
        eta=0.1, lmbda = 0.0,evaluation_data=test_data, monitor_training_cost=True, monitor_evaluation_cost=True)


    #The following line save the settings of the net to be a json file,
    #we can load it to reconstruct our net
    net.save("models/nets_model_1_100_B")
    return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

# training_data,test_data = notmnist_loader.load_data_wrapper()

training_data,test_data, train_labels, test_labels = notmnist_loader.add_noise2_loader()

crossValidation(training_data)

# evaluation_cost, evaluation_accuracy, \
#             training_cost, training_accuracy = testAutoEncoder()
#
# print "training error is {0}\n test error is {1}".format(training_cost[-1], evaluation_cost[-1])

