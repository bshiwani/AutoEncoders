
from network3 import sigmoid, tanh, ReLU, Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, load_data_shared_STL_grayscale
from sklearn import cross_validation

training_data, test_data = load_data_shared_STL_grayscale()

def CNN(training_data, test_data = None, epochs = 20, mini_batch_size = 20):
    print "CNN"
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 96, 96),
                          filter_shape=(16, 1, 5, 5),
                          poolsize=(2, 2),
                          activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 16, 46, 46),
                          filter_shape=(16, 16, 5, 5),
                          poolsize=(2, 2),
                          activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 16, 21, 21),
                          filter_shape=(16, 16, 4, 4),
                          poolsize=(2, 2),
                          activation_fn=ReLU),
        FullyConnectedLayer(n_in=16*9*9, n_out=500, activation_fn=ReLU),
        FullyConnectedLayer(n_in=500, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)],
        mini_batch_size= mini_batch_size)
    net.SGD(training_data, epochs, mini_batch_size, 0.1, validation_data = test_data)


def CV_run(training_data):
    cv = cross_validation.KFold(len(training_data), n_folds=10, shuffle=True)
    for trainCV, testCV in cv:
        cv_train = map(training_data.__getitem__, trainCV)
        cv_test = map(training_data.__getitem__, testCV)
        CNN(cv_train, cv_test)

def testModel(training_data, test_data = None, epochs = 20, mini_batch_size = 16):
    net = Network([
        FullyConnectedLayer(n_in=9216, n_out=1000, activation_fn=ReLU),
         FullyConnectedLayer(n_in=1000, n_out=500, activation_fn=ReLU),
         FullyConnectedLayer(n_in=500, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)],
        mini_batch_size= mini_batch_size)
    net.SGD(training_data, epochs, mini_batch_size, 0.05, validation_data = test_data)

CNN(training_data = training_data, test_data = test_data)
