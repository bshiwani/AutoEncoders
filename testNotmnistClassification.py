import network2
import notmnist_loader

training_data,valid_data, test_data = notmnist_loader.load_data_for_classifier_noisy()

net = network2.Network([784, 500, 100, 10])

net.SGD(training_data=training_data[:1000], epochs=20,mini_batch_size=10,
        eta=0.1, lmbda = 0.0,evaluation_data=valid_data[:1000], monitor_training_cost=True,
        monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_evaluation_accuracy=True)

