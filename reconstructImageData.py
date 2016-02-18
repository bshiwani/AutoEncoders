import network2
import notmnist_loader
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

# Load the test data
training_data,test_data = notmnist_loader.add_noise2_loader()
#training_data,test_data = notmnist_loader.load_data_wrapper()

#Load the trained NN model
net = network2.load("models/nets_model_Adenoise")

def getFeaturesData(weight):
    """
    get feature data for each hidden unit
    :param weight: a vector of weights, eg.weights[0]
    :return:
    """
    data = []
    #Matrix = [[0 for x in range(10)] for x in range(10)]
    n = len(weight)
    divisors = np.sqrt(np.sum(weight**2, 1))
    for i in xrange(n):
        data.append(weight[i]/divisors[i])
        # Matrix[][i] = (np.reshape(weight[i]/divisors[i], (28, 28)))

    return data

def reconstructImage(data, fileName, dimensions=[28, 28]):
    """
    reconstruct image from greyscale data
    :param data: a vector of output
    :param fileName: full file name (including path)
    :param dimensions: eg. [28, 28] means 28*28
    :return:
    """
    imageData = np.round(np.reshape((data-0.5) * 255.0 + 255.0/2, (dimensions[0], dimensions[1])))
    #data = np.reshape(data, dimensions[0], dimensions[1])
    misc.imsave(fileName, imageData)
    #plt.title(fileName.split("/")[-1])
    #plt.imshow(imageData, cmap=plt.cm.gray)
    #plt.show()

#print len(net.weights[1])


#Reconstruct Specific Images from TestData
outputData=[]
disp=[2,83,100,150,800,900,1000,1800]
for kk in range(len(disp)):
    i=disp[kk]
    temp=[[]]
    reconstructImage(test_data[i][0], "data/test_image"+str(i+1)+".png", [28, 28])
    temp=test_data[i][0]
    outputData.append(net.feedforward(temp))
    reconstructImage(outputData[kk], "data/recon_image"+str(i+1)+".png", [28, 28])
    weights = net.weights
    featuresData = getFeaturesData(weights[0])
    reconstructImage(featuresData[kk], "data/feature"+str(i+1)+"_0.png", [28, 28])
    plt.subplot(3, 10, kk+1)
    A=plt.imread('data/test_image'+str(i+1)+'.png')
    plt.imshow(A,cmap=plt.cm.gray)
    plt.subplot(3, 10, kk+11)
    A=plt.imread('data/feature'+str(i+1)+'_0.png')
    plt.imshow(A,cmap=plt.cm.gray)
    plt.subplot(3, 10, kk+21)
    A=plt.imread('data/recon_image'+str(i+1)+'.png')
    plt.imshow(A,cmap=plt.cm.gray)
plt.show()

