import matplotlib.pyplot as plt
import numpy as np
from myNeuralNetwork import MyNeuralNetwork

x_num, y_num = 200, 200
x_range = np.linspace(-5, 5, x_num)
y_range = np.linspace(-5, 5, y_num)

def funcDampenedBubbleWrap(x, y):
    denom = np.sin(x) + np.cos(y)
    numer = (x**2) + (y**2) + 1
    return denom / numer

def sampleFunc(x, y):
    return (np.sin(x*y) - np.cos(y)) / ((x**2 * y**2) + 1)

def xor(x, y):
    x = np.where(x>0, 1, 0)
    y = np.where(y>0, 1, 0)
    xor = np.logical_xor(x, y)
    return np.where(xor==False, 0, 1)

X, Y = np.meshgrid(x_range, y_range)
Z = sampleFunc(X, Y)

inputs = np.vstack((X.ravel(), Y.ravel()))
outputs = np.atleast_2d(Z.ravel())

nnDeep = MyNeuralNetwork(2, 1, 5, 20, mode="continuous")
nnWide = MyNeuralNetwork(2, 1, 5, 20, mode="continuous")

learningRate = 0.5
trainX, trainY = nnDeep.setupTraining(inputs, outputs, learningRate) # trainX and trainY are scrambled, inputs/outputs for nn NOT x y on graph
nnWide.setupTraining(inputs, outputs, learningRate)

#nn.train(inputs, outputs, learning_rate=[learningRate])

# Train each epoch manually to see progres
numEpochs = 100
graphEpochRate = 20
for epoch in range(numEpochs):
    nnDeep.run_epoch(trainX, trainY, batch_size=(trainX.shape[1])//50, alpha_momentum=0.4)
    nnWide.run_epoch(trainX, trainY, batch_size=(trainX.shape[1])//50)
    if ((epoch%graphEpochRate==0 or epoch==0) and graphEpochRate!=-1):
        # Plot underlying function and neural network output
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z, color='red', linewidth=0.5, rstride=6, cstride=6) 
        Z_deep_nn = nnDeep.get_yhat(inputs)[0].reshape(x_num, y_num)
        ax.plot_wireframe(X, Y, Z_deep_nn, color="blue", linewidth=0.5, rstride=3, cstride=3) 
        #Z_wide_nn = nnWide.get_yhat(inputs)[0].reshape(x_num, y_num)
        #ax.plot_wireframe(X, Y, Z_wide_nn, color="green", linewidth=0.5, rstride=3, cstride=3) 
        plt.show()