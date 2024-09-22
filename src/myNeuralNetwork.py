import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def pack(Ws, bs):
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

def softmax(z):
    softmax = np.exp(z)
    # Factors to apply to each element to normalize to "percentage" (each column is same value repeated)
    softMaxDivisionFactors = (np.reciprocal(np.sum(softmax, axis=0))).reshape(-1, z.shape[1])
    softMaxDivisionFactors =  np.repeat(softMaxDivisionFactors, z.shape[0], axis=0)
    softmax = softmax * softMaxDivisionFactors
    return softmax

def linear(z):
    return z

def relu(z):
    return np.where(z <= 0, 0, z)

def relu_prime(z):
    return np.where(z <= 0, 0, 1)

def L2Regularization(Ws, n, alpha=0):
    return 0.5*alpha*sum([np.sum(w*w) for w in Ws]) / n

def L2RegularizationPrime(weightLayer, n, alpha=0):
    return (alpha/n)*(weightLayer)

class MyNeuralNetwork():

    def __init__(self, num_input, num_output, num_hidden_layers, hidden_layer_size,
                 mode):
        self.num_hidden_layers = num_hidden_layers
        self.num_input = num_input
        self.hidden_layer_size = hidden_layer_size
        self.num_output = num_output
        self.num_hidden = int(self.num_hidden_layers) * [self.hidden_layer_size]

        # Gradient Descent Hyperparameters
        self.momentum_v = 0
        self.epochLearningRate = 0 # will be set later

        if (mode=="categorical"):
            self.lossFunc = self.fCE
            self.gradFunc = self.gradientFunc
            self.finalLayerActivationFunc = softmax
        elif (mode=="continuous"):
            self.lossFunc = self.mse
            self.gradFunc = self.gradientFunc
            self.finalLayerActivationFunc = linear
        else:
            raise "Valid mode from [categorical, continuous] not given"

    def unpack(self):
        # Unpack arguments
        Ws = []

        # Weight matrices
        start = 0
        end = self.num_input*self.num_hidden[0]
        W = self.weights[start:end]
        Ws.append(W)

        # Unpack the weight matrices as vectors
        for i in range(self.num_hidden_layers - 1):
            start = end
            end = end + self.num_hidden[i]*self.num_hidden[i+1]
            W = self.weights[start:end]
            Ws.append(W)

        start = end
        end = end + self.num_hidden[-1]*self.num_output
        W = self.weights[start:end]
        Ws.append(W)

        # Reshape the weight "vectors" into proper matrices
        Ws[0] = Ws[0].reshape(self.num_hidden[0], self.num_input)
        for i in range(1, self.num_hidden_layers):
            # Convert from vectors into matrices
            Ws[i] = Ws[i].reshape(self.num_hidden[i], self.num_hidden[i-1])
        Ws[-1] = Ws[-1].reshape(self.num_output, self.num_hidden[-1])

        # Bias terms
        bs = []
        start = end
        end = end + self.num_hidden[0]
        b = self.weights[start:end]
        bs.append(b)

        for i in range(self.num_hidden_layers - 1):
            start = end
            end = end + self.num_hidden[i+1]
            b = self.weights[start:end]
            bs.append(b)

        start = end
        end = end + self.num_output
        b = self.weights[start:end]
        bs.append(b)

        return Ws, bs

    # feeds forward on network
    def get_yhat_one_input(self, inputs): # TODO delete this after get_yhat works for array
        Ws, bs = self.unpack()
        zs = []
        zs.append(inputs)
        layerValues = inputs # to feed forward values for current layer
        for layer in range(self.num_hidden_layers+1):
            z = np.matmul(layerValues, Ws[layer].T) + bs[layer]
            zs.append(z)
            layerValues = self.relu(z) # applying non linear activation function

        return layerValues, zs

    def get_yhat(self, inputs):
        Ws, bs = self.unpack()
        zs = []
        zs.append(inputs) # each column is an input
        layerValues = inputs # to feed forward values for current layer
        for layer in range(self.num_hidden_layers+1):
            z = bs[layer].reshape(-1, 1) + Ws[layer] @ layerValues
            zs.append(z)
            if (layer!=self.num_hidden_layers): # Don't apply relu to final layer
                layerValues = relu(z)
            else:
                layerValues = self.finalLayerActivationFunc(z)

        return layerValues, zs

    def fCE(self, X, Y, alpha=0):
        y_hat = self.get_yhat(X)[0]
        ce = -np.sum(Y * np.log(y_hat))

        Ws, bs = self.unpack()
        ce = ce/X.shape[1] # average the loss of the result

        ce += L2Regularization(Ws, X.shape[1], alpha) # Add regularization
        return ce

    def mse(self, X, Y, alpha=0):
        y_hat = self.get_yhat(X)[0]
        mse = np.sum(0.5 * (Y - y_hat) * (Y - y_hat))

        Ws, bs = self.unpack()
        mse = mse/X.shape[1] # average the loss of the result

        mse += L2Regularization(Ws, X.shape[1], alpha) # Add regularization
        return mse
    
    # From Algorithm 6.4 of https://www.deeplearningbook.org/contents/mlp.html
    def gradientFunc(self, X, Y, alpha=0.1):
        out = None
        # TODO make work with numpy on X and Y inputs (still need for loop back through layers)
        wGrads = [] # reset gradients for this example
        bGrads = []

        y_hat, zs = self.get_yhat(X)
        Ws, bs = self.unpack()
        g = (y_hat - Y) # each column is cost for a single input
        for layer in range(self.num_hidden_layers+1, 0, -1):
            if (layer!=self.num_hidden_layers+1):
                g = g * relu_prime(zs[layer]) # over sum of inputs in this layer, transposed for same shape as g, each column still corresponds to one input
            
            # Get previous layer (do not apply activation if previous layer is first layer)
            if (layer==1): # NOTE: Could make a nicer system that handles each layer having its own activation instead of a clunky if statment
                prevH = zs[layer-1]
            else:
                prevH = relu(zs[layer-1]) # Get h value after activation to use, each column is this layer for specific input

            bGrads.append(g.T) # No regularization term, otherwise multiplied by 1
            # Get weight gradent for each input (third dimension to store for each input)
            wGrads.append(g @ prevH.T + L2RegularizationPrime(Ws[layer-1], X.shape[1], alpha))
            # calculate new g for each iinput as columns (jsut Ws[layer-1].T*g repackaged)
            g = Ws[layer-1].T @ g

        wGrads.reverse()
        wGrads = np.concat([e.ravel() for e in wGrads])
        bGrads.reverse()
        bGrads = [np.sum(e, axis=0).flatten() for e in bGrads] # Sum bias gradient before flattening over all inputs

        if (type(out)==type(None)):
            out = pack(wGrads, bGrads)
        else:
            out += pack(wGrads, bGrads) # add element wise

        # Average out gradient from total number of samples
        out = out / X.shape[1]
        return out
        
    # Creates an image representing the first layer of weights (W0).
    def show_W0(self):
        Ws, bs = self.unpack()
        W = Ws[0]
        n = int(self.num_hidden[0] ** 0.5)
        plt.imshow(np.vstack([
            np.hstack([ np.pad(np.reshape(W[idx1*n + idx2,:], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
        ]), cmap='gray'), plt.show()

    def setupTraining(self, X, y, initialLearningRate):
        """
        Sets up training by randomizing weights and biases and ordering 
        of training sets. Does not run any training or epochs.
        Also sets up hyperparameter for momentum v to 0,
        (as well as other hyperparameters for gradient descent in the future)
        """
        Ws, bs = self.initWeightsAndBiases()
        self.weights = pack(Ws, bs)

        reordering_idx = np.random.permutation(np.arange(X.shape[1]))
        X = X[:, reordering_idx]
        y = y[:, reordering_idx]

        self.momentum_v = 0 # initial momentum set to 0
        self.epochLearningRate = initialLearningRate
        return X, y

    def run_epoch(self, X, y, batch_size, alpha=0, decay=1, alpha_momentum=0):
        idx = 0
        while idx+batch_size<X.shape[1]: # Go through batches
            X_batch = X[:, idx:idx+batch_size]
            y_batch = y[:, idx:idx+batch_size]

            grad = self.gradFunc(X_batch, y_batch, alpha)
            self.momentum_v = alpha_momentum * self.momentum_v + self.epochLearningRate * grad
            self.weights = self.weights - self.momentum_v
            idx += batch_size

        # train on last portion of that batch
        X_batch = X[:, idx:]
        y_batch = y[:, idx:]
        # update grad TODO could package this into dunction to not be repeated
        grad = self.gradFunc(X_batch, y_batch, alpha)
        self.momentum_v = alpha_momentum * self.momentum_v + self.epochLearningRate * grad
        self.weights = self.weights - self.momentum_v

        self.epochLearningRate *= decay
        
        print("Learning Rate: " + str(self.epochLearningRate))
        print("Percent Correct: " + str(self.percent_correct_test(X, y)))
        print("Loss: " + str(self.lossFunc(X, y)))
      
    def train_neural_net_on_hp(self, X, y, epochs, batch_size, epsilon, alpha, decay, alpha_momentum=0):
        # reorder x for mini_batch
        self.setupTraining(X, y, epsilon)
        
        for epochNum in range(epochs):
            print("Epoch: " + str(epochNum))
            self.run_epoch(X, y, batch_size, alpha, decay, alpha_momentum) # Run epoch without resetting weights

        return self.weights

    def percent_correct_test(self, X, y):
        y_hat = self.get_yhat(X)[0]
        predictedAnswers = np.argmax(y_hat, axis=0)
        correctAnswers = np.argmax(y, axis=0)

        return len(list(filter(lambda x: x==0, predictedAnswers-correctAnswers))) / len(correctAnswers) # as percentage

    # TODO this whole thing would work better as a class that stores this relevant information
    def train(self, trainX, trainY, valX=None, valY=None, testX=None, testY=None,
            epoch_values = [20],
            minibatch_values = [128],
            learning_rate = [0.03],
            alpha = [0],
            decay = [1],
            alpha_mom = [0.5]): 
        
        combinations = list(product(epoch_values, minibatch_values, learning_rate, alpha, decay, alpha_mom))
        bestWeights, bestHP, bestValidLoss = None, None, None
        for epoch, mini_batch, learning_rate, alpha, decay, alpha_mom in combinations:
            print("Training: " + str([epoch, mini_batch, learning_rate, alpha, decay, alpha_mom]))
            newWeights = self.train_neural_net_on_hp(trainX, trainY, epoch, mini_batch, learning_rate, alpha, decay, alpha_mom)
            print("Training Loss: " + str(self.lossFunc(trainX, trainY)))
            if type(valX)==type(None): # If nothing to validation only do first hps
                bestWeights = newWeights
                break
            nextLoss = self.lossFunc(valX, valY, alpha) 
            print("Validation Loss: " + str(nextLoss))
            if bestValidLoss==None or nextLoss<bestValidLoss: # Update new best 
                bestHP = [epoch, mini_batch, learning_rate, alpha]
                bestWeights, bestValidLoss = newWeights, nextLoss 

        if (type(valX)!=type(None) or type(testX)!=type(None)):
            # Get testing answer
            print("Testing Results")
            print("Best HP: ", bestHP)
            print("Training Loss (Unregularized): ", self.lossFunc(trainX, trainY))
            print("Validation Loss (Unregularized): ", self.lossFunc(valX, valY))
            print("Test Loss (Unregularized): ", self.lossFunc(testX, testY))

        self.weights = bestWeights
        return bestWeights

    def initWeightsAndBiases(self):
        Ws = []
        bs = []

        # Sample each weight from a 0-mean Gaussian with std.dev. of 1/sqrt(numInputs).
        # Initialize biases to small positive number (0.01).

        np.random.seed(0)
        W = 2*(np.random.random(size=(self.num_hidden[0], self.num_input))/self.num_input**0.5) - 1./self.num_input**0.5
        Ws.append(W)
        b = 0.01 * np.ones(self.num_hidden[0])
        bs.append(b)

        for i in range(self.num_hidden_layers - 1):
            W = 2*(np.random.random(size=(self.num_hidden[i], self.num_hidden[i+1]))/self.num_hidden[i]**0.5) - 1./self.num_hidden[i]**0.5
            Ws.append(W)
            b = 0.01 * np.ones(self.num_hidden[i+1])
            bs.append(b)

        W = 2*(np.random.random(size=(self.num_output, self.num_hidden[-1]))/self.num_hidden[-1]**0.5) - 1./self.num_hidden[-1]**0.5
        Ws.append(W)
        b = 0.01 * np.ones(self.num_output)
        bs.append(b)
        return Ws, bs


def to_one_hot(Y, num_categories):
    Y = np.repeat(Y, num_categories).reshape(-1, num_categories).T
    # a matrix to subtract from Y to gets 0 values at desired locations. Has row index for each element
    rows = np.repeat(np.arange(num_categories), Y.shape[1]).reshape(num_categories, -1)
    Y = np.where((Y-rows)==0, 1, 0) # as one hot, where each element with correct row (that got zeroed) now set to one
    return Y
    
def load_data(brightnessChanges=[], horizontalFlip=True):
    # Load data
    X_tr_total = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 784)).T
    y_tr_total = np.load("fashion_mnist_train_labels.npy")

    # Augment data with brightnesses given
    augmentedX, augmentedY = [], []
    if brightnessChanges!=[]:
        for brightnessChange in brightnessChanges:
            if (brightnessChange>0): # If brightening cap at 255 brightness
                augmentedX.append(np.where(X_tr_total+brightnessChange<=255, X_tr_total + brightnessChange, 255)) # Cap output at 255 in brightness
            else: # If darkening, make sure never less than 0
                augmentedX.append(np.where(X_tr_total+brightnessChange>=0, X_tr_total + brightnessChange, 0)) 
            
            augmentedY.append(y_tr_total)

    if (horizontalFlip):
        augmentedX.append(np.flip(X_tr_total.reshape(28, 28, -1), axis=1).reshape(784, -1)) # Add horizontally flipped version
        augmentedY.append(y_tr_total)

    # Combine all augmented data
    X_tr_total = np.hstack((X_tr_total, np.hstack(augmentedX)))
    y_tr_total = np.hstack((y_tr_total, np.hstack(augmentedY))) # IN this case, the y values are always the same, just copied over

    # divide by 255 for fashion data to normalize
    X_tr_total = (X_tr_total / 255) - 0.5

    rand_indexes = np.random.permutation(np.arange(X_tr_total.shape[1])) # get random indexes of number of columns
    # get 20% of the rand_indexes for validation and 80% for for training
    validation_prop = 0.2
    val_indexes = rand_indexes[0:int(len(rand_indexes)*validation_prop)]

    valX = X_tr_total[:, val_indexes]
    valY = y_tr_total[val_indexes]
    valY = to_one_hot(valY, 10)

    testX = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 784)).T
    # divide by 255 for fashion data to normalize
    testX = (testX / 255) - 0.5 
    testY = np.load("fashion_mnist_test_labels.npy")
    testY =  to_one_hot(testY, 10)

    # get NOT val indexes for tr
    tr_indexes = rand_indexes[int(len(rand_indexes)*validation_prop):]

    trainX = X_tr_total[:, tr_indexes]
    trainY = y_tr_total[tr_indexes]
    trainY = to_one_hot(trainY, 10)

    return trainX, trainY, valX, valY, testX, testY

if __name__ == "__main__":
    nn = MyNeuralNetwork(784, 10, 2, 10, mode="categorical")
    
    trainX, trainY, valX, valY, testX, testY = load_data()
    bestWeights = nn.train(trainX, trainY, valX, valY, testX, testY, epoch_values=[3]) 
    nn.show_W0()