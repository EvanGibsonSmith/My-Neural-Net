import numpy as np

arrIn = np.array([1, 2, 3]).reshape(1, -1)
print(arrIn)
iter = 4
out = np.repeat(arrIn, iter, axis=0).reshape(iter, arrIn.shape[1])
print(out)
print(out[:, 1])

a = np.ones((2, 2, 2))
print(a)
b = np.arange(2 * 2 * 2).reshape((2, 2, 2))
print(b)
print(np.matmul(a,b).shape)
print(np.matmul(a,b))

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.stack((a, b)).T)

def softmax(z):
    softmax = np.exp(z)
    softMaxDivisionFactors = (np.reciprocal(np.sum(softmax, axis=0))).reshape(-1, z.shape[1])
    softMaxDivisionFactors =  np.repeat(softMaxDivisionFactors, z.shape[0], axis=0)
    softmax = softmax * softMaxDivisionFactors
    return softmax

z = np.array([0, 2, 1, 1, 2, 1]).reshape(3, 2)
print(softmax(z))