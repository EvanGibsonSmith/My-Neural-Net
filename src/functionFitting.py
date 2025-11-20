import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # needed for projection='3d'
import numpy as np
from math import pi
from scipy.special import erf
from myNeuralNetwork import MyNeuralNetwork

x_num, y_num = 100, 100
x_range = np.linspace(-1, 1, x_num)
y_range = np.linspace(-1, 10, y_num)
save_name = "nn_momentum_fitting.mp4"

def funcDampenedBubbleWrap(x, y):
    denom = np.sin(x) + np.cos(y)
    numer = (x**2) + (y**2) + 1
    return denom / numer

def two_bar_contact(X, T, T_left=1.0, T_right=0.0, alpha=0.1):
    """
    Vectorized: X and T can be any shape, even from meshgrid (https://en.wikipedia.org/wiki/Self-similar_solution).
    Computes two-bar contact solution for heat tranfer over time:
        T(x,t) = (T_left + T_right)/2 + (T_left - T_right)/2 * erf(-X / (2*sqrt(alpha*T)))

    - Avoid divide-by-zero by clamping T to a tiny value.
    """
    T_safe = np.maximum(T, 1e-12)
    return (T_left + T_right)/2 + (T_left - T_right)/2 * erf(-X / (2*np.sqrt(alpha*T_safe)))

    
def dampedBeatingSine(x, y):
    """
    u(x,t) = A1 * exp(-alpha * k1^2 * t) * sin(k1 * x + phi1)
          + A2 * exp(-alpha * k2^2 * t) * sin(k2 * x + phi2)
    Default parameters chosen to produce nice spatial beating and different decay rates.
    """
    alpha = 0.15

    # Mode 1
    A1   = 1.0
    k1   = 2 * np.pi / 5
    phi1 = 0.2

    # Mode 2
    A2   = 0.7
    k2   = 2 * np.pi / 6
    phi2 = 0.5

    term1 = A1 * np.exp(-alpha * (k1**2) * y) * np.sin(k1 * x + phi1)
    term2 = A2 * np.exp(-alpha * (k2**2) * y) * np.sin(k2 * x + phi2)

    return term1 + term2

def sampleFunc(x, y):
    return (np.sin(x*y) - np.cos(y)) / ((x**2 * y**2) + 1)

def xor(x, y):
    x = np.where(x>0, 1, 0)
    y = np.where(y>0, 1, 0)
    xor = np.logical_xor(x, y)
    return np.where(xor==False, 0, 1)

X, Y = np.meshgrid(x_range, y_range)
Z = two_bar_contact(X, Y)

inputs = np.vstack((X.ravel(), Y.ravel()))
outputs = np.atleast_2d(Z.ravel())

nnDeep = MyNeuralNetwork(2, 1, 5, 20, mode="continuous")

learningRate = 0.1
trainX, trainY = nnDeep.setupTraining(inputs, outputs, learningRate) # trainX and trainY are scrambled, inputs/outputs for nn NOT x y on graph

#nn.train(inputs, outputs, learning_rate=[learningRate])


# Collect predictions each epoch
predictions = []

numEpochs = 50
graphEpochRate = 1  # collect every epoch for smooth animation
alpha_momentum = 0.9

# Get predictions
for epoch in range(numEpochs):
    nnDeep.run_epoch(trainX, trainY, batch_size=(trainX.shape[1])//50, alpha_momentum=alpha_momentum)
    Z_deep_nn = nnDeep.get_yhat(inputs)[0].reshape(x_num, y_num)
    predictions.append(Z_deep_nn)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Set to gray back
fig.patch.set_facecolor('#4a4a4a')
ax.set_facecolor('#4a4a4a')
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = True
    axis.pane.set_facecolor((0.29, 0.29, 0.29, 1.0))

ax.grid(False)
ax.set_xlabel("X", color="#f1f1f1")
ax.set_ylabel("Time", color="#f1f1f1")
ax.set_zlabel("Temperature", color="#f1f1f1")
ax.set_zlim(0, 1)

#surf_true = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.5, linewidth=0, antialiased=True)
surf_true = ax.plot_wireframe(X, Y, Z, color='red', linewidth=0.8, rstride=4, cstride=4)
surf_nn = None

def update(frame):
    global surf_nn
    # Remove previous surface if it exists
    if surf_nn is not None:
        surf_nn.remove()  
    # Plot new surface
    surf_nn = ax.plot_surface(
        X, Y, predictions[frame],
        cmap='viridis',
        alpha=0.8,
        linewidth=0,
        antialiased=True
    )
    ax.set_title(f"Momentum {alpha_momentum}, Epoch {frame+1}/{numEpochs}", color="#f1f1f1")

anim = FuncAnimation(fig, update, frames=len(predictions), interval=200)
anim.save(save_name, fps=10, dpi=150)
plt.show()
