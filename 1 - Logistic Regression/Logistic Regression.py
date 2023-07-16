### Wine Classification with Logistic Regression ###

from math import log, e
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import save_csv

def sigmoid(x):
    return 1 / (1 + e**(-x))

data = io.loadmat("data.mat")

def cost(w, x, y, l):
    cost = 0
    log_loss = 0
    for i in range(6000):

        yi = y[i][0]
        si = sigmoid(np.dot(x[i], w))
        if 0 < si < 1:
            cost_term = yi * log(si) + (1 - yi) * log(1 - si)
            log_loss += cost_term

    cost = l * np.linalg.norm(w, ord = 2)**2 - log_loss     
    return cost

def stochastic_gradient_descent(x, y, l):
    w = np.array([0.003] * 12)
    iterations = []
    cost_fn = []
    for i in range(6000):
        iterations.append(i)
        print(i)
        c = cost(w, x, y, l)
        cost_fn.append(c)

        ep = 0.001 / (i + 1)
        w = w - ep * (2 * l * w - (y[i][0] - sigmoid(np.dot(x[i], w))) * x[i])

    return iterations, cost_fn ,w

n_samples = len(data["X"])
idx = np.random.permutation(n_samples)
x = data["X"][idx]
y = data["y"][idx]
l = 0.2

iterations, cost_fn, w = stochastic_gradient_descent(x, y, l)
plt.plot(iterations, cost_fn)
plt.xlabel('# iterations')
plt.ylabel('cost function')
plt.show()

