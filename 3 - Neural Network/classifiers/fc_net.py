import numpy as np
from layers import *

class FullyConnectedNet(object):    
    """
    A fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of [H, ...], and perform classification over C classes.

    The architecure should be like affine - relu - affine - softmax for a one
    hidden layer network, and affine - relu - affine - relu- affine - softmax for
    a two hidden layer network, etc.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim, hidden_dim,
                 num_classes = 10, weight_scale = 1e-2):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A list of integer giving the sizes of the hidden layers
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """

        self.num_layers = 1 + len(hidden_dim)
        self.params = {}

        layers_dims = [input_dim] + hidden_dim + [num_classes]
        for i in range(self.num_layers):    
            self.params['W' + str(i+1)] = weight_scale * np.random.randn(layers_dims[i], layers_dims[i+1])    
            self.params['b' + str(i+1)] = np.zeros((1, layers_dims[i+1]))    

    def loss(self, X, y = None):    
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        
        scores = None    
        out, cache = {}, {}
        out[0] = X

        # Forward pass: compute loss
        for i in range(self.num_layers - 1):    
            # Unpack variables from the params dictionary    
            W, b = self.params['W' + str(i+1)], self.params['b' + str(i+1)]
            out[i+1], cache[i] = affine_relu_forward(out[i], W, b)

        W, b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
        scores, _cache = affine_forward(out[self.num_layers - 1], W, b)

        # If test, return early
        if y is None:   
            return scores

        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)

        # Backward pass: compute gradients
        dout = {}
        t = self.num_layers - 1
        dout[t], grads['W'+str(t+1)], grads['b'+str(t+1)] = affine_backward(dscores, _cache)

        for i in range(t):    
            dout[t-1-i], grads['W'+str(t-i)], grads['b'+str(t-i)] = affine_relu_backward(dout[t-i], cache[t-1-i])

        return loss, grads
