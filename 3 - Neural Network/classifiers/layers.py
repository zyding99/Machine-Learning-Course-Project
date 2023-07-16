from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    # Reshape x into rows
    N = x.shape[0]
    x_into_row = x.reshape(N, -1)         # (N,D)
    out = np.dot(x_into_row, w) + b       # (N,M)
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)                       # (N,D)    
    dx = np.reshape(dx, x.shape)                 # (N,d1,...,d_k)   
    x_into_row = x.reshape(x.shape[0], -1)            # (N,D)    
    dw = np.dot(x_into_row.T, dout)                   # (D,M)    
    db = np.sum(dout, axis = 0, keepdims = True)     # (1,M)    

    return dx, dw, db


def ReLU(x):
    return np.where(x < 0, 0, x)


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = ReLU(x)    
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache  
    dx = dout    
    dx[x <= 0] = 0

    return dx

### Convenience layer ###
def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    inter_out, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(inter_out)
    cache = (fc_cache, relu_cache)
    return out, cache
 
 
def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    loss, dx = 0.0, None
    probs = np.exp(x - np.max(x, axis = 1, keepdims = True))    
    probs /= np.sum(probs, axis = 1, keepdims = True)    
    N = x.shape[0]   
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N    
    dx = probs.copy()    
    dx[np.arange(N), y] -= 1    
    dx /= N    

    return loss, dx
