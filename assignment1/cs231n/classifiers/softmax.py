import numpy as np
from random import shuffle
from past.builtins import xrange
import math


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    for i in xrange(num_train):
        scores = X[i].dot(W)
        logc = -np.max(scores)
        scores = scores + logc
        correct_score = scores[y[i]]
        scores[y[i]] = 0
        loss -= correct_score - math.log(np.sum(np.exp(scores)) - 1)

    loss /= num_train
    loss += reg * np.sum(W * W)

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    A1 = X.dot(W)
    A2 = np.max(A1, axis=1)
    A3 = A1 - A2[:, np.newaxis]
    A4 = A3[np.arange(num_train), y]
    A5 = A3
    A5[np.arange(num_train), y] = 0

    A6 = np.exp(A5)
    A7 = np.sum(A6, axis=1)
    A8 = np.log(A7) - 1
    loss_sum = np.sum(A4 - A8)

    loss_num_train = -loss_sum / num_train
    loss = loss_num_train + reg * np.sum(W * W)

    # d_W = 2 * reg * W  # d_loss/d_W
    # d_l_n_t = 1
    # d_loss_sum = d_l_n_t * -1 / num_train  # d_loss/d_loss_sum
    # d_A4 = (1 + np.zeros(A4.shape)) * d_loss_sum
    # d_A8 = (np.zeros(A4.shape) - 1) * d_loss_sum
    # d_A7 = np.divide(1, A7) * d_A8
    # d_A6 = (1 + np.zeros(A6.shape)) * d_A7
    # d_A3 = A3 * A6 #elementwise czy do?
    # help = (np.zeros_like(A3)+1)
    # help[np.arange(num_train), y] = 0
    # d_A3 = d_A3 + (np.zeros_like(A3)+1)

    return loss, dW
