import numpy as np
from random import shuffle
import scipy as sc

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
  num_classes = dW.shape[1]

  for i in range(num_train):
    scores_i = W.T.dot(X[i, :])
    
    maximum = np.max(scores_i) # trick to deal with numerical instability
    scores_i -= maximum
    
    # compute the denominator in the log part
    suma = 0
    for el in scores_i:
      suma += np.exp(el)
      
    nominator = np.exp(scores_i[y[i]]) # compute the nominator in the log part
    loss -= np.log(nominator/suma)
    
    # compute the graient
    for j in range(num_classes):
      p = np.exp(scores_i[j])/suma
      dW[:, j] += (p-(j == y[i])) * X[i, :]
  
  # divide the loss/gradient with the number of examples and add the regularization term  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W) 
  dW /= num_train
  dW += reg * W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  
  scores = W.T.dot(X.T)
  scores -= np.max(scores)
  correct_scores = scores[y, range(num_train)]

  nominator = np.exp(correct_scores)
  denominator = np.exp(scores)
  denominator = np.sum(denominator, axis = 0)
  
  loss -= np.log(nominator/denominator)
  loss = np.mean(loss)
  
  p = np.exp(scores)/np.sum(np.exp(scores), axis=0)
  ind = np.zeros(p.shape)
  ind[y, range(num_train)] = 1
  dW = np.dot((p-ind), X)
  dW /= num_train
  dW = dW.T

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  return loss, dW

