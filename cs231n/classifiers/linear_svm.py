import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        count += 1
        dW[:, j] += X[i]
        
    dW[:, y[i]] += -count * X[i]     

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  

  # Add regularization to the loss and to the weights
  dW += reg * W
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  
  # vectorized version of the loss
  loss = 0.0
  margin_size = 1
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_score = scores[np.arange(num_train), y]

  margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + margin_size)
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins)  
  
  loss /= num_train 
  loss += 0.5 * reg * np.sum(W * W) # regularize 

  # vectorized version of the gradient

  X_mask = np.zeros(margins.shape)
  X_mask[margins > 0] = 1
  
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts
  dW = X.T.dot(X_mask)
  dW /= num_train # average out weights
  dW += reg*W # regularize the weights  

  return loss, dW
