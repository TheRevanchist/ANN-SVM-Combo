import numpy as np
import math


class ThreeLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/(input_size * hidden_size))
    self.params['b1'] = np.zeros(hidden_size)
    #self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/(hidden_size * output_size))
    #self.params['b2'] = np.zeros(output_size)    
    self.params['W2'] = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0/(hidden_size * hidden_size))
    self.params['b2'] = np.zeros(hidden_size)
    self.params['W3'] = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/(hidden_size * output_size))
    self.params['b3'] = np.zeros(output_size) 


  def ReLU(self, x):
     """ A helper function which performs ReLU activations given a numpy vector """
     return np.maximum(x, 0.0)

  def loss(self, X, y=None, reg=0.0, p=0.5):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    N, D = X.shape
         

    # Compute the forward pass - first layer
    z1 = X.dot(W1) + b1
    a1 = self.ReLU(z1)
    
    U1 = (np.random.rand(*a1.shape) < p) / p
    a1 *= U1
    
    # Compute the forward pass - second layer
    z2 = a1.dot(W2) + b2
    a2 = self.ReLU(z2)
    
    U2 = (np.random.rand(*a2.shape) < p) / p
    a2 *= U2
    
    # Compute the forward pass - output layer
    scores = a2.dot(W3) + b3
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores
    
    loss = 0.0

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    corect_logprobs = -np.log(probs[range(N), y])
    data_loss = np.sum(corect_logprobs) / N
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2) + 0.5 * reg * np.sum(W3 * W3)
    loss = data_loss + reg_loss

    # Backward pass: compute gradients
    grads = {}
    
    # gradients for the weights from the hidden layer to the softmax layer
    dscores = probs
    dscores[range(N), y] -= 1
    dscores /= N
    
    # backprop into W3 and b3
    grads['W3'] = np.dot(a2.T, dscores)  
    grads['b3'] = np.sum(dscores, axis=0)
    
    # backprop into the second hidden layer
    dhidden_2 = np.dot(dscores, W3.T)
    dhidden_2 *= U2
    
    # backprop the ReLU non-linearity
    dhidden_2[a2 <= 0] = 0
    
    # backprop into W2 and b2
    grads['W2'] = np.dot(a1.T, dhidden_2)
    grads['b2'] = np.sum(dhidden_2, axis = 0)
    
    # backprop into the first hidden layer
    dhidden = np.dot(dhidden_2, W2.T)
    dhidden *= U1
    
    # backprop the ReLU non-linearity
    dhidden[a1 <= 0] = 0
    
    # backprop into W,b
    grads['W1'] = np.dot(X.T, dhidden)
    grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    grads['W3'] += reg * W3
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1

    return loss, grads
    
    
  def loss2(self, X, y=None, reg=0.0, p=0.5):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    N, D = X.shape
    margin_size = 1

    # Compute the forward pass - first layer
    z1 = X.dot(W1) + b1
    a1 = self.ReLU(z1)
    
    U1 = (np.random.rand(*a1.shape) < p) / p
    a1 *= U1
    
    # Compute the forward pass - second layer
    z2 = a1.dot(W2) + b2
    a2 = self.ReLU(z2)
    
    U2 = (np.random.rand(*a2.shape) < p) / p
    a2 *= U2    
    
    # Compute the forward pass - output layer
    scores = a2.dot(W3) + b3
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores
    
    loss = 0.0

    correct_class_score = scores[np.arange(N), y]  
    margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + margin_size)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins)
    loss /= N 
    loss += 0.5 * reg * np.sum(W2 * W2) + 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W3 * W3) # regularize 

    # Backward pass: compute gradients
    grads = {}
    X_mask = np.zeros(margins.shape)
    X_mask[margins > 0] = 1
  
    incorrect_counts = np.sum(X_mask, axis=1)
    X_mask[np.arange(N), y] = -incorrect_counts
    dscores = X_mask
    dscores /= N # average out weights
    
    # backprop into W3 and b3
    grads['W3'] = np.dot(a2.T, dscores)
    grads['b3'] = np.sum(dscores, axis = 0)
    
    # backprop into the second hidden layer
    dhidden_2 = np.dot(dscores, W3.T)  
    dhidden_2 *= U2
    
    # backprop the ReLU non-linearity
    dhidden_2[a1 <= 0] = 0
    
    # backprop into W2 and b2
    grads['W2'] = np.dot(a1.T, dhidden_2)
    grads['b2'] = np.sum(dhidden_2, axis = 0)
    
    # backprop into the first hidden layer
    dhidden = np.dot(dhidden_2, W2.T)
    dhidden *= U1
    
    # backprop the ReLU non-linearity
    dhidden[a1 <= 0] = 0
    
    # backprop into W,b
    grads['W1'] = np.dot(X.T, dhidden)
    grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1
    grads['W3'] += reg * W3

    return loss, grads     

  def stochastic_gradient_descent(self, learning_rate, grads):
    """ Does the optimizations using the simple mini-batch gradient descent """
    
    self.params['W1'] += -learning_rate * grads['W1']
    
    self.params['b1'] += -learning_rate * grads['b1']
    
    self.params['W2'] += -learning_rate * grads['W2'] 
    
    self.params['b2'] += -learning_rate * grads['b2']
    
    self.params['W3'] += -learning_rate * grads['W3']
     
    self.params['b3'] += -learning_rate * grads['b3']
    
  
  def momentum(self, learning_rate, grads) :
    """ Does the optimizations using mini-batch gradient descent in addition to 
        momentum """
    
    mu = 0.9
    v1, v2, v3, b1, b2, b3 = 0, 0, 0, 0, 0, 0
    
    v1 = mu * v1 - learning_rate * grads['W1']
    self.params['W1'] += v1
       
    v2 = mu * v2 - learning_rate * grads['W2']
    self.params['W2'] += v2
    
    v3 = mu * v3 - learning_rate * grads['W3']
    self.params['W3'] += v3    
     
    b1 = mu * b1 - learning_rate * grads['b1']
    self.params['b1'] += b1
       
    b2 = mu * b2 - learning_rate * grads['b2']
    self.params['b2'] += b2
    
    b3 = mu * b3 - learning_rate * grads['b3']
    self.params['b3'] += b3


  def nesterov_momentum(self, learning_rate, grads) :
    """ Does the optimizations using mini-batch gradient descent in addition to 
        nesterov's momentum """
    
    mu = 0.9
    v1, v2, v3, b1, b2, b3 = 0, 0, 0, 0, 0, 0 
    v_prev1, v_prev2, v_prev3, b_prev1, b_prev2, b_prev3 = v1, v2, v3, b1, b2, b3
      
    v1 = mu * v1 - learning_rate * grads['W1']
    self.params['W1'] += -mu * v_prev1 + (1 + mu) * v1
    
    v2 = mu * v2 - learning_rate * grads['W2']
    self.params['W2'] += -mu * v_prev2 + (1 + mu) * v2  
    
    v3 = mu * v3 - learning_rate * grads['W3']
    self.params['W3'] += -mu * v_prev3 + (1 + mu) * v3     
      
    b1 = mu * b1 - learning_rate * grads['b1']
    self.params['b1'] += -mu * b_prev1 + (1 + mu) * b1
    
    b2 = mu * b2 - learning_rate * grads['b2']
    self.params['b2'] += -mu * b_prev2 + (1 + mu) * b2
    
    b3 = mu * b3 - learning_rate * grads['b3']
    self.params['b3'] += -mu * b_prev3 + (1 + mu) * b3
    
    
  def adagrad(self, learning_rate, grads):
    """ Does the optimizations using the Adagrad optimizer """
    
    cache1, cache2, cache3, cacheb1, cacheb2, cacheb3 = 0, 0, 0, 0, 0, 0
    eps = 1e-6
    
    cache1 += grads['W1']**2
    self.params['W1'] += - learning_rate * grads['W1'] / (np.sqrt(cache1) + eps)

    cache2 += grads['W2']**2
    self.params['W2'] += - learning_rate * grads['W2'] / (np.sqrt(cache2) + eps)
    
    cache3 += grads['W3']**2
    self.params['W3'] += - learning_rate * grads['W3'] / (np.sqrt(cache3) + eps)    
    
    cacheb1 += grads['b1']**2
    self.params['b1'] += - learning_rate * grads['b1'] / (np.sqrt(cacheb1) + eps)

    cacheb1 += grads['b2']**2
    self.params['b2'] += - learning_rate * grads['b2'] / (np.sqrt(cacheb2) + eps)        

    cacheb3 += grads['b3']**2
    self.params['b3'] += - learning_rate * grads['b3'] / (np.sqrt(cacheb3) + eps) 
    

  def RMSProp(self, learning_rate, learning_rate_decay, grads):
    """ Does the optimizations using RMSProp optimizer """
        
    cache1, cache2, cache3, cacheb1, cacheb2, cacheb3 = 0, 0, 0, 0, 0, 0
    eps = 1e-8   
    
    cache1 = learning_rate_decay * cache1 + (1 - learning_rate_decay) * grads['W1']**2
    self.params['W1'] += - learning_rate * grads['W1'] / (np.sqrt(cache1) + eps)

    cache2 = learning_rate_decay * cache2 + (1 - learning_rate_decay) * grads['W2']**2
    self.params['W2'] += - learning_rate * grads['W2'] / (np.sqrt(cache2) + eps)          

    cache3 = learning_rate_decay * cache3 + (1 - learning_rate_decay) * grads['W3']**2
    self.params['W3'] += - learning_rate * grads['W3'] / (np.sqrt(cache3) + eps) 

    cacheb1 = learning_rate_decay * cacheb1 + (1 - learning_rate_decay) * grads['b1']**2
    self.params['b1'] += - learning_rate * grads['b1'] / (np.sqrt(cacheb1) + eps)

    cacheb2 = learning_rate_decay * cacheb2 + (1 - learning_rate_decay) * grads['b2']**2
    self.params['b2'] += - learning_rate * grads['b2'] / (np.sqrt(cacheb2) + eps)

    cacheb3 = learning_rate_decay * cacheb3 + (1 - learning_rate_decay) * grads['b3']**2
    self.params['b3'] += - learning_rate * grads['b3'] / (np.sqrt(cacheb3) + eps)  

  def adam(self, learning_rate, grads):
    """ Does the optimizations using the adam optimizer """  
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8 
    
    v1, v2, v3, b1, b2, b3 = 0, 0, 0, 0, 0, 0
    m1, m2, m3, mb1, mb2, mb3 = 0, 0, 0, 0, 0, 0
    
    m1 = beta1*m1 + (1-beta1)*grads['W1']
    v1 = beta2*v1 + (1-beta2)*(grads['W1']**2)
    self.params['W1'] += - learning_rate * m1 / (np.sqrt(v1) + eps)
      
    m2 = beta1*m2 + (1-beta1)*grads['W2']
    v2 = beta2*v2 +(1-beta2)*(grads['W2']**2)
    self.params['W2'] += - learning_rate * m2 / (np.sqrt(v2) + eps)
    
    m3 = beta1*m3 + (1-beta1)*grads['W3']
    v3 = beta2*v3 + (1-beta2)*(grads['W3']**2)
    self.params['W3'] += - learning_rate * m3 / (np.sqrt(v3) + eps)    

    mb1 = beta1*mb1 + (1-beta1)*grads['b1']
    b1 = beta2*b1 + (1-beta2)*(grads['b1']**2)
    self.params['b1'] += - learning_rate * mb1 / (np.sqrt(b1) + eps)   
      
    mb2 = beta1*mb2 + (1-beta1)*grads['b2']
    b2 = beta2*b2 + (1-beta2)*(grads['b2']**2)
    self.params['b2'] += - learning_rate * mb2 / (np.sqrt(b2) + eps) 
    
    mb3 = beta1*mb3 + (1-beta1)*grads['b3']
    b3 = beta2*b3 + (1-beta2)*(grads['b3']**2)
    self.params['b3'] += - learning_rate * mb3 / (np.sqrt(b3) + eps)
    

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100, optimizer='Adam', p = 0.5,
            batch_size=200, verbose=False, svm=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
        
      sample_indices = np.random.choice(np.arange(num_train), batch_size)
      X_batch = X[sample_indices]
      y_batch = y[sample_indices]
      
      if svm:
          loss, grads = self.loss2(X_batch, y_batch, reg, p)
      else:
          loss, grads = self.loss(X_batch, y_batch, reg, p)
      loss_history.append(loss) 
      
      # check what optimizer to use 
      if optimizer == 'Adam':
        self.adam(learning_rate, grads)  
      elif optimizer == 'RMSProp':
        self.RMSProp(learning_rate, learning_rate_decay, grads)
      elif optimizer == 'Adagrad':
        self.adagrad(learning_rate, grads)
      elif optimizer == 'Nesterov':
        self.nesterov_momentum(learning_rate, grads)
      elif optimizer == 'Momentum':
        self.momentum(learning_rate, grads)
      else:
        self.stochastic_gradient_descent(learning_rate, grads)        

      if math.isnan(loss):
          break
      
      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    z1 = X.dot(self.params['W1']) + self.params['b1']
    a1 = np.maximum(0, z1) # pass through ReLU activation function
    z2 = a1.dot(self.params['W2']) + self.params['b2']
    a2 = np.maximum(0, z2)
    scores = a2.dot(self.params['W3']) + self.params['b3']
    y_pred = np.argmax(scores, axis=1)

    return y_pred


