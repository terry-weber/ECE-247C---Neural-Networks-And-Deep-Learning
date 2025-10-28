import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False, bn_params = {'mode': 'train'}):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
    W2_row_size = num_filters * input_dim[1]//2 * input_dim[2]//2
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(W2_row_size, hidden_dim)) 
    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))

    self.params['b1'] = np.zeros(num_filters)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['b3'] = np.zeros(num_classes)

    self.params['y1'] = np.random.normal( 1 , 1e-3, num_filters)
    self.params['beta1'] = np.zeros( num_filters )
    self.params['y2'] = np.random.normal( 1 , 1e-3, hidden_dim)
    self.params['beta2'] = np.zeros( hidden_dim )

    self.bn_params = bn_params
    self.bn_params2 = dict(bn_params)
  
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1, y1, beta1 = self.params['W1'], self.params['b1'], self.params['y1'], self.params['beta1']
    W2, b2, y2, beta2 = self.params['W2'], self.params['b2'], self.params['y2'], self.params['beta2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]

    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    
    
    out_1, cache_1 = conv_batchnorm_relu_forward(X, W1, b1, conv_param, y1, beta1, self.bn_params)
    out_2, cache_2 = max_pool_forward_fast(out_1, pool_param)
    out_3, cache_3 = affine_batchnorm_relu_forward(out_2, W2, b2, y2, beta2, self.bn_params2)
    out_4, cache_4 = affine_forward(out_3, W3, b3)
    scores = out_4

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    loss, dscores = softmax_loss(scores, y)
    loss += sum(0.5*self.reg*np.sum(W_tmp**2) for W_tmp in [W1, W2, W3])

    dx_3, grads['W3'], grads['b3'] = affine_backward(dscores, cache_4)
    dx_2, grads['W2'], grads['b2'], grads['y2'], grads['beta2'] = affine_batchnorm_relu_backward( dx_3, cache_3 )
    dx_2_prime = max_pool_backward_fast( dx_2, cache_2 )
    dx_1, grads['W1'], grads['b1'], grads['y1'], grads['beta1'] = conv_batchnorm_relu_backward( dx_2_prime, cache_1 )

    grads['W3'] += self.reg*self.params['W3']
    grads['W2'] += self.reg*self.params['W2']
    grads['W1'] += self.reg*self.params['W1']
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
