from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, dropout=0, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, 
                 use_batch_norm=False, dtype=np.float32):
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
        self.use_dropout = dropout > 0
        self.use_batch_norm = use_batch_norm
        self.params = {}
        self.reg = reg
        self.num_layers = 3
        self.dtype = dtype
        self.pool_height = 2
        self.pool_width = 2
        self.pool_stride = 2

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # NUmber of channels
        C, H, W = input_dim
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        H_pool = (H - self.pool_height) / 2 + 1
        W_pool = (W - self.pool_width) / 2 + 1
        self.params['W2'] = np.random.randn(np.prod((num_filters, H_pool, W_pool)), hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)

        # Initialize the parameters for batch normalization if necessary
        if self.use_batch_norm:
            self.params['gamma1'] = np.ones(num_filters) 
            self.params['beta1'] = np.zeros(num_filters)
            self.params['gamma2'] = np.ones(hidden_dim)
            self.params['beta2'] = np.zeros(hidden_dim)

        # Set dropout parameters if necessary
        self.dropout_param={}
        if self.use_dropout:
            self.dropout_param ={'mode':'train', 'p':dropout}

        self.bn_params = []
        if self.use_batch_norm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        if self.use_batch_norm:
            gamma1 = self.params['gamma1']
            beta1 = self.params['beta1']
            gamma2 = self.params['gamma2']
            beta2 = self.params['beta2']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': self.pool_height, 'pool_width': self.pool_width, 
                      'stride': self.pool_stride}

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batch_norm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        # Convolutional layer going forward
        if self.use_batch_norm:
            first_layer_scores, first_layer_cache = conv_bn_relu_pool_forward(X, W1, b1,
                                                                              gamma1, beta1,
                                                                              conv_param,
                                                                              self.bn_params[0],
                                                                              pool_param)
        else:
            first_layer_scores, first_layer_cache = conv_relu_pool_forward(X, W1, b1, 
                                                                           conv_param,
                                                                           pool_param)

        # Fully connected layers going forward
        if self.use_batch_norm:    
            second_layer_scores, second_layer_cache = affine_bn_relu_forward(first_layer_scores,
                                                                             W2, b2, gamma2, beta2, 
                                                                             self.bn_params[1], 
                                                                             dropout=self.use_dropout, 
                                                                             dropout_param=self.dropout_param)
        else:
            second_layer_scores, second_layer_cache = affine_relu_forward(first_layer_scores, 
                                                                          W2, b2, 
                                                                          dropout=self.use_dropout,
                                                                          dropout_param=self.dropout_param)

        # Output layer going forward
        scores, output_cache = affine_forward(second_layer_scores, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        grads = {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # Compute loss
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        
        # Compute the gradient
        grads['W1'] = self.reg * W1
        grads['W2'] = self.reg * W2
        grads['W3'] = self.reg * W3

        # Output layer going backward
        dx, dw, db = affine_backward(dscores, output_cache)
        grads['W3'] += dw
        grads['b3'] = db

        # Fully connected layers going backward
        if self.use_batch_norm:
            dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, second_layer_cache, dropout=self.use_dropout)
            grads['gamma2'] = dgamma
            grads['beta2'] = dbeta

        else:
            dx, dw, db = affine_relu_backward(dx, second_layer_cache, dropout=self.use_dropout)
        grads['W2'] += dw
        grads['b2'] = db

        # Convolutional layers going backward.
        if self.use_batch_norm:
            _, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(dx, first_layer_cache)
            grads['gamma1'] = dgamma
            grads['beta1'] = dbeta

        else:
            _, dw, db = conv_relu_pool_backward(dx, first_layer_cache)
        grads['W1'] += dw
        grads['b1'] = db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class FullConvNet(object):
    """
    A convolutional neural network of any depth followd by a fully connected netwrok.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3,32,32), num_filters=[32], hidden_layers=[100], 
                 num_classes=10 ,filter_size=7, weight_scale=1e-3, reg=0, dropout=0, 
                 use_batch_norm=False, dtype=np.float32):
        """
        Initialize the networkgit config user.email.
        Inputs:
        - Input_deim: The dimension of the images to be classified.
        - num_filters: An array representing dimension of all of the cnn layers.
        - hidden_layers: An array representing dimension of all hidden layers
          to the fully connected network.
        - num_classes: Integer representing the number of classes.
        - filter_size: The size of every filter applied in the cnn.
        - weight_scale: Scalar giving the standard deviation for the randomly 
          initialized weights.
        - reg: The regularization parameter of the netowrk.
        - dropout: The probability of drping out a node in all the hidden layers
          of the fully connected netwrok.
        - use_batch_norm: A boolean to use batch normalization if necessary.

        """
        self.params={}
        self.use_dropout = dropout > 0
        self.use_batch_norm = use_batch_norm
        self.conv_params = {'stride': 1, 'pad': (filter_size - 1) // 2}
        self.pool_params = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        self.num_conv_layers = len(num_filters)
        self.num_hidden_layers = len(hidden_layers)
        self.bn_params = []
        self.dropout_params = []
        self.reg = reg

        # Initialize batch normalization parameters if necessary.
        num_layers = self.num_conv_layers + self.num_hidden_layers
        if self.use_batch_norm:
            for i in range(num_layers):
                self.bn_params.append({'mode':'train'})
        # Initialize dropout parameters if necessary
        if self.use_dropout:
            self.dropout_params = {'mode':'trian', 'p':dropout}

        C, H, W = input_dim
        channels, HH, WW = C, H, W
        # Initialize the parameters for the Convolutional network.
        for i in range(1, self.num_conv_layers+1):
            self.params['W{}'.format(i)] = np.random.randn(num_filters[i-1], 
                                                           channels, filter_size, 
                                                           filter_size) * weight_scale
            self.params['b{}'.format(i)] = np.zeros(num_filters[i-1])
            # Keeping track of the Height and Width of the image as we convolve
            # it through multiple layers. After pooling make sure the dimensions
            # make sense
            if (HH <=  self.pool_params['pool_height']):
                raise Exception('The pool height and input height are equal'.\
                    format(self.pool_params['pool_height'], HH))
            else:
                HH = (HH - self.pool_params['pool_height']) / self.pool_params['stride'] + 1
            if (WW <= self.pool_params['pool_width']):
                raise Exception('The pool width and input width are equal'.\
                    format(self.params['pool_width'], WW))
            else:
                WW = (WW - self.pool_params['pool_width']) / self.pool_params['stride'] + 1


            # Updating the number of channels for the new input.
            channels = num_filters[i-1]
            # Initialize the parameters for the batch normalization if necessary.
            if self.use_batch_norm:
                self.params['gamma{}'.format(i)] = np.ones(channels)
                self.params['beta{}'.format(i)] = np.zeros(channels)

        # Initialize the parameters for the fully connected network.
        fc_input_dim = np.prod((HH, WW, channels))
        for i in range(1, self.num_hidden_layers+1):
            self.params['W{}'.format(i+self.num_conv_layers)] = np.random.randn(fc_input_dim, 
                                                                                hidden_layers[i-1]) * weight_scale
            self.params['b{}'.format(i+self.num_conv_layers)] = np.zeros(hidden_layers[i-1])
            # Initialize the parameters for batch normalization if necessary.
            if self.use_batch_norm:
                self.params['gamma{}'.format(i+self.num_conv_layers)] = np.ones(hidden_layers[i-1])
                self.params['beta{}'.format(i+self.num_conv_layers)] = np.zeros(hidden_layers[i-1])
            fc_input_dim = hidden_layers[i-1]

        # Initialize the parameters for the last layer of the fully connected network.
        self.params['W{}'.format(i+self.num_conv_layers+1)] = np.random.randn(hidden_layers[i-1],
                                                                              num_classes) * weight_scale
        self.params['b{}'.format(i+self.num_conv_layers+1)] = np.zeros(num_classes)

        # Convert the dtype for the parameters of the model.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluates the loss and gradient for the full cnn.
        Inputs:
        - X: The input images of shape (N, C, H, W)
        - y: The class for each image where each class 0 <= c <= num_classes.
        """

        # Findout if it's trainig or test time
        mode = 'train'
        if y is None:
            mode = 'test'

        # Set the mode for batch normalization and dropout parameters if needed.
        if self.use_batch_norm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        if self.use_dropout:
            self.dropout_params['mode'] = mode

        # Compute the forward pass fo the cnn.
        caches = []
        input_layer = X
        for i in range(1, self.num_conv_layers+1):
            w = self.params['W{}'.format(i)]
            b = self.params['b{}'.format(i)]

            if self.use_batch_norm:
                gamma = self.params['gamma{}'.format(i)]
                beta = self.params['beta{}'.format(i)]
                layer_score, layer_cache = conv_bn_relu_pool_forward(input_layer, w, b, gamma, beta,
                                                                     self.conv_params, self.bn_params[i-1], 
                                                                     self.pool_params)
            else:
                layer_score, layer_cache = conv_relu_pool_forward(input_layer, w, b, self.conv_params, 
                                                                  self.pool_params)
            input_layer = layer_score
            caches.append(layer_cache)

        # Compute the forward pass for the fully connected net.
        num_layers = self.num_conv_layers + self.num_hidden_layers
        for i in range(self.num_conv_layers+1, num_layers+1):
            w = self.params['W{}'.format(i)]
            b = self.params['b{}'.format(i)]
            if self.use_batch_norm:
                gamma = self.params['gamma{}'.format(i)]
                beta = self.params['beta{}'.format(i)]
                layer_score, layer_cache = affine_bn_relu_forward(input_layer, w, b, gamma, beta,
                                                                  self.bn_params[i-1],
                                                                  dropout=self.use_dropout, 
                                                                  dropout_param=self.dropout_params)
            else:
                layer_score, layer_cache = affine_relu_forward(input_layer, w, b, dropout=self.use_dropout, 
                                                               dropout_param=self.dropout_params)
            input_layer = layer_score
            caches.append(layer_cache)

        # Compute the forward pass for the output layer.
        w = self.params['W{}'.format(i+1)]
        b = self.params['b{}'.format(i+1)]
        scores, output_cache = affine_forward(input_layer, w, b)

        # If testing time return the scores
        if mode == 'test':
            return scores

        # Compute the loss
        loss, dscores = softmax_loss(scores, y)

        # Add regularization to the loss and the corresponding gradient.
        grads = {}
        for i in range(1, num_layers+2):
            w = 'W{}'.format(i)
            loss += 0.5 * self.reg * np.sum(self.params[w]**2)
            grads[w] = self.reg * self.params[w]

        # Compute the gradients using backprop on the fully connected net.
        # Start with the output layer
        w = 'W{}'.format(num_layers+1)
        b = 'b{}'.format(num_layers+1)
        dx, dw, db = affine_backward(dscores, output_cache)
        grads[w] += dw
        grads[b] = db
        for i in range(num_layers, self.num_conv_layers, -1):
            cache = caches[i-1]
            w = 'W{}'.format(i)
            b = 'b{}'.format(i)
            if self.use_batch_norm:
                gamma = 'gamma{}'.format(i)
                beta = 'beta{}'.format(i)
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, cache, self.use_dropout)
                grads[gamma] = dgamma
                grads[beta] = dbeta
            else:
                dx, dw, db = affine_relu_backward(dx, cache)
            grads[w] += dw
            grads[b] = db

        # Compute the gradeints using backprop on the convolutional layers.
        for i in range(self.num_conv_layers, 0, -1):
            cache = caches[i-1]
            w = 'W{}'.format(i)
            b = 'b{}'.format(i)
            if self.use_batch_norm:
                gamma = 'gamma{}'.format(i)
                beta = 'beta{}'.format(i)
                dx, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(dx, cache)
                grads[gamma] = dgamma
                grads[beta] = dbeta
            else:
                dx, dw, db = conv_relu_pool_backward(dx, cache)
            grads[w] += dw
            grads[b] = db

        return loss, grads


















































