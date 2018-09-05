import torch
import numpy as np
from random import shuffle

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
  num_train = len(X)
  _ , num_classes = W.shape

  for i in range(num_train):
    scores = X[i,: ].dot(W)
    scores -= np.max(scores)
    sum_exp = np.sum(np.exp(scores))
    loss += (np.log(sum_exp) - scores[y[i]])
    for j in range(num_classes):
      dW[ :,j] += (np.exp(scores[j]) / sum_exp ) * X[i,: ]
    dW[ :, y[i]] -= X[i,: ]

  loss /= num_train
  dW /= num_train

  # add regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  dW = np.zeros_like(W)
  num_train = len(X)

  scores = X.dot(W)
  scores -= np.max(scores, axis=1)[ :, np.newaxis]
  exp_scores = np.exp(scores)
  loss = -scores[np.arange(len(scores)), y] + np.log(np.sum(exp_scores, axis=1))
  loss = np.sum(loss) / num_train
  # compute gradient
  exp_scores /= np.sum(exp_scores, axis=1)[ :, np.newaxis]
  exp_scores[np.arange(len(exp_scores)), y] -= 1
  dW = X.T.dot(exp_scores) / num_train

  # Add regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  return loss, dW

def softmax_loss_vectorized_pytorch(W, X, y, reg):
  num_train = X.size(0)

  scores = X.mm(W)
  scores -= torch.max(scores, dim=1, keepdim=True)[0]
  row = torch.arange(X.size(0)).long()
  col = y.long()

  loss_fn_Li = torch.nn.LogSoftmax(dim=1)
  loss_i = -loss_fn_Li(scores)[row, col]
  loss = torch.sum(loss_i) / num_train
  # 计算梯度
  exp_scores = torch.exp(scores)
  exp_scores /= torch.sum(exp_scores, dim=1, keepdim=True)
  exp_scores[row, col] -= 1
  dW = X.t().mm(exp_scores) / num_train

  # 加入正则式
  loss += reg * torch.sum(W.pow(2))
  dW += 2 * reg * W

  return loss, dW