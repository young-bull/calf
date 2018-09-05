from __future__ import print_function

import torch
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
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
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    # print("W1:", self.params['W1'])
    self.params['b1'] = np.zeros((1,hidden_size))
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros((1, output_size))

  def loss(self, X, y=None, reg=0.0):
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
    N, D = X.shape
    _ , num_classes = W2.shape 

    # Compute the forward pass
    hidden = X.dot(W1) + b1
    hidden = np.maximum(0, hidden)
    scores = hidden.dot(W2) + b2
 
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    scores -= np.max(scores, axis=1)[ :, np.newaxis]
    scores_exp = np.exp(scores)
    loss = -scores[np.arange(len(scores)), y] + np.log(np.sum(scores_exp, axis=1))
    loss = np.sum(loss) / N

    # Comute the gradients
    grads = {}
    scores_exp /= np.sum(scores_exp, axis=1)[ :, np.newaxis]
    scores_exp[np.arange(len(scores_exp)), y] -= 1
    dScores = scores_exp / N
    grads['W2'] = np.dot(hidden.T, dScores)
    grads['b2'] = np.dot(np.ones((1, N)), dScores)
    dHidden = np.dot(dScores, W2.T)
    dHidden = dHidden * (hidden > 0)
    grads['W1'] = np.dot(X.T, dHidden)
    grads['b1'] = np.dot(np.ones((1, N)), dHidden)

    # Add regularization
    loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    grads['W1'] += 2 * reg * W1
    grads['W2'] += 2 * reg * W2

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
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

    for it in range(num_iters):
      random_idx = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[random_idx]
      y_batch = y[random_idx]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      self.params['W1'] -= learning_rate * grads['W1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['b2'] -= learning_rate * grads['b2']

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

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
    return np.argmax(self.loss(X), axis=1)




class TwoLayerNet_Pytorch(object):
  """
  一个两层的全连接神经网络。网络有一个输入层维度为N， 一个隐藏层维度为H，实现C分类。
  我们用softmax损失函数和L2正则式训练网络。
  网络在第一个全连接层后使用一个ReLU非线性变换

  网络的结构：
  输入 - 全连接层 - ReLU - 全连接层 - softmax

  第二个全连接层的输出是每个类的得分
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    初始化模型。权重被初始化为小的随机值，偏移bias初始化为0
    权重和偏移放置在变量self.params中，它是一个字典，其键名如下：

    W1: 第一层的权重值，形状(D, H)
    b1: 第一层偏移，形状(H, )
    W2: 第二层权重，形状(H, C)
    b2: 第二层偏移，形状(C, )

    输入：
    - input_size: 维度为D的输入数据
    - hidden_size: 隐藏层中的神经元数量 H 
    - output_size: 种类数量 C
    """
    self.params = {}
    self.params['W1'] = torch.from_numpy(std * np.random.randn(input_size, hidden_size))
    self.params['b1'] = torch.from_numpy(np.zeros((1,hidden_size)))
    self.params['W2'] = torch.from_numpy(std * np.random.randn(hidden_size, output_size))
    self.params['b2'] = torch.from_numpy(np.zeros((1, output_size)))


    # self.params['W1'] = std * torch.randn(input_size, hidden_size)
    # print("[torch] W1:", self.params['W1'])
    # self.params['b1'] = torch.zeros(1,hidden_size)
    # self.params['W2'] = std * torch.randn(hidden_size, output_size)
    # self.params['b2'] = torch.zeros(1, output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    为两层的全连接神经网络计算损失和梯度

    输入：
    - X: 输入数据(N, D), 每一个X[i]是一个训练样本
    - y: 训练标签向量。这个参数可选。如果没有，则返回scores分类；如果有，返回
          损失和梯度
    - reg: 正则化强度

    返回：
    如果y是None，返回形状为(N, C)的矩阵scores分类，scores[i, c]是输入样本X[i]在c类上的得分
    如果y不是None，返回一个元组：
    - loss: 本批训练样本的损失值(数据损失和正则损失)
    - grads: 一个字典，与self.params有相同的键名，键值为损失函数对应的参数梯度
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.size()
    _ , num_classes = W2.size() 

    # 前向传播
    hidden = X.mm(W1) + b1
    hidden = torch.max(hidden, torch.tensor([0.0], dtype=torch.float64))
    scores = hidden.mm(W2) + b2
 
    # 如果target目标没有给定，跳过
    if y is None:
      return scores

    # 计算softmax损失
    loss_fn_Li = torch.nn.LogSoftmax(dim=1)
    row = torch.arange(X.size(0)).long()
    col = y.long()

    scores -= torch.max(scores, dim=1, keepdim=True)[0]
    loss_i = -loss_fn_Li(scores)[row, col]
    loss = torch.sum(loss_i) / N

    # 计算梯度
    grads = {}
    exp_scores = torch.exp(scores)

    exp_scores /= torch.sum(exp_scores, dim=1, keepdim=True)
    exp_scores[row, col] -= 1
    dScores = exp_scores / N
    grads['W2'] = torch.mm(hidden.t(), dScores)
    grads['b2'] = torch.mm(torch.ones(1, N).double(), dScores)
    dHidden = torch.mm(dScores, W2.t())
    dHidden = dHidden * (hidden > 0).double()
    grads['W1'] = torch.mm(X.t(), dHidden)
    grads['b1'] = torch.mm(torch.ones(1, N).double(), dHidden)

    # 加入正则化
    loss += reg * (torch.sum(W1.pow(2)) + torch.sum(W2.pow(2)))
    grads['W1'] += 2 * reg * W1
    grads['W2'] += 2 * reg * W2

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    使用随机梯度下降来训练神经网络

    输入：
    - X: 形状为(N, D)的训练数据
    - y: 形状为(N, )的训练标签, y[i] = c意味着X[i]有标签c
    - X_val：形状为(N_val, D)的验证数据
    - y_val：形状为(N_val, )的验证标签
    - learning_rate：优化用的学习率
    - learning_rate_decay: 标量，用于每代缩小学习率的因子
    - reg：正则强度
    - num_iters：当优化时，采用的迭代次数
    - batch_size：每次迭代使用的训练样本数
    - verbose: 如果为 true，打印优化过程
    """
    num_train = X.size(0)
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      random_idx = torch.randint(num_train, (batch_size, )).long()
      X_batch = X[random_idx]
      y_batch = y[random_idx]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      self.params['W1'] -= learning_rate * grads['W1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['b2'] -= learning_rate * grads['b2']

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).double().mean()
        val_acc = (self.predict(X_val) == y_val).double().mean()
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
    使用训练权重来预测标签。对于每个数据样本，预测C个类别的得分
    并且根据最高分给每个样本分配所属类别

    输入：
    - X：形状为(N, D)，用于分类的N个D维的数据点 

    返回：
    - y_pred：形状(N, ) 为每个X的元素预测其所属类别，
              对于所有的i，y_pred[i] = c意味着预测X[i]属于c类
    """
    return torch.argmax(self.loss(X), dim=1)

