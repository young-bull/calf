import numpy as np
from collections import Counter
import torch
from torchvision.transforms import transforms

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        dists[i,j] = np.sum((X[i]-self.X_train[j])**2)
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      dists[i,:] = np.sum((self.X_train-X[i,:])**2, axis=1)
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    return np.sum((self.X_train * self.X_train), axis=1) + np.sum(X * X, axis=1)[:, np.newaxis] - 2 * np.dot(X, self.X_train.T)

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      nearest_x = np.argsort(dists[i,:])[:k]
      # print("最近的样本：", nearest_x)
      nearest_y = self.y_train[nearest_x]
      # print("最近的样本的类别：", nearest_y)
      winner_class = Counter(nearest_y).most_common(1)[0][0]
      # print("预测：", winner_class)
      y_pred[i] = winner_class

    return y_pred


transform = transforms.Compose([
                transforms.ToTensor(),
            ])

class KNN:
    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test, k=1):
        dists = self.compute_distance(X_test)
        return self.predict_labels(dists, k=k)
        
    def compute_distance(self, X_test):
      a2 = torch.sum(X_test.pow(2), 1, keepdim=True)
      _2ab = X_test.mm(self.X_train.t())
      b2 = torch.sum(self.X_train.t().pow(2), 0, keepdim=True)
      dist = a2 - 2 * _2ab + b2
      return dist
    
    def predict_labels(self, dists, k=1):
        """
        找到最近的K个样本，让它们投票：
        dist的第i行是test的第[i]个样本，列是test[i]样本与train[j]样本的距离
            1. 找到第i行中，最小的k个值
            2. 找到这k个值对应的类别
            3. 取出现次数最多的类别
        """
        # 1. 找到第i行，与X_train距离最小的k个X_test
        _, indices = torch.topk(dists, k, dim=1, largest=False)
        # print("[torch]最近的样本：", indices)
        
        # 2. 找到这k个X_train对应的类别
        kinds = self.y_train[indices]
        #print("[torch]最近样本的类别：", kinds)
        # print("[torch] 类别：", kinds)
        
        # 3. 统计出现次数最多的类别
        predict, __ = torch.mode(kinds)
        # print("[torch] 预测：", predict)
        # print("")
        
        return predict