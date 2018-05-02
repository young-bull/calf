import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

class DataLoader(object):
    
    def __init__(self, dataset_path='~/.datasets/'):
        self.path = os.path.expanduser(dataset_path)
        if (False == os.path.exists(self.path)):
            print("Can not find dataset folder, please reset it")
    
    def get_data(self, kind='train'):
        
        if ((kind != 'train') & (kind != 't10k')):
            print("kind is train or t10k")
            return None, None
        
        labels_path = os.path.join(self.path,'%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(self.path,'%s-images-idx3-ubyte.gz' % kind)
        
        if (False == os.path.exists(labels_path)):
            print("Can not find mnist label file: " , labels_path)
            print("please go to `http://yann.lecun.com/exdb/mnist/` download dataset")
            return None, None
            
        with gzip.open(labels_path,'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(),dtype=np.uint8,offset=8)
        with gzip.open(images_path,'rb') as imgpath:
            images = np.frombuffer(imgpath.read(),dtype=np.uint8,offset=16).reshape(len(labels),784)
        
        return images, labels
    
    @classmethod
    def show_images(self, images):
        n = images.shape[0]
        _, plts = plt.subplots(1,n,figsize=(15,15))
        for i in range(n):
            plts[i].imshow(images[i].reshape((28,28)))
            plts[i].axes.get_xaxis().set_visible(False)
            plts[i].axes.get_yaxis().set_visible(False)
        plt.show()

    @classmethod
    def get_labels(self, label):
        text_labels = [
            't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
        ]
        return [[text_labels[i]] for i in label]