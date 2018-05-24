
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import numpy

class DataSet(object):
    def __init__(self, images, labels, dtype= dtypes.float32):
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid images dtype %r, expected uint8 or float32' %dtype)
        else:
            assert images.shape[0] == labels.shape[0],('images.shape: %s labels.shape: %s'%(images.shape, labels.shape))
            self._num_examples = images.shape[0]

            if dtype == dtypes.float32:
                #Convert from [0, 255] -> [0.0, 1.0]
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0/255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
          # Finished epoch
            self._epochs_completed += 1
          # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
          # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return (numpy.concatenate((images_rest_part, images_new_part), axis=0) , 
                      numpy.concatenate((labels_rest_part, labels_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
