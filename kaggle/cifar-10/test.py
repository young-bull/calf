import os
import sys
import datetime

import numpy as np
import pandas as pd

from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision

sys.path.append('..')
import utils
from utils import transform_train, transform_test
from utils import ResNet18 as ResNet

# 命令行需要输入 ResNet 的 Params 文件
param_file = sys.argv[1]
if (param_file == None) or (False == os.path.isfile(param_file)):
    print("please input params file")
    exit

print(param_file)

data_dir = '/gpu_data/datasets/cifar-10'
label_file = 'trainLabels.csv'
train_dir = 'train'
test_dir = 'test'
input_dir = 'train_valid_test'
batch_size = 128

input_str = data_dir + '/' + input_dir + '/'

# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1, transform=transform_test)
loader = gluon.data.DataLoader
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')

# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def get_net(ctx):
    num_outputs = 10
    #net = gluon.model_zoo.vision.densenet121(classes=num_outputs,ctx=ctx)
    net = ResNet(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    return net


ctx = utils.try_gpu()
net = get_net(ctx)
net.hybridize()
net.load_parameters(param_file)

preds = []
i=0
for data, label in test_data:
    i += 1
    output = net(data.as_in_context(ctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())
    #print(output.argmax(axis=1).astype(int).asnumpy())
    print("%.2f" % (i/(300000/batch_size)))

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))
#print(sorted_ids)
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_ds.synsets[x])
df.to_csv('submission.csv', index=False)
