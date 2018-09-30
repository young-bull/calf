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
from utils import ResNet164_v2, DenseNet

# 命令行需要输入 ResNet164_v2 和 DenseNet 的 Params 文件
param_file1 = sys.argv[1]
param_file2 = sys.argv[2]

if (param_file1 == None) or (param_file2 == None) or (False == os.path.isfile(param_file1)) or (False == os.path.isfile(param_file2)):
    print("please input params file")
    exit

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

ctx = utils.try_gpu()

net1 = ResNet164_v2(10)
net1.load_params(param_file1, ctx=ctx)
net1.hybridize()

net2 = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
net2.load_params(param_file2, ctx=ctx)
net2.hybridize()

preds = []
i=0
for data, _ in test_data:
    i += 1
    output1 = nd.softmax(net1(data.as_in_context(ctx)))
    output2 = nd.softmax(net2(data.as_in_context(ctx)))
    output = 0.95 * output1 + 0.96 * output2
    
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())
    print("%.2f" % (i/(300000/batch_size)))

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_ds.synsets[x])
df.to_csv('submission.csv', index=False)
