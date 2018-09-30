import os
import sys
import datetime

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd, autograd as ag, gluon
from mxnet.gluon import nn
from mxnet import init
from mxnet import image

from dataloader import DataLoader

ctx = mx.gpu()
batch_size = 50

dl = DataLoader('/gpu_data/fashion-mnist/')
train_img,train_labels = dl.get_data(kind='train')
test_img,test_labels = dl.get_data(kind='t10k')

train_img_nd = nd.array(train_img).astype(np.float32)/255
train_lab_nd = nd.array(train_labels).astype(np.float32)
test_img_nd = nd.array(test_img).astype(np.float32)/255
test_lab_nd = nd.array(test_labels).astype(np.float32)

arch_A = ((1,64),(1,128),(2,256),(2,512),(2,512))
arch_B = ((2,64),(2,128),(2,256),(2,512),(2,512))
arch_D = ((2,64),(2,128),(3,256),(3,512),(3,512))
arch_E = ((2,64),(2,128),(4,256),(4,512),(4,512))

def vgg_stack(arch):
    out = nn.Sequential()
    for (num_convs, channels) in arch:
        seq = nn.Sequential()
        for _ in range(num_convs):
            seq.add(nn.Conv2D(channels=channels,kernel_size=3,
                          padding=1,activation='relu'))
        seq.add(nn.MaxPool2D(pool_size=2, strides=2))
        out.add(seq)
    return out
    
vgg_net = nn.Sequential()
with vgg_net.name_scope():
    vgg_net.add(
        vgg_stack(arch_A),
        nn.Flatten(),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'),
        nn.Dropout(0.5),
        nn.Dense(10))
vgg_net.initialize(ctx=ctx, init=init.Xavier())

def net_vgg_gluon(X):
    return vgg_net(X)

def resize_img(data, resize=None):
    if resize:
        n = data.shape[0]
        new_data = nd.zeros((n, resize, resize, data.shape[3]))
        for i in range(n):
            new_data[i] = image.imresize(data[i].as_in_context(mx.cpu()), 
                                         resize, resize)
        data = new_data
    return nd.transpose(data.astype('float32'),(0,3,1,2))/255

def net(data, net_type='wb'):
    data = data.as_in_context(ctx)
    # data.shape: 100*784
    if net_type == 'wb':
        return net_wb(data)
    elif net_type == 'mlp':
        return net_mlp(data)
    elif net_type == 'mlp_gluon':
        return net_mlp_gluon(data)
    elif net_type == 'lenet':
        data = data.reshape((data.shape[0],1,28,28))
        return net_lenet(data.astype('float32'))
    elif net_type == 'lenet_gluon':
        data = data.reshape((data.shape[0],1,28,28))
        return net_lenet_gluon(data.astype('float32'))
    elif net_type == 'vgg_gluon':
        data = data.reshape((data.shape[0],28,28,1))
        data = resize_img(data, resize=96)
        data = data.as_in_context(ctx)
        print(data.shape)
        return net_vgg_gluon(data.astype('float32'))
    else:
        return None

vgg_net.load_params("vgg-net-params", ctx=ctx)

result = net(train_img_nd[:100],net_type='vgg_gluon').argmax(axis=1).asnumpy()
print(result)
print(train_lab_nd[:10].asnumpy())

params = vgg_net.collect_params()
print(params)
for k in params.keys():
    print(params[k].data())
    break
