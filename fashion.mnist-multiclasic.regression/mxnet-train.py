
# coding: utf-8

import os
import sys
import datetime

import numpy as np
import pandas as pd
#import matplotlib.pylab as plt

import mxnet as mx
from mxnet import ndarray as nd, autograd as ag, gluon
from mxnet.gluon import nn
from mxnet import init
from mxnet import image

from dataloader import DataLoader

print("python: %r" % sys.version)
print("numpy: %r" % np.__version__)
#print("matplotlib: %r" % plt.__version__)
print("pandas: %r" % pd.__version__)
print("mxnet: %r" % mx.__version__)

ctx = mx.gpu(0)
batch_size = 100
print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

dl = DataLoader('/gpu_data/datasets/fashion-mnist/')
train_img,train_labels = dl.get_data(kind='train')
test_img,test_labels = dl.get_data(kind='t10k')


train_img.shape

# 修改为float以便求导，除以255做归一化
train_img_nd = nd.array(train_img).astype(np.float32)/255
train_lab_nd = nd.array(train_labels).astype(np.float32)
test_img_nd = nd.array(test_img).astype(np.float32)/255
test_lab_nd = nd.array(test_labels).astype(np.float32)

def data_iter(batch_size=100, kind='train'):
    if kind != 'train':
        idx = list(range(len(test_labels)))
        for i in range(0,len(test_labels), batch_size):
            j = nd.array(idx[i:min(i+batch_size,len(test_labels))])
            yield nd.take(test_img_nd,j).as_in_context(ctx), nd.take(test_lab_nd,j).as_in_context(ctx)
    else:
        idx = list(range(len(train_labels)))
        for i in range(0,len(train_labels), batch_size):
            j = nd.array(idx[i:min(i+batch_size,len(train_labels))])
            yield nd.take(train_img_nd,j).as_in_context(ctx), nd.take(train_lab_nd,j).as_in_context(ctx)

num_input = 28*28
num_output = 10


W = nd.random_normal(shape=(num_input, num_output),ctx=ctx)
b = nd.random_normal(shape=(num_output),ctx=ctx)

wb_params = [W,b]
for p in wb_params:
    p.attach_grad()

def net_wb(X):
    return nd.dot(X,W) + b

def relu(X):
    return nd.maximum(X,0)

num_hidden = 256
weight_scale = .01

# MXNet 的公式是 y=XW+b，吴恩达的公式是 y=WX+b，所以两者W的shape定义是相反的。
W1 = nd.random_normal(shape=(num_input,num_hidden), scale=weight_scale,ctx=ctx)
b1 = nd.zeros(num_hidden,ctx=ctx)

W2 = nd.random_normal(shape=(num_hidden,num_output),scale=weight_scale,ctx=ctx)
b2 = nd.zeros(num_output,ctx=ctx)

mlp_params = [W1,b1,W2,b2]
for p in mlp_params:
    p.attach_grad()

def net_mlp(X):
    X = X.reshape((-1, num_input))
    h1 = relu(nd.dot(X,W1) + b1)
    output = nd.dot(h1,W2) + b2
    return output

mlp_net = gluon.nn.Sequential()
with mlp_net.name_scope():
    mlp_net.add(gluon.nn.Dense(256, activation="relu"))
    mlp_net.add(gluon.nn.Dropout(0.5))
    mlp_net.add(gluon.nn.Dense(10))
mlp_net.initialize(ctx=ctx)

def net_mlp_gluon(X):
    return mlp_net(X)

weight_scale = .01

# output channels = 20, kernel = (5,5)
lenet_W1 = nd.random_normal(shape=(20,1,5,5), scale=weight_scale, ctx=ctx)
lenet_b1 = nd.zeros(lenet_W1.shape[0], ctx=ctx)

# output channels = 50, kernel = (3,3)
lenet_W2 = nd.random_normal(shape=(50,20,3,3), scale=weight_scale, ctx=ctx)
lenet_b2 = nd.zeros(lenet_W2.shape[0], ctx=ctx)

# output dim = 128
lenet_W3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=ctx)
lenet_b3 = nd.zeros(lenet_W3.shape[1], ctx=ctx)

# output dim = 10
lenet_W4 = nd.random_normal(shape=(lenet_W3.shape[1], 10), scale=weight_scale, ctx=ctx)
lenet_b4 = nd.zeros(lenet_W4.shape[1], ctx=ctx)

lenet_params = [lenet_W1, lenet_b1, lenet_W2, lenet_b2, lenet_W3, lenet_b3, lenet_W4, lenet_b4]
for param in lenet_params:
    param.attach_grad()

def net_lenet(X, verbose=False):    
    # 第一层卷积
    h1_conv = nd.Convolution(
        data=X, weight=lenet_W1, bias=lenet_b1, kernel=lenet_W1.shape[2:], num_filter=lenet_W1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(
        data=h1_activation, pool_type="max", kernel=(2,2), stride=(2,2))
    # 第二层卷积
    h2_conv = nd.Convolution(
        data=h1, weight=lenet_W2, bias=lenet_b2, kernel=lenet_W2.shape[2:], num_filter=lenet_W2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(2,2), stride=(2,2))
    h2 = nd.flatten(h2)
    # 第一层全连接
    h3_linear = nd.dot(h2, lenet_W3) + lenet_b3
    h3 = nd.relu(h3_linear)
    # 第二层全连接
    h4_linear = nd.dot(h3, lenet_W4) + lenet_b4
    if verbose:
        print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('1st dense:', h3.shape)
        print('2nd dense:', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear

lenet = nn.Sequential()
with lenet.name_scope():
    lenet.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )
lenet.initialize(ctx=ctx)

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

# data shape: (batchsize, height, width, channel)
# out  shape: (batchsize, channel, height, width)
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
        return lenet(data.astype('float32'))
    elif net_type == 'vgg_gluon':
        data = data.reshape((data.shape[0],28,28,1))
        data = resize_img(data, resize=96)
        data = data.as_in_context(ctx)
        return vgg_net(data.astype('float32'))
    else:
        return None

def softmax(X):
    e = nd.exp(X)
    t = nd.sum(e, axis=1, keepdims=True)
    return e/t

def cross_entropy(yhat,y):
    return - nd.log(nd.pick(yhat,y)) 

loss_softmax_ce = gluon.loss.SoftmaxCrossEntropyLoss()

def SGD(params, lr):
    for p in params:
        p[:] = p - lr*p.grad

sgd = mx.optimizer.SGD()

def get_trainer(net):
    return gluon.Trainer(net.collect_params(), 
                        'sgd', 
                        {'learning_rate': 0.5})

def accuracy(yhat,y):
    return nd.mean(yhat.argmax(axis=1)==y).asscalar()

def train(epochs=5, learning_rate=.5, net_type='wb', params=wb_params, trainer=None):
    t1 = datetime.datetime.now()

    prev_time = datetime.datetime.now()
    for e in range(epochs):
        train_loss = 0.
        train_accu = 0.
        test_accu  = 0.

        i = 0
        for data,label in data_iter(batch_size=batch_size):
            i += 1
            print('\r%4d %d%%' % (i, 100*i/(train_img.shape[0]/batch_size)), end='')
            with ag.record():
                yhat = net(data, net_type)
                #loss = cross_entropy(softmax(yhat),label)
                loss = loss_softmax_ce(yhat,label)
            loss.backward()
            if trainer == None:
                SGD(params, learning_rate/batch_size)
            else:
                trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_accu += accuracy(yhat,label)

        for data,label in data_iter(batch_size=batch_size,kind='test'):
            data  = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            yhat = net(data, net_type)
            test_accu += accuracy(yhat, label)
        
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print("\r%s epoch:%d; loss:%f; Train accu:%f; Test accu:%f; Time:%s" % (
                        net_type, e, 
                        train_loss/len(train_labels)*batch_size,
                        train_accu/len(train_labels)*batch_size,
                        test_accu/len(train_labels)*batch_size,
                        time_str))

    print(datetime.datetime.now()-t1)



train(net_type='wb', params=wb_params)

train(net_type='mlp', params=mlp_params)

train(net_type='mlp_gluon',trainer=get_trainer(mlp_net))

train(net_type='lenet', params=lenet_params)

train(net_type='lenet_gluon', trainer=get_trainer(lenet),learning_rate=0.2)
lenet.save_parameters("lenet-params-"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

train(net_type='vgg_gluon', trainer=get_trainer(vgg_net), epochs=3)
vgg_net.save_parameters("vgg-net-params-"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

