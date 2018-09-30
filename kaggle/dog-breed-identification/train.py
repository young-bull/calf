import os
import sys
import datetime
from time import sleep

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd
from mxnet import image
from mxnet import autograd
from mxnet import gluon
from mxnet import init
from mxnet.gluon.data import vision

sys.path.append('..')
import utils
from utils import ResNet18
from utils import ResNet164_v2
from utils import DenseNet

input_str = '/gpu_data/datasets/dog-breed-identification/train_valid_test/'
batch_size = 32
if len(sys.argv) >= 4:
    batch_size = int(sys.argv[3])

def transform_train(data, label):
    im = image.imresize(data.astype('float32') / 255, 96, 96)
    auglist = image.CreateAugmenter(data_shape=(3, 96, 96), resize=0, 
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=None, std=None, 
                        brightness=0, contrast=0, 
                        saturation=0, hue=0, 
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

def transform_test(data, label):
    im = image.imresize(data.astype('float32') / 255, 96, 96)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train_stanford', flag=1, transform=transform_train)
valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1, transform=transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid', flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1, transform=transform_test)

# 使用 ImageFolderDataset + gluon.data.DataLoader 得到的data自带input和label，还能iter
loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')

# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def get_net(ctx):
    num_outputs = 120
    net_choice = 0
    if len(sys.argv) >= 3:
        net_choice = int(sys.argv[2])
    if net_choice == 1:
        net = ResNet164_v2(num_outputs)
    elif net_choice == 2:
        net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=num_outputs)
    else:
        net = ResNet18(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    net.hybridize()
    return net

# 这里不再使用K折交叉验证，而是依赖验证集的结果来调参。
# 验证集用此函数计算 valid_loss，保存起来供调参使用
def get_loss_acc(data, net, ctx=mx.cpu()):
    loss = acc = 0.0
    if isinstance(data, mx.io.MXDataIter):
        data.reset()
    for feas, label in data:
        label = label.as_in_context(ctx)
        output = net(feas.as_in_context(ctx))
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
        acc  += nd.mean(output.argmax(axis=1)==label).asscalar()
    return loss / len(data), acc / len(data)

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})

    total_time_start = datetime.datetime.now()
    detail = pd.DataFrame(columns=['train_loss','train_acc','valid_loss','valid_acc','lr'])

    prev_time = datetime.datetime.now()
    file_name = "lets go"
    for epoch in range(num_epochs):
        train_loss = train_acc = valid_loss = valid_acc = .0

        if  (epoch > 0 and epoch % lr_period == 0) : 
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc  += nd.mean(output.argmax(axis=1)==label).asscalar()

        train_loss = train_loss / len(train_data)
        train_acc  = train_acc / len(train_data)

        epoch_str = ("Epoch %d. Train loss: %f, Train acc: %f" % (epoch, train_loss, train_acc))
        if valid_data is not None:  
            valid_loss, valid_acc = get_loss_acc(valid_data, net, ctx)
            epoch_str += (", Valid loss: %f, Valid acc %f, " % (valid_loss, valid_acc))
        
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
        prev_time = cur_time

        detail.loc[len(detail)] = pd.Series({
                        'train_loss':train_loss,
                        'train_acc':train_acc,
                        'valid_loss':valid_loss,
                        'valid_acc':valid_acc,
                        'lr':trainer.learning_rate})
        
        file_name = ("%s-%d-[%d-%f-%f-%d-%f]-[%f-%f-%f-%f]" % 
            (net.name, batch_size, epoch, trainer.learning_rate, wd, lr_period, lr_decay, 
            train_loss, train_acc, valid_loss, valid_acc))
        
        # 每隔10轮or最后1轮：保存一下参数和训练过程
        if (epoch > 0 and epoch % 10 == 0) or (epoch == (num_epochs-1)):
            net.save_parameters(file_name+".params")
            detail.to_csv(file_name+".csv", index=False)

    print(datetime.datetime.now()-total_time_start)

if __name__ == '__main__':
    # 使用哪套参数
    num = -1
    if len(sys.argv) >= 2:
        num = int(sys.argv[1])
    params_file="dog-params.csv"

    tstart = datetime.datetime.now()
    for i in range(100 if(num==-1) else 1):
        # 重新获取 net
        ctx = utils.try_gpu()
        print(ctx)
        net = get_net(ctx)
        print(net.name)

        # 获取一套参数
        if num == -1 :
            # 从参数表+got表自动中获取一套待测的
            n,p = utils.get_todo_param(params_file=params_file)
            if p is None:
                break
        else:
            # 从参数表中获取指定的一套(num 从1开始)
            n,p = utils.get_param(num,params_file=params_file)
            n = 0

        print(p)

        # 开始训练
        print("="*60,"%i(%i) Start"%(i,n))
        # 这里添加 None 后，s取出来才是object的，s['num_epochs']才是int型的
        p['jiong'] = None
        s = p.iloc[0]
        train(net, train_data, valid_data, s['num_epochs'], s['learning_rate'], 
                s['weight_decay'], ctx, s['lr_period'], s['lr_decay'])
        print("="*60,"%i(%i) End"%(i,n))
        sleep(3)

    # 保存参数表
    print("="*60, "All End")
    print(datetime.datetime.now()-tstart)