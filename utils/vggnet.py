from mxnet.gluon import nn
from mxnet import init

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

def get_vgg(arch, ctx):
    vgg_net = nn.Sequential()
    with vgg_net.name_scope():
        vgg_net.add(
            vgg_stack(arch),
            nn.Flatten(),
            nn.Dense(4096, activation='relu'),
            nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'),
            nn.Dropout(0.5),
            nn.Dense(10))
    vgg_net.initialize(ctx=ctx, init=init.Xavier())
    return vgg_net