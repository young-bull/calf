from mxnet.gluon import nn


class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)

    
class ResNet18(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet18, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            net.add(
                nn.BatchNorm(),
                nn.Conv2D(64, kernel_size=3, strides=1),
                nn.MaxPool2D(pool_size=3, strides=2),
                Residual(64),
                Residual(64),
                Residual(128, same_shape=False),
                Residual(128),
                Residual(256, same_shape=False),
                Residual(256),
                Residual(512, same_shape=False),
                Residual(512),
                nn.GlobalAvgPool2D(),
                nn.Dense(num_classes)
        )

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out


if __name__ == '__main__':
    from mxnet import nd
    net = ResNet18(10, verbose=True)
    net.initialize()
    x = nd.random.uniform(shape=(4, 3, 96, 96))
    y = net(x)
    print(y)