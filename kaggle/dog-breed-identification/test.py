import os
import sys
import numpy as np
from mxnet import nd
sys.path.append('..')
import utils
from train import get_net, train, test_data, input_str, train_valid_ds, batch_size

ctx = utils.try_gpu()
net = get_net(ctx)

param_file=""
if len(sys.argv) >= 2:
    param_file = str(sys.argv[1])
    print(param_file)
    if not os.path.exists(param_file):
        exit("Params file is not exist")
else:
    exit("Please give params file")


net.load_parameters(param_file)

i=0
outputs = []
for data, label in test_data:
    i += 1
    output = nd.softmax(net(data.as_in_context(ctx)))
    outputs.extend(output.asnumpy())
    print("\r%.2f" % (i/(10357/batch_size)), end='')

ids = sorted(os.listdir(input_str+'test/unknown'))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, outputs):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')