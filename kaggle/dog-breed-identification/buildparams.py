import os
import sys
sys.path.append('..')
from utils.parameters import build_params
import collections

params_dic = collections.OrderedDict()
params_dic['num_epochs'] = [ 200, 3 ]
params_dic['learning_rate'] = [ 0.01, 0.05 ]
params_dic['weight_decay'] = [ 1e-4 ]
params_dic['lr_period'] = [ 80 ]
params_dic['lr_decay'] = [ 0.1 ]

build_params(params_dic, params_file="dog-params.csv")