import sys
sys.path.append('..')
from utils import reorg_data

data_dir = '/gpu_data/datasets/dog-breed-identification'
label_file = 'labels.csv'
train_dir = 'train'
test_dir = 'test'
input_dir = 'train_valid_test'
valid_ratio = 0.1

reorg_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)