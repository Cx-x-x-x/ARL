from arl_1 import Block, BasicBlock

save_dir = 'model_112/'
tensorboard_dir = 'sp_111/112/'

# /Disk1/chenxin/model/model_73/net_040.pth
# /Disk1/chenxin/model/resnet50-19c8e357.pth
pthfile = '/Disk1/chenxin/model/resnet18-5c106cde.pth'

block = BasicBlock  # (add_softmax='channel', add_dropout='conv2')
layers = [2, 2, 2, 2]
dropout = 'fc'

factor_init = 0.01

Optimizer = 'sgd'
lr = 0.001
BatchSize = 64
wd = 7.11E-04

Epoch = 100



