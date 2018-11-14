import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from util.models import GCN
from util.utils import sparse_mx_to_torch_sparse_tensor


#参数设置

#setting
hidden = 16
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 100
seed = 42
no_cuda = False
fastmode = False

#导入数据
# Load data
f1 = open('adj.pkl', 'rb')
adj = pkl.load(f1)
adj = sparse_mx_to_torch_sparse_tensor(adj)
f2 = open('features.pkl', 'rb')
features = pkl.load(f2)
features = torch.FloatTensor(features)
f3 = open('labels.pkl', 'rb')
labels = pkl.load(f3)
labels = torch.FloatTensor(labels)