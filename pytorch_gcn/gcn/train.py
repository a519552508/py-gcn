from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')
import random
import time
import pickle as pkl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from util.models import GCN
from util.utils import sparse_mx_to_torch_sparse_tensor,load_data,accuracy


#setting
hidden = 16
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 100
seed = 42
fastmode =False

#load_data
# d = 1000
# g = nx.random_graphs.watts_strogatz_graph(d,8,0.1)# 小世界网络图，节点1000，每个节点8个邻居以0.1概率相连
# adj =nx.adjacency_matrix(g).astype(np.float32)
# adj =sparse_mx_to_torch_sparse_tensor(adj)
# #features = np.array(np.ones(d)).reshape([d,1])
# features = np.random.random(d).reshape([d,1])
# features = torch.FloatTensor(features)
# # labels = np.random.randint(5,size=d).reshape([d,1])
# labels = np.random.random(d).reshape([d,1])
# labels = torch.FloatTensor(labels)

#Load data
f1 = open('adj.pkl', 'rb')
adj = pkl.load(f1)
adj = sparse_mx_to_torch_sparse_tensor(adj)
f2 = open('features.pkl', 'rb')
features = pkl.load(f2)
features = torch.FloatTensor(features)
f3 = open('labels.pkl', 'rb')
labels = pkl.load(f3)
labels = torch.FloatTensor(labels)
print(labels)

adj, features, labels, idx_train,idx_val,idx_test = load_data('cora')


features, adj, labels = Variable(features), Variable(adj), Variable(labels)
print(type(features))

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=int(labels.max()) + 1,
            dropout=dropout)
print(model)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

def train(epoch):
    t = time.time()
    model.train()
    output = model(features, adj)
    print('output的类型',type(output),output.shape)
    print('labels的类型',type(labels),labels.shape)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    optimizer.zero_grad()
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output, labels)
    #acc_test = accuracy(output, labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]))
          #"accuracy= {:.4f}".format(acc_test.data[0]))

# Train model
t_total = time.time()
l1_train = []
for epoch in range(100):
    l1_train.append(train(epoch))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))



z=[l1_t.item() for l1_t in l1_train]
print(z)