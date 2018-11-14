import pickle as pkl
import torch
import scipy.sparse as sp
import sys
import numpy as np
import networkx as nx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """把scipy稀疏矩阵转换成torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col.astype(np.float32)))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(file_name):
    #解析索引文件
    index =[]
    for line in open(file_name):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask.
        l:n_sample
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """将稀疏矩阵转换为元组表示"""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """规范化特征矩阵并转换为元组表示"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

#数据适配模块
def load_data(data_str):
    '''
    从gcn / data目录加载输入数据

    ind.dataset_str.x =>训练实例的特征向量为scipy.sparse.csr.csr_matrix对象;
    ind.dataset_str.tx =>测试实例的特征向量为scipy.sparse.csr.csr_matrix对象;
    ind.dataset_str.allx =>标记和未标记的训练实例的特征向量
        （ind.dataset_str.x的超集）作为scipy.sparse.csr.csr_matrix对象;
    ind.dataset_str.y =>标记的训练实例的单热标签为numpy.ndarray对象;
    ind.dataset_str.ty =>测试实例的单热标签为numpy.ndarray对象;
    ind.dataset_str.ally => ind.dataset_str.allx中实例的标签为numpy.ndarray对象;
    ind.dataset_str.graph =>格式为{index：[index_of_neighbor_nodes]}的dict为collections.defaultdict
        宾语;
    ind.dataset_str.test.index =>图表中测试实例的索引，归纳设置为列表对象。

    必须使用python pickle模块保存上面的所有对象。
    :param data_str:
    :return:
    '''
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../gcn/{}/ind.{}.{}".format(data_str, data_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../gcn/{}/ind.{}.test.index".format(data_str,data_str))
    test_idx_range = np.sort(test_idx_reorder)

    # 构建标准化特征矩阵 (2708,1433)
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize_features(features)

    # build symmetric adjacency matrix 构建对称邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    
    #labels单热向量编码 (2078,7)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 1000)

    #     train_mask = sample_mask(idx_train, labels.shape[0])
    #     val_mask = sample_mask(idx_val, labels.shape[0])
    #     test_mask = sample_mask(idx_test, labels.shape[0])

    #     y_train = np.zeros(labels.shape)
    #     y_val = np.zeros(labels.shape)
    #     y_test = np.zeros(labels.shape)
    #     y_train[train_mask, :] = labels[train_mask, :]
    #     y_val[val_mask, :] = labels[val_mask, :]
    #     y_test[test_mask, :] = labels[test_mask, :]
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test