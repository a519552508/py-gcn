import math

import torch

from torch.nn.parameter import Parameter
import torch.nn as nn


class SparseMM(torch.autograd.Function):
    """"""
    def forward(self,matrix1,matrix2):
        """为backward方法中提供matrix1,matrix2对象"""
        self.save_for_backward(matrix1,matrix2)
        return torch.mm(matrix1,matrix2)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None

        if self.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class GraphConvolution(nn.Module):
    """
    简单GCN层，similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Parameter 是torch.autograd.Variable的一个字类，常被用于Module的参数。例如权重和偏置。
        # self.weight = Parameter(torch.Tensor(in_features,out_features))
        # if bias :
        #     self.bias =Parameter(torch.Tensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        # Parameter 是torch.autograd.Variable的一个字类，常被用于Module的参数。例如权重和偏置。
        # Parameter(tensor, requires_grad代表要求梯度)
        self.weight = torch.nn.Parameter(torch.zeros(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.weight.data, gain=math.sqrt(2))  # 均匀分布生成值，填充输入的张量或变量
        self.bias = torch.nn.Parameter(torch.rand(out_features))
        # self.reset_parameters()

    def reset_parameters(self):
        stdv =1. /math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self,input, adj):
        support = torch.mm(input,self.weight)
        output = SparseMM()(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'