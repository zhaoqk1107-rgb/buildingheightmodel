import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init="xavier"):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        if init == "uniform":
            # print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == "xavier":
            # print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == "kaiming":
            # print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)  # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode="fan_in")
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        # support = torch.mm(input, self.weight) #  self.weight(200,256)
        # output = torch.spmm(adj, support)
        # Qikang Zhao 改：
        # input: (B, N, in_features) -> e.g., (4, 200, 256)
        # weight: (in_features, out_features) -> e.g., (256, 512)
        # 使用 torch.matmul 代替 torch.mm 以支持 Batch 维度
        # (B, N, in) @ (in, out) -> (B, N, out)
        support = torch.matmul(input, self.weight)
        # adj: (B, N, N) -> e.g., (4, 200, 200)
        # support: (B, N, out) -> e.g., (4, 200, 512)
        # 使用 torch.matmul 代替 torch.spmm，因为 adj 在这里是 Dense Tensor 且带 Batch
        # (B, N, N) @ (B, N, out) -> (B, N, out)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class Feat2Graph(nn.Module):
    def __init__(self, num_feats):
        super(Feat2Graph, self).__init__()
        self.wq = nn.Linear(num_feats, num_feats)
        self.wk = nn.Linear(num_feats, num_feats)

    def forward(self, x):
        qx = self.wq(x)
        kx = self.wk(x)
        dot_mat = qx.matmul(kx.transpose(-1, -2))
        adj = F.normalize(dot_mat.square(), p=1, dim=-1)
        return x, adj


class GCN(nn.Module):
    def __init__(self, nfeat=256, nhid=512, dropout=False, init="xavier"):
        super(GCN, self).__init__()
        self.graph = Feat2Graph(nfeat)
        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nhid, init=init)
        self.gc3 = GraphConvolution(nhid, nfeat, init=init)
        self.dropout = dropout

    def forward(self, x): # x: (B, 200(nq), 256);
        x1, adj = self.graph(x) # x1: (B, 200(nq), 256); adj: (B, 200(nq), 200)
        x2 = F.relu(self.gc1(x1, adj))
        x3 = F.relu(self.gc2(x2, adj))
        x4 = F.relu(self.gc3(x3, adj))
        return x4


if __name__ == "__main__":
    feats = torch.randn(200, 128).cuda()

    model = GCN(128, 256).cuda()
    out = model(feats)
    print()
