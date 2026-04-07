from spatial.graph_conv import GraphConv
from utils.channel_projection import Linear


import torch
import torch.nn as nn


class GraphDiffusion(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(GraphDiffusion, self).__init__()
        self.nconv = GraphConv()
        self.mlp = Linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho
