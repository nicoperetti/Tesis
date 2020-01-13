#coding: utf-8

import torch
import torch.nn as nn


class Bilinear(nn.Module):
    def __init__(self, in1_features, in2_features, bias=True):
        super().__init__()
        self.bilin = nn.modules.Bilinear(in1_features, in2_features, 1, bias=bias)

    def forward(self, X, Y):
        if X.ndimension() == 1:
            X = X.unsqueeze(0)
        # x = x.expand(y.size(0), x.size(1))
        # return self.bilin(x, y)
        else:
            X = [x.unsqueeze(0) for x in X]
        return [self.bilin(x.expand(y.size(0), x.size(1)), y) for x, y in zip(X, Y)]

    def project_x(self, X):
        if X.ndimension() == 1:
            X = X.unsqueeze(0)
        assert self.bilin.weight.size()[0] == 1
        return torch.mm(X, self.bilin.weight[0])

    def project_y(self, Y):
        if Y.ndimension() == 1:
            Y = Y.unsqueeze(0)
        assert self.bilin.weight.size()[0] == 1
        return torch.mm(Y, self.bilin.weight[0].transpose(1, 0))


class Linear(nn.Module):
    def __init__(self, in2_features, bias=True):
        super().__init__()
        self.linear = nn.modules.Linear(in2_features, 1, bias=bias)

    def forward(self, Y):
        return [self.linear(y) for y in Y]