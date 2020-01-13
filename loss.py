#coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable

CUDA_AVAILABLE = torch.cuda.is_available()


class SJELoss1(nn.Module):
    """ Structured Joint Embedding loss
    """
    def __init__(self):
        super().__init__()

    def _loss(self, f):
        M = f.size(0)
        loss = torch.clamp(1.0 + f[1:] - f[0].expand(M-1, 1), min=0.)
        return loss.max()

    def forward(self, batch):
        loss = [self._loss(f) for f in batch]
        return sum(loss)  # returns Variable with requires_grad=True


class SJELoss2(nn.Module):
    """ Structured Top-1 hit loss
    """
    def __init__(self):
        super().__init__()

    def _loss(self, f):
        M = f.size(0)
        margin = Variable(1. - 1. / torch.arange(1, M+1), requires_grad=False)
        if CUDA_AVAILABLE:
            margin = margin.cuda()
        loss = torch.clamp(margin.unsqueeze_(1) + f - f[0].expand(M, 1), min=0.)
        return loss.max()

    def forward(self, batch):
        loss = [self._loss(f) for f in batch]
        return sum(loss)


class SRankingLoss(nn.Module):
    """ Structured Ranking loss
    """
    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = int(top_k)

    def _loss(self, f):
        M = f.size(0)
        top_k = min(self.top_k, M-1)
        loss = Variable(torch.FloatTensor([0]))
        if CUDA_AVAILABLE:
            loss = loss.cuda()
        for k in range(1, top_k+1):
            margin = Variable(1. / torch.arange(k, M), requires_grad=False)
            if CUDA_AVAILABLE:
                margin = margin.cuda()
            loss_k = torch.clamp(margin.unsqueeze_(1) + f[k:] - f[k-1].expand(M-k, 1), min=0.)
            loss += loss_k.sum()
        return loss

    def forward(self, batch):
        loss = [self._loss(f) for f in batch]
        return sum(loss)


class SJELoss1Square(nn.Module):
    """Structured Joint Embedding loss."""

    def __init__(self):
        """Init method."""
        super().__init__()

    def _loss(self, f):
        """Loss."""
        m = f.size(0)
        loss = 1.0 + torch.pow(torch.clamp(f[1:] - f[0].expand(m - 1, 1), min=0.), 2)
        return loss.max()

    def forward(self, batch):
        """Forward pass."""
        loss = [self._loss(f) for f in batch]
        return sum(loss)
