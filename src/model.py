'''
Adapted from https://github.com/weihua916/powerful-gnns
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def create_model(features, labels):
    input_dim = features.shape[-1]
    output_dim = len(np.unique(labels))
    model = MLP(input_dim, output_dim)
    return model


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU()
        self.linear_out = nn.Sequential(nn.Linear(output_dim, output_dim), nn.LogSoftmax())

    def forward(self, x):
        N, SS, F_ = x.shape
        x = x.view(-1, F_)
        x = F.normalize(self.act(self.linear(x)), dim=-1, p=2)
        x = x.view(N, SS, -1)
        x = torch.prod(x, dim=1)
        x = self.linear_out(x)
        return x


