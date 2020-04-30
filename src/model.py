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
        self.linear = nn.Linear(input_dim, input_dim)
        self.act = nn.ReLU()
        self.linear_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        N, SS, F = x.shape
        x = x.view(-1, F)
        x = self.act(self.linear(x))
        x = x.view(N, SS, -1)
        x = torch.prod(x, dim=1)
        x = self.linear_out(x)
        return x


