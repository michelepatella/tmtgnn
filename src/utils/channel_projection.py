import torch
import torch.nn as nn


class ChannelProjection(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(ChannelProjection, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

    def forward(self, x):
        return self.mlp(x)
