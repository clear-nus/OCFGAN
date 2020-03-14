#  Provides residual blocks for ResNet.
#  Based on: https://github.com/ozanciga/gans-with-pytorch/blob/master/wgan-gp/models.py
#
#  Copyright (c) 2020 Abdul Fatir Ansari. All rights reserved.
#  This work is licensed under the terms of the MIT license.
#  For a copy, see <https://opensource.org/licenses/MIT>.
#
import torch.nn as nn
import torch
import numpy as np

class MeanPoolConv(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(MeanPoolConv, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        self.model = nn.Sequential(conv1)
    def forward(self, x):
        out = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4.0
        out = self.model(out)
        return out

class ConvMeanPool(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(ConvMeanPool, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        self.model = nn.Sequential(conv1)
    def forward(self, x):
        out = self.model(x)
        out = (out[:,:,::2,::2] + out[:,:,1::2,::2] + out[:,:,::2,1::2] + out[:,:,1::2,1::2]) / 4.0
        return out

class UpsampleConv(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super(UpsampleConv, self).__init__()

        self.model = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size-1)//2, bias=True)
        )
    def forward(self, x):
        x = x.repeat((1, 4, 1, 1)) 
        out = self.model(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_output, k_size, resample='up', bn=True, spatial_dim=None):
        super(ResidualBlock, self).__init__()

        self.resample = resample

        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size)
            self.conv2 = nn.Conv2d(n_output, n_output, k_size, padding=(k_size-1)//2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size)
            self.out_dim = n_output
        elif resample == 'down':
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = ConvMeanPool(n_input, n_output, k_size)
            self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size)
            self.out_dim = n_output
            self.ln_dims = [n_input, spatial_dim, spatial_dim]
        else:
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv2 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size-1)//2)
            self.conv_shortcut = None
            self.out_dim = n_input
            self.ln_dims = [n_input, spatial_dim, spatial_dim]

        self.model = nn.Sequential(
            nn.BatchNorm2d(n_input) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv1,
            nn.BatchNorm2d(self.out_dim) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv2,
        )

    def forward(self, x):
        if self.conv_shortcut is None:
            return x + self.model(x)
        else:
            return self.conv_shortcut(x) + self.model(x)

