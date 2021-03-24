from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    InstanceNorm2d,
    MaxPool2d,
    Module,
    ModuleDict,
    ReLU,
    Sequential,
)


class ConvBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = True,
            transposed: bool = False,
            norm: bool = False,
            instance_norm: bool = False,
            pool: bool = False,
            act: bool = True,
        ):
        super().__init__()
        block_ordered_dict = OrderedDict()
        block_ordered_dict['conv'] = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ) if not transposed else ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        if norm:
            block_ordered_dict['norm'] = BatchNorm2d(num_features=out_channels)
        if instance_norm:
            block_ordered_dict['norm'] = InstanceNorm2d(
                num_features=out_channels,
                affine=True,
                track_running_stats=True,
            )
        if pool:
            block_ordered_dict['pool'] = MaxPool2d(kernel_size=2)
        if act:
            block_ordered_dict['act'] = ReLU()
        self.conv_block = Sequential(block_ordered_dict)

    def forward(self, x):
        x = self.conv_block(x)

        return x


class ResBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
        ):
        super().__init__()
        self.res_block = Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False,
                instance_norm=True,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False,
                instance_norm=True,
                act=False,
            ),
        )

    def forward(self, x):
        x = x + self.residual_block(x)

        return x


if __name__ == '__main__':
    b = ConvBlock(5, 10)
    print(b)