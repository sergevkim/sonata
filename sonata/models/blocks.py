from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Identity,
    InstanceNorm2d,
    Linear,
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
            padding: int = 1,
        ):
        super().__init__()
        self.res_block = Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                instance_norm=True,
            ),
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                instance_norm=True,
                act=False,
            ),
        )

    def forward(self, x):
        x = x + self.res_block(x)

        return x


def get_adastats(features):
    bs, c = features.shape[:2]
    features = features.view(bs, c, -1)
    mean = features.mean(dim=2).view(bs,c,1,1)
    std = features.var(dim=2).sqrt().view(bs,c,1,1)

    return mean, std


def AdaIN(content_feat, style_feat):
    #calculating channel and batch specific stats
    smean, sstd = get_adastats(style_feat)
    cmean, cstd = get_adastats(content_feat)
    csize = content_feat.size()
    norm_content = (content_feat - cmean.expand(csize)) / cstd.expand(csize)

    return norm_content * sstd.expand(csize) + smean.expand(csize)


class AdaSkipBlock(Module):
    def __init__(
            self,
            in_channels,
            out_channels,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.ada_creator = nn.Sequential(
            Linear(
                in_features=3,
                out_features=16,
            ),
            Linear(
                in_features=16,
                out_features=64,
            ),
            Linear(
                in_features=64,
                out_features=256,
            ),
        )
        self.ada = AdaIN
        self.dense = ConvBlock(
            in_channels=in_channels*2,
            out_channels=in_channels,
        )

    def forward(self, content, style, hook):
        x = self.ada_creator(style)
        ada_params = x.view((x.shape[0], self.in_channels, -1))
        ada = self.ada(hook, ada_params)
        combined = torch.cat([content, ada], dim=1)
        x = self.dense(combined)

        return x


class AdaResBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
        ):
        super().__init__()

        self.ada_block = AdaSkipBlock(
            in_channels=in_channels,
            out_channels=in_channels,
        )
        self.res_block = Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
            ),
            ResBlock(
                in_channels=out_channels,
                out_channels=out_channels,
            ),
        )
        if in_channels != out_channels:
            self.skip = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                act=False,
            )
        else:
            self.skip = Identity()

    def forward(
            self,
            content,
            style,
            hook,
        ):
        ada = self.ada_block(content, style, hook)
        res = self.res_block(ada)

        if self.skip is not None:
            content = self.skip(content)

        return res + content


if __name__ == '__main__':
    b = ConvBlock(5, 10)
    print(b)