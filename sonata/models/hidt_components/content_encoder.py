import torch
from torch.nn import (
    Module,
    Sequential,
)

from sonata.models.blocks import ConvBlock, ResBlock
from sonata.utils import ParametersCounter


class ContentEncoder(Module):
    def __init__(self):
        super().__init__()
        self.res_block_1 = Sequential(
            ConvBlock(
                in_channels=3,
                out_channels=8,
            ),
            ResBlock(
                in_channels=8,
                out_channels=8,
            )
        )
        self.conv_block_2 = ConvBlock(
            in_channels=8,
            out_channels=16,
            stride=2,
        )
        self.res_block_3 = ResBlock(
            in_channels=16,
            out_channels=16,
        )
        self.res_block_4 = Sequential(
            ConvBlock(
                in_channels=16,
                out_channels=32,
            ),
            ResBlock(
                in_channels=32,
                out_channels=32,
            ),
        )
        self.conv_block_5 = ConvBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
        )
        self.res_block_6 = Sequential(
            ConvBlock(
                in_channels=64,
                out_channels=128,
            ),
            ResBlock(
                in_channels=128,
                out_channels=128,
            ),
        )
        self.conv_block_7 = ConvBlock(
            in_channels=128,
            out_channels=128,
        )

    def forward(self, image):
        hooks = []

        x = self.res_block_1(image)
        hooks.append(x)
        x = self.conv_block_2(x)
        x = self.res_block_3(x)
        hooks.append(x)
        x = self.res_block_4(x)
        hooks.append(x)
        x = self.conv_block_5(x)
        x = self.res_block_6(x)
        hooks.append(x)
        x = self.conv_block_7(x)

        return x, hooks


if __name__ == '__main__':
    model = ContentEncoder()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    print(n_params)

    inputs = torch.randn(4, 3, 256, 256)
    outputs = model(inputs)
    print(outputs[0].shape)
    print([hook.shape for hook in outputs[1]])
