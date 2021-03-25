import torch
from torch.nn import (
    Module,
    Sequential,
)

from sonata.models.blocks import AdaResBlock, ConvBlock
from sonata.utils import ParametersCounter


class Decoder(Module):
    def __init__(
            self,
        ):
        super().__init__()
        self.ada_res_block_1 = AdaResBlock(
            in_channels=128,
            out_channels=128,
        )
        self.ada_res_block_2 = AdaResBlock(
            in_channels=128,
            out_channels=64,
        )
        self.conv_block_3 = ConvBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            transposed=True,
        )
        self.ada_res_block_4 = AdaResBlock(
            in_channels=32,
            out_channels=16,
        )
        self.ada_res_block_5 = AdaResBlock(
            in_channels=16,
            out_channels=16,
        )
        self.conv_block_6 = ConvBlock(
            in_channels=16,
            out_channels=8,
            kernel_size=4,
            stride=2,
            transposed=True,
        )
        self.ada_res_block_7 = AdaResBlock(
            in_channels=8,
            out_channels=3,
        )

    def forward(
            self,
            content,
            style,
            hooks,
        ):
        x = self.ada_res_block_1(content, style, content)
        x = self.ada_res_block_2(x, style, hooks[3])
        x = self.conv_block_3(x)
        x = self.ada_res_block_4(x, style, hooks[2])
        x = self.ada_res_block_5(x, style, hooks[1])
        x = self.conv_block_6(x)
        x = self.ada_res_block_7(x, style, hooks[0])

        return x, None


if __name__ == '__main__':
    model = Decoder()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    print(n_params)

    inputs = torch.randn(4, 128, 64, 64)
    style = torch.randn(4, 3)
    hooks = [
        torch.randn(4, 8, 256, 256),
        torch.randn(4, 16, 128, 128),
        torch.randn(4, 32, 128, 128),
        torch.randn(4, 128, 64, 64),
    ]
    outputs = model(inputs, style, hooks)
    print(outputs.shape)
