from torch.nn import (
    Module,
    Sequential,
)

from sonata.models.blocks import ConvBlock, ResBlock
from sonata.utils import ParametersCounter


class ContentEncoder(Module):
    def __init__(self, ni=3):
        super().__init__()
        self.model = Sequential(
            ResBlock(
                in_channels=3,
                out_channels=8,
            ),
            ConvBlock(
                in_channels=8,
                out_channels=16,
                stride=2,
            ),
            ResBlock(
                in_channels=16,
                out_channels=16,
            ),
            ResBlock(
                in_channels=16,
                out_channels=32,
            ),
            ConvBlock(
                in_channels=32,
                out_channels=64,
                stride=2,
            ),
            ResBlock(
                in_channels=64,
                out_channels=128,
            ),
            ConvBlock(
                in_channels=128,
                out_channels=128,
            ),
        )

    def forward(self, x):
        x = self.model(x)

        return x


if __name__ == '__main__':
    model = ContentEncoder()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    print(n_params)

