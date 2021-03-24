from torch.nn import (
    Flatten,
    Linear,
    Module,
    Sequential,
)

from sonata.models.blocks import ConvBlock, ResBlock
from sonata.utils import ParametersCounter


class UnconditionalDiscriminator(Module):
    def __init__(
            self,
        ):
        super().__init__()
        self.model = Sequential(
            ConvBlock(
                in_channels=3,
                out_channels=64,
                stride=2,
            ),
            ResBlock(
                in_channels=64,
                out_channels=128,
            ),
            ResBlock(
                in_channels=128,
                out_channels=128,
            ),
            ConvBlock(
                in_channels=128,
                out_channels=1,
            ),
            Flatten(),
            Linear(128, 1),
        )

    def forward(
            self,
            image,
        ):
        x = self.model(image)

        return x


if __name__ == '__main__':
    model = UnconditionalDiscriminator()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    print(n_params)

