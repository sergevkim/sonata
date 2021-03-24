import torch
from torch.nn import (
    Linear,
    Module,
    ReLU,
    Sequential,
)

from sonata.models.blocks import ResBlock
from sonata.utils import ParametersCounter


class ConditionalDiscriminator(Module):
    "Projection based discrminator, adapted from: https://github.com/XHChen0528/SNGAN_Projection_Pytorch"
    def __init__(self, num_feat=64):
        super().__init__()
        self.num_feat = num_feat
        self.activation = ReLU()
        self.blocks = [ResBlock(3, num_feat)]
        self.blocks.extend([
            ResBlock(
                num_feat*(2**i),
                num_feat*(2**(i+1)),
                stride=2,
            ) for i in range(4)
        ])

        self.l6 = torch.nn.utils.spectral_norm(
            Linear(num_feat * 16, 1)
        )
        self.style = torch.nn.utils.spectral_norm(
            Linear(3, num_feat * 16)
        )
        self._initialize()

    def _initialize(self):
        torch.nn.init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            torch.nn.init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        for block in self.blocks:
            x = block(x)

        h = self.activation(x)
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)

        if y is not None:
            output += torch.sum(self.style(y) * h, dim=1, keepdim=True)

        return output


if __name__ == '__main__':
    model = ConditionalDiscriminator()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    print(n_params)

