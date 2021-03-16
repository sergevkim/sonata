from torch.nn import (
    Module,
)

from sonata.utils import ParametersCounter


class Discriminator(Module):
    def __init__(
            self,
        ):
        super().__init__()

    def forward(
            self,
            image,
        ):
        pass


if __name__ == '__main__':
    model = Discriminator()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    print(n_params)

