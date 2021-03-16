from torch.nn import (
    Module,
)

from sonata.utils import ParametersCounter


class Generator(Module):
    def __init__(
            self,
        ):
        super().__init__()

    def forward(
            self,
            content,
            style,
        ):
        pass


if __name__ == '__main__':
    model = Generator()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    print(n_params)

