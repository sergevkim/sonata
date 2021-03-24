from torch.nn import Module


class ParametersCounter:
    @staticmethod
    def count(
            model: Module,
            trainable: bool = False,
        ) -> int:
        if trainable:
            n_params = sum(
                p.numel() for p in model.parameters()
                if p.requires_grad
            )
        else:
            n_params = sum(p.numel() for p in model.parameters())

        return n_params

