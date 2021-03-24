import pdb

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.optim.optimizer import Optimizer

from sonata.models import BaseModule
from sonata.models.hidt_components import (
    ContentEncoder,
    Decoder,
    Discriminator,
    StyleEncoder,
)
from sonata.utils import ParametersCounter


class HiDTModel(BaseModule):
    def __init__(
            self,
            device: torch.device = torch.device('cpu'),
            learning_rate: float = 3e-4,
            verbose: bool = True,
        ):
        super().__init__()
        self.device = device

        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.criterion_dist = MSELoss() #KL? distance between batch dist and standart normal
        self.criterion_rec = L1Loss() #+
        self.criterion_seg = CrossEntropyLoss() #+
        self.criterion_c = L1Loss() #+
        self.criterion_s = L1Loss() #+
        self.criterion_cyc = L1Loss() #+
        self.criterion_seg_r = CrossEntropyLoss() #+
        self.criterion_c_r = L1Loss() #+
        self.criterion_s_r = L1Loss() #+
        self.criterion_rec_r = L1Loss() #+

    def forward(
            self,
            x,
        ):
        pass

    def training_step(
            self,
            batch,
            batch_idx,
            optimizer_idx,
        ):
        x = batch
        x = x.to(self.device)
        x_prime = x_prime.to(self.device) #TODO think about sampling

        if optimizer_idx == 0: #generator step
            #autoencoding branch
            c = self.content_encoder(x)
            s = self.style_encoder(x)
            loss_dist = self.criterion_dist(s) #TODO check that s distribution looks like N(0, I)
            x_tilde, m = self.generator(
                content=c,
                style=s,
            )
            loss_rec = self.criterion_rec(x_tilde, x)

            #swapping branch TODO discriminator
            s_prime = self.style_encoder(x_prime)
            x_hat, m_hat = self.generator(
                content=c,
                style=s_prime,
            )
            loss_seg = self.criterion_seg(m_hat, m)
            c_hat = self.content_encoder(x_hat)
            s_hat = self.style_encoder(x_hat)
            loss_c = self.criterion_c(c_hat, c)
            loss_s = self.criterion_s(s_hat, s_prime)

            c_prime = self.content_encoder(x_prime)
            x_prime_hat, _ = self.generator(
                content=c_prime,
                style=s,
            )
            s_prime_hat = self.style_encoder(x_prime_hat)
            x_hat_tilde = self.generator(
                content=c_hat,
                style=s_prime_hat,
            )
            loss_cyc = self.criterion_cyc(x_hat_tilde, x)

            #noise branch TODO discriminator
            s_r = torch.randn(size=None) #TODO think about size
            x_r, m_r = self.generator(
                content=c,
                style=s_r,
            )
            loss_seg_r = self.criterion_seg_r(m_r, m)
            c_r_tilde = self.content_encoder(x_r)
            s_r_tilde = self.style_encoder(x_r)
            loss_c_r = self.criterion_c_r(c_r_tilde, c)
            loss_s_r = self.criterion_s_r(s_r_tilde, s_r)
            x_r_tilde, _ = self.generator(
                content=c_r_tilde,
                style=s_r_tilde,
            )
            loss_rec_r = self.criterion_rec_r(x_r_tilde, x_r)

            loss_terms = [
                loss_adv + loss_adv_r,
                loss_rec + loss_rec_r + loss_cyc,
                loss_seg + loss_seg_r,
                loss_c + loss_c_r,
                loss_s,
                loss_s_r,
                loss_dist,
            ]
            lambdas = [1/7 for i in range(7)]

            loss = 0
            for i in range(7):
                loss += lambdas[i] * loss_terms[i]

            return loss

        if optimizer_idx == 1: #discriminator step
            pass

    def validation_step(
            self,
            batch,
            batch_idx,
        ):
        loss = self.training_step(
            batch=batch,
            batch_idx=batch_idx,
            optimizer_idx=0,
        )

        return loss

    def configure_optimizers(
            self,
        ):
        optimizer = Adam(
            params=self.generator.parameters(), # TODO union params with encoders params
            lr=self.learning_rate,
        )

        return [optimizer], []


if __name__ == '__main__':
    model = HiDTModel()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    pdb.set_trace()
    print(n_params)

