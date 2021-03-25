from typing import Optional

import numpy as np
import scipy.linalg as linalg
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.models import inception_v3 as Inception


class MetricCalculator:
    @staticmethod
    def calculate_frechet_distance(
            mu_1: Tensor,
            mu_2: Tensor,
            sigma_1: Tensor,
            sigma_2: Tensor,
            eps: float = 1e-6,
        ):
        mu_1 = mu_1.numpy()
        mu_2 = mu_2.numpy()
        sigma_1 = sigma_1.numpy()
        sigma_2 = sigma_2.numpy()

        delta_mu = mu_1 - mu_2

        offset = np.eye(sigma_1.shape[0]) * eps

        covmean, _ = linalg.sqrtm(
            (sigma_1 + offset) @ (sigma_2 + offset),
            disp=False,
        )

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        distance = (
            delta_mu @ delta_mu +
            sigma1.trace() +
            sigma2.trace() -
            2 * covmean.trace()
        )

        return distance

    @staticmethod
    def calculate_activation_statistics(
            model,
            dataloader,
            inception,
        ):

        device = torch.device(
            'cuda' if next(a.parameters()).is_cuda else
            'cpu',
        )
        true_vecs, ae_vecs = [], []

        for images, _ in dataloader:
            bs = len(images)
            images = images.to(device)
            true_vecs.append(inception(images)[0].reshape(bs, -1))
            ae_vecs.append(inception(model.generate(images))[0].reshape(bs, -1))

        true_data = torch.stack(
            true_vecs,
        ).detach().cpu().view(bs * len(dataloader), -1).numpy()
        ae_data = torch.stack(
            ae_vecs,
        ).detach().cpu().view(bs * len(dataloader), -1).numpy()
        mu_1 = true_data.mean(axis=1)
        mu_2 = ae_data.mean(axis=1)

        sigma_1 = np.cov(true_data, rowvar=False)
        sigma_2 = np.cov(ae_data, rowvar=False)

        return mu_1, mu_2, sigma_1, sigma_2

    @classmethod
    @torch.no_grad()
    def calculate_fid(
            cls,
            model: Module,
            dataloader: DataLoader,
            classifier: Optional[Module] = None,
        ):
        if classifier is None:
            classifier = Inception(
                pretrained=True,
                progress=True,
            )
        model.eval()
        classifier.eval()

        mu_1, mu_2, sigma_1, sigma_2 = cls.calculate_activation_statistics(
            model=model,
            dataloader=dataloader,
            inception=classifier,
        )
        fid_value = cls.calculate_frechet_distance(
            mu_1=mu_1,
            mu_2=mu_2,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
        )

        return fid_value

    @classmethod
    def criterion_adv(cls, val, targ):
        return 0.5 * torch.mean((val - targ) ** 2)

    @staticmethod
    def cov(m):
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        mt = m.t()
        return fact * m.matmul(mt).squeeze()

    @staticmethod
    def one_norm(p1, p2):
        return (p1 - p2).abs().sum()

    @classmethod
    def criterion_dist(cls, style):
        styles = style#torch.cat([orig_style, orig2_style])
        smeans = styles.mean()
        cov_m = cls.cov(styles)
        cov_diag = torch.diag(cov_m)

        loss = (
            cls.one_norm(smeans, torch.ones(1)) +
            cls.one_norm(cov_m, torch.eye(cov_m.shape[0])) +
            cls.one_norm(cov_diag, torch.ones(cov_diag.shape))
        )

        return loss