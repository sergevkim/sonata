from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class BaseDataModule(ABC):
    def __init__(
            self,
            data_path: Path,
            batch_size: int,
            num_workers: int,
        ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def prepare_data(
            data_path: Path,
        ):
        pass

    @abstractmethod
    def setup(
            self,
            val_ratio: float,
        ) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        pass

