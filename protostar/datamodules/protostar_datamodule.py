from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from protostar.datamodules import BaseDataModule


class ProtostarDataset(Dataset):
    def __init__(
            self,
        ):
        pass

    def __len__(self):
        pass

    def __getitem__(
            self,
            idx,
        ):
        pass


class ProtostarDataModule(BaseDataModule):
    def setup(
            self,
            val_ratio,
        ):
        data = self.prepare_data(
            data_path=self.data_path,
        )
        full_dataset = ProtostarDataset(
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )

