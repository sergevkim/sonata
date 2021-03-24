from pathlib import Path
from typing import List

import cv2
import torch
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    Resize,
)
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from sonata.datamodules import BaseDataModule


class LandscapesDataset(Dataset):
    def __init__(
            self,
            data_path: Path,
            transform: List,
        ):
        self.filenames = list(str(p) for p in data_path.glob('*.jpg'))
        self.transform = Compose(transform)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(
            self,
            idx,
        ):
        filename = self.filenames[idx]
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image


class LandscapesDataModule(BaseDataModule):
    def setup(
            self,
            val_ratio: float = 0.1,
            new_size: int = 128,
        ):
        data_transform = [
            Resize(new_size, new_size),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            HorizontalFlip(),
            ToTensorV2(),
        ]
        full_dataset = LandscapesDataset(
            data_path=self.data_path,
            transform=data_transform,
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )

