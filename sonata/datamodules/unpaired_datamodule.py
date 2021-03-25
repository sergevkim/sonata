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
import torchaudio
from sonata.datamodules import BaseDataModule


class UnpairedDataset(Dataset):
    def __init__(
            self,
            img_data_path: Path,
            wav_data_path: Path,
            img_transform: List,
            wav_size
        ):
        self.wav_filenames = list(str(p) for p in wav_data_path.glob('*.wav'))
        self.img_filenames = list(str(p) for p in img_data_path.glob('*.jpg'))
        self.img_transform = Compose(img_transform)
        self.wav_size = wav_size

        self.dataset_len = min(len(self.img_filenames), len(self.wav_filenames))

    def __len__(self):
        return self.dataset_len

    def __getitem__(
            self,
            idx,
        ):
        wav_filename = self.wav_filenames[idx]
        wav, sr = torchaudio.load(self.wav_filenames[idx])

        img_filename = self.img_filenames[idx]
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.img_transform:
            img = self.img_transform(image=img)['image']

        return img, wav[:, :self.wav_size]


class UnpairedDataModule(BaseDataModule):
    def setup(
            self,
            img_data_path,
            wav_data_path,
            val_ratio: float = 0.1,
            wav_size: int = 8192,
            new_size: int = 256,
        ):
        img_transform = [
            Resize(new_size, new_size),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            HorizontalFlip(),
            ToTensorV2(),
        ]

        full_dataset = UnpairedDataset(
            img_data_path=img_data_path,
            wav_data_path=wav_data_path,
            img_transform=img_transform,
            wav_size=wav_size
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )

