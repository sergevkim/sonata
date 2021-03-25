from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CommonArguments:
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    neptune_api_token: str = open('./neptune_api_token.txt').readline()[:-1]
    neptune_experiment_name: str = 'sonata'
    neptune_project_name: str = 'raccooncoder/sonata'
    seed: int = 9
    verbose: int = 1
    version: str = 'sonata0.1'


@dataclass
class DataArguments:
    batch_size: int = 16
    img_data_path: Path = Path('./data/landscape')
    wav_data_path: Path = Path('./data/warblrb10k_public/wav')
    new_size: int = 256
    num_workers: int = 8
    val_ratio: float = 0.1


@dataclass
class TrainArguments:
    max_epoch: int = 10
    one_batch_overfit: int = 0
    learning_rate: float = 3e-4
    save_period: int = 20
    scheduler_gamma: float = 0.5
    scheduler_step_size: int = 10


@dataclass
class SpecificArguments:
    specific: bool = False


print(CommonArguments.device)

