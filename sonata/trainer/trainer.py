from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.utils as utils
import tqdm
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from sonata.datamodules import BaseDataModule
from sonata.loggers import BaseLogger
from sonata.models import BaseModule


class Trainer:
    def __init__(
            self,
            max_epoch: int = 1,
            one_batch_overfit: bool = False,
            save_period: int = 20,
            verbose: bool = False,
            version: str = 'v0',
            logger: Optional[BaseLogger] = None,
        ):
        self.max_epoch = max_epoch
        self.one_batch_overfit = one_batch_overfit
        self.save_period = save_period
        self.verbose = verbose
        self.version = version
        self.logger = logger

    @classmethod
    def save_checkpoint(
            self,
            model: Module,
            optimizers: List[Optimizer],
            epoch_idx: int,
            checkpoint_path: Path,
        ) -> None:
        checkpoint = {
            'model': model,
            'model_state_dict': model.state_dict(),
            'epoch_idx': epoch_idx,
        }

        if len(optimizers) == 1:
            optimizer = optimizers[0]
            checkpoint['optimizer'] = optimizer
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        else:
            for i, optimizer in enumerate(optimizers):
                checkpoint[f'optimizer_{i}'] = optimizer
                checkpoint[f'optimizer_{i}_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def load_checkpoint(
            model: Module,
            optimizers: List[Optimizer],
            checkpoint_path: Path,
        ) -> None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint[f'optimizer_{i}_state_dict'])

    @torch.enable_grad()
    def training_epoch(
            self,
            model: BaseModule,
            train_dataloader: DataLoader,
            optimizers: List[Optimizer],
            epoch_idx: int,
        ) -> None:
        model.train()
        metrics = defaultdict(list)

        for batch_idx, batch in enumerate(tqdm.tqdm(train_dataloader)):
            for optimizer_idx, optimizer in enumerate(optimizers):
                info = model.training_step(
                    batch=batch,
                    batch_idx=batch_idx,
                    optimizer_idx=optimizer_idx,
                )

                for metric_name, value in info.items():
                    if type(value) is Tensor:
                        metrics[metric_name].append(value.item())
                    else:
                        metrics[metric_name].append(value)

                loss = info['loss']
                loss.backward()
                utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=10,
                )
                optimizer.step()
                optimizer.zero_grad()

            model.training_step_end(batch_idx=batch_idx)

            if self.one_batch_overfit:
                break

        for metric_name, values in metrics.items():
            mean_value = sum(values) / len(values)
            if self.logger is not None:
                self.logger.log_metric(
                    metric_name=f'train/{metric_name}',
                    metric_value=mean_value,
                )
            if self.verbose:
                print(f'{metric_name}: {mean_value}')

        model.training_epoch_end(epoch_idx=epoch_idx)

    @torch.no_grad()
    def validation_epoch(
            self,
            model: BaseModule,
            val_dataloader: DataLoader,
            schedulers: List[_LRScheduler],
            epoch_idx: int,
        ) -> None:
        model.eval()
        metrics = defaultdict(list)

        for batch_idx, batch in enumerate(tqdm.tqdm(val_dataloader)):
            info = model.validation_step(
                batch=batch,
                batch_idx=batch_idx,
            )

            for metric_name, value in info.items():
                if type(value) is Tensor:
                    metrics[metric_name].append(value.item())
                else:
                    metrics[metric_name].append(value)

            loss = info['loss']
            model.validation_step_end(batch_idx=batch_idx)

        for metric_name, values in metrics.items():
            mean_value = sum(values) / len(values)
            if self.logger is not None:
                self.logger.log_metric(
                    metric_name=f'val/{metric_name}',
                    metric_value=mean_value,
                )
            if self.verbose:
                print(f'{metric_name}: {mean_value}')

        for scheduler in schedulers:
            scheduler.step()

        model.validation_epoch_end(epoch_idx=epoch_idx)

    def fit(
            self,
            model: BaseModule,
            datamodule: BaseDataModule,
        ) -> None:
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        optimizers, schedulers = model.configure_optimizers()

        self.validation_epoch(
            model=model,
            val_dataloader=val_dataloader,
            schedulers=[],
            epoch_idx=0,
        )
        for epoch_idx in range(1, self.max_epoch + 1):
            print(f'Epoch {epoch_idx}')
            self.training_epoch(
                model=model,
                train_dataloader=train_dataloader,
                optimizers=optimizers,
                epoch_idx=epoch_idx,
            )
            self.validation_epoch(
                model=model,
                val_dataloader=val_dataloader,
                schedulers=schedulers,
                epoch_idx=epoch_idx,
            )
            if epoch_idx % self.save_period == 0:
                checkpoint_path = \
                    Path.cwd() / 'models' / f'{self.version}-e{epoch_idx}.pt'
                self.save_checkpoint(
                    model=model,
                    optimizers=optimizers,
                    epoch_idx=epoch_idx,
                    checkpoint_path=checkpoint_path,
                )

    @torch.no_grad()
    def predict(
            self,
            model: BaseModule,
            datamodule: BaseDataModule,
        ) -> List[Tensor]:
        test_dataloader = datamodule.test_dataloader()

        predicts = list()

        for batch_idx, batch in enumerate(test_dataloader):
            predict = model.test_step(
                batch=batch,
                batch_idx=batch_idx,
            )
            predicts.append(predict)
            model.test_step_end(batch_idx=batch_idx)

        model.test_epoch_end(epoch_idx=epoch_idx)

        return predicts


if __name__ == '__main__':
    trainer = Trainer(
        logger=None,
        max_epoch=1,
        verbose=True,
        version='0',
    )
    print('Done!')

