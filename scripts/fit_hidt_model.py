from argparse import ArgumentParser
from pathlib import Path

from sonata.datamodules import LandscapesDataModule
from sonata.loggers import NeptuneLogger
from sonata.models import HiDTModel
from sonata.trainer import Trainer
from sonata.utils.randomer import Randomer

from configs.hidt_config import (
    CommonArguments,
    DataArguments,
    TrainArguments,
    SpecificArguments,
)


def main(args):
    Randomer.set_seed(seed=args.seed)

    model = HiDTModel(
        learning_rate=args.learning_rate,
        verbose=args.verbose,
        device=args.device,
    ).to(args.device)

    datamodule = LandscapesDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(
        new_size=args.new_size,
        val_ratio=args.val_ratio,
    )

    logger = NeptuneLogger(
        api_token=args.neptune_api_token,
        project_name=args.neptune_project_name,
        experiment_name=args.neptune_experiment_name,
    )

    trainer = Trainer(
        logger=logger,
        max_epoch=args.max_epoch,
        one_batch_overfit=args.one_batch_overfit,
        save_period=args.save_period,
        verbose=args.verbose,
        version=args.version,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    default_args_dict = {
        **vars(CommonArguments()),
        **vars(DataArguments()),
        **vars(TrainArguments()),
        **vars(SpecificArguments()),
    }

    for arg, value in default_args_dict.items():
        parser.add_argument(
            f'--{arg}',
            type=type(value),
            default=value,
            help=f'<{arg}>, default: {value}',
        )

    args = parser.parse_args()

    main(args)

