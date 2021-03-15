import neptune
from torch import Tensor

from sonata.loggers import BaseLogger


class NeptuneLogger(BaseLogger):
    def __init__(
            self,
            api_token,
            project_name,
            experiment_description=None,
            experiment_name=None,
            params=None,
        ):
        self.project = neptune.init(
            project_qualified_name=project_name,
            api_token=api_token,
        )
        self.experiment = neptune.create_experiment(
            name=experiment_name,
            params=params,
        )

    def log_metric(
            self,
            metric_name,
            metric_value,
            step=None,
        ):
        if step is None:
            neptune.log_metric(
                log_name=metric_name,
                x=metric_value,
            )
        else:
            neptune.log_metric(
                log_name=metric_name,
                x=step,
                y=metric_value,
            )

if __name__ == '__main__':
    logger = NeptuneLogger(
        api_token='9',
        project_name='9',
    )
    print(logger)

