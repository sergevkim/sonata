from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from torch import Tensor


class BaseLogger(ABC):
    def __init__(
            self,
            api_token: str,
            project_name: str,
            experiment_name: Optional[str] = None,
            experiment_description: Optional[str] = None,
            params: Optional[Dict[str, Any]] = None,
        ):
        super().__init__()

    def log_image(
            self,
            image_name: str,
            image,
            step: Optional[int] = None,
        ) -> None:
        pass

    @abstractmethod
    def log_metric(
            self,
            metric_name: str,
            metric_value: Union[Tensor, float],
            step: Optional[int] = None,
        ) -> None:
        pass

    def log_metrics(
            self,
            metrics: Dict[str, Union[Tensor, float]],
            step: Optional[int] = None,
        ) -> None:
        for metric_name, metric_value in metrics.items():
            self.log_metric(
                metric_name=metric_name,
                metric_value=metric_value,
                step=step,
            )

