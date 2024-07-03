import torch
from typing import List
import numpy as np
import pandas as pd

from data.image_utils import SoftEncoder, load_ab_pair_counts


class Accuracy:

    def __init__(self) -> None:
        """Constructor."""
        self._soft_encoder = SoftEncoder()
    
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        predicted_classes = self._soft_encoder.get_classes(preds)
        gt_classes = self._soft_encoder.get_classes(targets)
        return torch.sum(predicted_classes == gt_classes) / predicted_classes.numel()
    

class WeightedAccuracy:

    def __init__(self) -> None:
        """Constructor."""
        self._soft_encoder = SoftEncoder()
        self._weights = torch.Tensor([1 / count for count in load_ab_pair_counts().values()])
    
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        predicted_classes = self._soft_encoder.get_classes(preds)
        gt_classes = self._soft_encoder.get_classes(targets)

        gt_classes_ = gt_classes.view(-1, 1)  # N x 1
        weights = self._weights.view(1, -1).expand(gt_classes_.shape[0], -1)  # N x 265
        weights = torch.gather(weights, 1, gt_classes_).flatten()  # N x 1 -> N

        return torch.sum(weights * (predicted_classes == gt_classes)) / predicted_classes.numel()


class EuclidianDistanceError:

    def __init__(self) -> None:
        """Constructor."""
        self._soft_encoder = SoftEncoder()
    
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        return torch.mean(torch.sqrt(torch.sum((preds - targets) ** 2, dim=1)))


class MetricClaculator:

    def __init__(self, metrics: List[str]) -> None:
        """Constructor."""
        self._metrics_func_mapper = self._initialize_metrics(metrics=metrics)
        self._metrics_data = {metric_name: [] for metric_name in metrics}

        self._soft_encoder = SoftEncoder()
        self._weights = torch.Tensor([1 / count for count in load_ab_pair_counts().values()])

    def _initialize_metrics(self, metrics: List[str]) -> dict:
        metrics_func_mapper = {}
        for metric in metrics:
            if metric == 'accuracy':
                metrics_func_mapper['accuracy'] = Accuracy()
            elif metric == 'weighted_accuracy':
                metrics_func_mapper['weighted_accuracy'] = WeightedAccuracy()
            elif metric == 'error':
                metrics_func_mapper['error'] = EuclidianDistanceError()
            else:
                print(f"[ERROR] Metric {metric} is not recognized")
        return metrics_func_mapper
    
    def reset(self) -> None:
        self._metrics_data = {metric_name: [] for metric_name in self._metrics_data}

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        for metric_name, metric_func in self._metrics_func_mapper.items():
            self._metrics_data[metric_name].append(metric_func(preds, targets))

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict({
            metric_name: [np.mean(metric_values)] for metric_name, metric_values in self._metrics_data.items()
        })
