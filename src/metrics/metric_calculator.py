import torch
from typing import List
import numpy as np
import pandas as pd

from data.image_utils import SoftEncoder, load_ab_pair_counts


class RawAccuracy:

    def __init__(self, threshold: int = 10) -> None:
        """Constructor."""
        self._soft_encoder = SoftEncoder()
        self._threshold = threshold
    
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        """Calculate class accuracy within threshold."""
        predicted_classes = self._soft_encoder.get_classes(preds)
        target_classes = self._soft_encoder.get_classes(targets)
        return (predicted_classes == target_classes).sum() / predicted_classes.numel()
        

class Accuracy:

    def __init__(self, threshold: int = 10) -> None:
        """Constructor."""
        self._soft_encoder = SoftEncoder()
        self._threshold = threshold
    
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        """Calculate class accuracy within threshold."""
        dist = torch.linalg.norm(targets - preds, axis=1)

        within_thresh = dist <= self._threshold
        num_within_thresh = torch.count_nonzero(within_thresh)

        if num_within_thresh == 0:
            return 0
        
        return num_within_thresh / within_thresh.numel()
    

class WeightedAccuracy:

    def __init__(self, threshold: int = 10) -> None:
        """Constructor."""
        self._soft_encoder = SoftEncoder()
        self._threshold = threshold
        self._weights = torch.Tensor([1 / count for count in load_ab_pair_counts().values()])
        self._weights /= self._weights.sum()
        self._weights = self._weights.view(1, -1, 1, 1).expand(-1, -1, 64, 64)
    
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        """Calculate weighted class accuracy within threshold."""
        pred_classes = self._soft_encoder.get_classes(targets)
        weights = self._weights.expand(preds.shape[0], -1, -1, -1)
        weights = torch.gather(weights, 1, pred_classes)

        dist = torch.linalg.norm(targets - preds, axis=1)

        within_thresh = dist <= self._threshold
        num_within_thresh = torch.count_nonzero(within_thresh)

        if num_within_thresh == 0:
            return 0
        
        return (within_thresh * weights).sum()


class AUC:
    def __init__(self, threshold_min: int = 0, threshold_max: int = 151, threshold_step: int =10) -> None:
        """Constructor."""
        self._threshold_min = threshold_min
        self._threshold_max = threshold_max
        self._threshold_step = threshold_step

        self._thresholds = np.arange(self._threshold_min, self._threshold_max, self._threshold_step)
        self._accuracy_calculators = [
            Accuracy(threshold=threshold) for threshold in self._thresholds
        ]
    
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        """Calculate weighted class accuracy within threshold."""
        aucs = [accuracy_calc(preds, targets) for accuracy_calc in self._accuracy_calculators]
        return sum(aucs) / len(self._thresholds)
    

class WeightedAUC:
    def __init__(self, threshold_min: int = 0, threshold_max: int = 151, threshold_step: int =10) -> None:
        """Constructor."""
        self._threshold_min = threshold_min
        self._threshold_max = threshold_max
        self._threshold_step = threshold_step

        self._thresholds = np.arange(self._threshold_min, self._threshold_max, self._threshold_step)
        self._accuracy_calculators = [
            WeightedAccuracy(threshold=threshold) for threshold in self._thresholds
        ]
    
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        """Calculate weighted class accuracy within threshold."""
        aucs = [accuracy_calc(preds, targets) for accuracy_calc in self._accuracy_calculators]
        return sum(aucs) / len(self._thresholds)


class EuclidianDistanceError:

    def __init__(self) -> None:
        """Constructor."""
        self._soft_encoder = SoftEncoder()
    
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        return torch.linalg.norm(preds - targets, dim=1).mean()


class MetricClaculator:

    def __init__(self, metrics: List[str]) -> None:
        """Constructor."""
        self._metrics_func_mapper = self._initialize_metrics(metrics=metrics)
        self._metrics_data = {metric_name: [] for metric_name in metrics}

    def _initialize_metrics(self, metrics: List[str]) -> dict:
        metrics_func_mapper = {}
        for metric in metrics:
            if metric == 'raw_accuracy':
                metrics_func_mapper['raw_accuracy'] = RawAccuracy()
            elif metric == 'accuracy':
                metrics_func_mapper['accuracy'] = Accuracy()
            elif metric == 'weighted_accuracy':
                metrics_func_mapper['weighted_accuracy'] = WeightedAccuracy()
            elif metric == 'auc':
                metrics_func_mapper['auc'] = AUC()
            elif metric == 'weighted_auc':
                metrics_func_mapper['weighted_auc'] = WeightedAUC()
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
