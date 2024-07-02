import torch
from ..data.image_utils import load_ab_pair_counts

# Example tensor with dimensions [64, 1, 64, 64]
predictions = torch.randn(64, 1, 64, 64)  # Replace this with your actual tensor
targets = torch.randint(0, 2, (64, 1, 64, 64))  # Replace with your ground truth tensor

# Flatten the tensors
flattened_predictions = predictions.view(-1)
flattened_targets = targets.view(-1)


def compute_map(preds: torch.Tensor, targets: torch.Tensor):
    sorted_indices = torch.argsort(preds, descending=True)
    sorted_targets = targets[sorted_indices]
    tp = (sorted_targets == 1).float()
    fp = (sorted_targets == 0).float()
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / tp.sum()
    precision = torch.cat([torch.tensor([0.]), precision])
    recall = torch.cat([torch.tensor([0.]), recall])
    map_score = torch.sum((recall[1:] - recall[:-1]) * precision[1:])
    return map_score


def compute_auc(preds: torch.Tensor, targets: torch.Tensor):
    sorted_indices = torch.argsort(preds, descending=True)
    sorted_targets = targets[sorted_indices]
    tp = (sorted_targets == 1).float()
    fp = (sorted_targets == 0).float()
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    tpr = tp_cumsum / tp.sum()
    fpr = fp_cumsum / fp.sum()
    tpr = torch.cat([torch.tensor([0.]), tpr])
    fpr = torch.cat([torch.tensor([0.]), fpr])
    auc_score = torch.trapz(tpr, fpr)
    return auc_score


def compute_top1_accuracy(preds: torch.Tensor, targets: torch.Tensor):
    probas = torch.sigmoid(preds)
    predicted_class = (probas > 0.5).long()
    correct = (predicted_class == targets).sum().float()
    accuracy = correct / targets.numel()
    return accuracy


class MetricClaculator:

    def __init__(self) -> None:
        """Constructor."""
        self._class_weights = list(load_ab_pair_counts().values())

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Any:
        pass
