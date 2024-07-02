import pickle
import torch
from torch import nn

Z_SHAPE = 64
TOL = 1e-8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_ab_pair_counts(path: str = "data/ILSVRC/Metadata/ab_pair_counts.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_class_weights_mapper(l: float = 0.5):
    pairs_counter = load_ab_pair_counts()

    total_pairs = len(pairs_counter)
    total_count = sum(pairs_counter.values())

    normalized_distribution = {pair: count/total_count for pair, count in pairs_counter.items()}
    uniform_distribution = {pair: 1/total_pairs for pair in pairs_counter}

    weights = {
        pair: 1 / ((1 - l) * normalized_distribution[pair] + l * uniform_distribution[pair])
        for pair in uniform_distribution
    }
    total_weights = sum(weights.values())
    return {pair: weights[pair] / total_weights for pair in weights}


def get_class_weights_tensor(l: float = 0.5):
    class_weights_mapper = get_class_weights_mapper(l=l)
    weights = list(class_weights_mapper.values())
    weights_tensor = torch.tensor(weights)
    return weights_tensor.to(device)


class MultinomialCrossEntropyLoss(nn.Module):
    
    def __init__(self, batch_size: int, l: float = 0.5, use_weights: bool = True):
        super().__init__()
        self._batch_size = batch_size
        self._weights = get_class_weights_tensor(l=l)
        self._weights = self._weights.view(1, -1, 1, 1).expand(batch_size, -1, Z_SHAPE, Z_SHAPE)  # (BS, 256, 64, 64)
        self._use_weights = use_weights

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(target * torch.log(prediction + TOL), dim=1, keepdim=True)  # (BS, 1, 64, 64)

        if self._use_weights:
            q = torch.argmax(target, dim=1, keepdim=True)  # (BS, 1, 64, 64)
            weights = torch.gather(self._weights, 1, q)  # (BS, 1, 64, 64)
            cross_entropy = cross_entropy * weights  # (BS, 1, 64, 64)

        loss = - torch.mean(cross_entropy) # scalar
        return loss
