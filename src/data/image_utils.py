import pickle
from typing import Union

import math
import numpy as np
import torch
import pickle

IMAGE_SIZE = (224, 224)
Z_SIZE = 64
K_NEAREST_NEIGHBOURS = 5
    

def load_ab_pair_counts(path: str = "data/ILSVRC/Metadata/ab_pair_counts.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_quantized_ab_pairs(image_lab: np.ndarray, grid: int = 10):
    """Get list of ab pairs of image"""
    ab_components = image_lab[:, :, 1:].reshape(-1, 2)
    ab_components = ab_components.astype(int) - 128
    ab_quant = ab_components // grid * grid
    return [tuple(pair) for pair in ab_quant.tolist()]


class GaussianKernel:
    def __init__(self, sigma: float = 5.0) -> None:
        self._scale = 1 / (math.sqrt(2 * math.pi) * sigma)
        self._exp_scale = - 1 / (2 * sigma ** 2)

    def __call__(self, x: torch.tensor):
        return self._scale * torch.exp(self._exp_scale * x)


class SoftEncoder:

    def __init__(self, sigma: int = 5) -> None:
        """Constructor."""
        quantized_pairs = list(load_ab_pair_counts())
        self._num_classes = len(quantized_pairs)
        self._quantized_pairs = torch.Tensor(quantized_pairs).T.view(1, 1, 2, -1).expand(Z_SIZE, Z_SIZE, 2, -1)  # (64, 64, 2, 256)
        self._gaussian_kernel = GaussianKernel(sigma)

    def __call__(self, ab: np.ndarray) -> torch.Tensor:
        """Apply soft encoding on ab components."""
        # Convert ab to a torch tensor and add the required dimension
        ab_tensor = torch.from_numpy(ab).unsqueeze(-1)  # Shape: (64, 64, 2, 1)

        squared_diff = (self._quantized_pairs - ab_tensor) ** 2  # Shape: (64, 64, 2, 256)
        squared_dist = squared_diff.sum(dim=2)  # Shape: (64, 64, 256)

        # Use torch.topk to find the k nearest neighbours, which is more efficient than argsort
        topk_values, neighbours = torch.topk(squared_dist, K_NEAREST_NEIGHBOURS, dim=-1, largest=False)  # Shapes: (64, 64, 5), (64, 64, 5)

        # Apply the Gaussian kernel function to the gathered values
        processed_values = self._gaussian_kernel(topk_values)  # Shape: (64, 64, 5)

        # Create the result tensor and scatter the processed values into it
        result_tensor = torch.zeros_like(squared_dist)  # Shape: (64, 64, 256)
        result_tensor.scatter_(2, neighbours, processed_values)  # Shape: (64, 64, 256)

        # Normalize the result tensor along the last dimension
        result_tensor /= result_tensor.sum(dim=-1, keepdim=True)  # Shape: (64, 64, 256)

        # Permute the result tensor to get the desired output shape
        return result_tensor.permute(2, 0, 1)  # Shape: (256, 64, 64)
    
    def get_gt_class(self, ab: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(ab, np.ndarray):
            ab = torch.from_numpy(ab).unsqueeze(-1)  # 64, 64, 2, 1
        else:
            ab = ab.permute(1, 2, 0).unsqueeze(0)
        ab_diff = torch.sum((self._quantized_pairs - ab) ** 2, dim=2) # (64, 64, 256)
        return torch.argmin(ab_diff, dim=-1)  # (64, 64)

    def get_one_hot_encoded(self, ab: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        gt_class = self.get_gt_class(ab)  # (64, 64)
        one_hot_encoded_tensor = torch.nn.functional.one_hot(gt_class, num_classes=self._num_classes)  # (64, 64, 256)
        return one_hot_encoded_tensor.permute(2, 0, 1)  # Shape is (256, 64, 64)
