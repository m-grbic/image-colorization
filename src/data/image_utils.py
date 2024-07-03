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

    def __init__(self, sigma: float = 5.0) -> None:
        """Constructor."""
        quantized_pairs = list(load_ab_pair_counts())
        self._num_classes = len(quantized_pairs)
        self._quantized_pairs = torch.Tensor(quantized_pairs).T.view(1, 1, 2, -1).expand(Z_SIZE, Z_SIZE, 2, -1)  # (64, 64, 2, 256)
        self._a_values = self._quantized_pairs[:, :, :1, :].permute(2, 3, 0, 1)
        self._b_values = self._quantized_pairs[:, :, 1:, :].permute(2, 3, 0, 1)
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
    
    def get_classes(self, ab: torch.Tensor):
        assert ab.min() >= -128 and ab.max() <= 128
        assert ab.dim() == 4 and ab.shape[1:] == torch.Size([2, 64, 64])
        a = ab[:, :1, :, :]  # Shape: (BS, 1, 64, 64)
        b = ab[:, 1:, :, :]  # Shape: (BS, 1, 64, 64)
        dist = (a - self._a_values) ** 2 + (b - self._b_values) ** 2  # Shape: (BS, 265, 64, 64)
        classes = torch.argmin(dist, dim=1)
        assert classes.dim() == 3 and classes.shape[0] == ab.shape[0] and classes.shape[1:] == torch.Size([64, 64])
        return classes.flatten()  # (BS, 64, 64) -> flatten
