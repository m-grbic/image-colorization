from torch.utils.data import DataLoader
from typing import Optional

from .dataset import BaseDataset
from .sampler import SubsetRandomSampler


def create_dataloader(
        dataset: BaseDataset,
        sampler: Optional[SubsetRandomSampler] = None,
        batch_size: int = 64,
        shuffle: bool = False,
        num_workers: int = 10
    ):
    return DataLoader(
        dataset=dataset, 
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )
