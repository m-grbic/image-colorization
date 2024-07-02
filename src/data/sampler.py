from torch.utils.data.sampler import Sampler
import numpy as np
import pandas as pd
from typing import List


class SubsetRandomSampler(Sampler):
    def __init__(self, dataframe: pd.DataFrame, subset_size: int = 320_000) -> None:
        self._indices = dataframe.index.tolist()
        self._subset_size = subset_size
        self._used = []

    def _sample(self, n: int) -> List[int]:
        """Sample n unsued indices from the indices list."""
        indices_to_sample = [ind for ind in self._indices if ind not in self._used]
        sampled_indices = np.random.choice(indices_to_sample, n, replace=False)
        
        self._used.extend(sampled_indices)

        if len(sampled_indices) < n:
            additionaly_sampled_indices = self._sample(n - len(sampled_indices))
            sampled_indices.extend(additionaly_sampled_indices)
            self._used = additionaly_sampled_indices

        return sampled_indices
        

    def __iter__(self):
        indices = self._sample(n=self._subset_size)

        if self._subset_size == len(self._used):
            self._used = []
        
        return iter(indices)

    def __len__(self):
        return self._subset_size
