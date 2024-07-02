import pandas as pd
from typing import Tuple
import os


def load_metadata(metadata_path: str = 'data/ILSVRC/Metadata') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load DataFrames for train, val and test."""
    return (
        pd.read_pickle(os.path.join(metadata_path, 'train.pkl')),
        pd.read_pickle(os.path.join(metadata_path, 'val.pkl')),
        pd.read_pickle(os.path.join(metadata_path, 'test.pkl'))
    )