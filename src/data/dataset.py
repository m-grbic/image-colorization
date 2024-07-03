from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import os

from .image_utils import SoftEncoder
from .image_loader import load_train_data, load_eval_data, load_visual_data, load_regression_train_data


IMAEGNET_MEAN = [0.485, 0.456, 0.406]
IMAEGNET_STD = [0.229, 0.224, 0.225]
RGB2LUMINANCE_COEFF = [0.2126, 0.7152, 0.0722]

LUMINANCE_MEAN = sum([coeff * mean for coeff, mean in zip(RGB2LUMINANCE_COEFF, IMAEGNET_MEAN)])
LUMINANCE_STD = np.sqrt(sum([coeff * (std ** 2) for coeff, std in zip(RGB2LUMINANCE_COEFF, IMAEGNET_STD)]))


class BaseDataset(Dataset):
    """Common dataset functionalities."""

    _dataset_mapper = {
        'train': 'data/ILSVRC/Data/CLS-LOC/train',
        'val': 'data/ILSVRC/Data/CLS-LOC/val',
        'test': 'data/ILSVRC/Data/CLS-LOC/test'
    }

    def __init__(self, dataframe: pd.DataFrame, sigma: float = 0.5):
        """Constructor"""
        self._dataframe = dataframe
        self._soft_encoder = SoftEncoder(sigma=sigma)
        self._transforms = transforms.Compose([
            transforms.Normalize(mean=LUMINANCE_MEAN, std=LUMINANCE_STD),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    def _create_image_path(self, idx: int) -> str:
        return os.path.join(
            self._dataset_mapper[self._dataframe.at[idx, 'dataset']],
            self._dataframe.at[idx, 'image_path']
        )

    def __len__(self):
        return len(self._dataframe)


class TrainDataset(BaseDataset):

    def __getitem__(self, idx):
        image_path = self._create_image_path(idx)       
        x, y = load_train_data(image_path, self._soft_encoder)
        x = self._transforms(x)
        return x, y


class TrainRegressionDataset(BaseDataset):

    def __getitem__(self, idx):
        image_path = self._create_image_path(idx)       
        x, y = load_regression_train_data(image_path)
        x = self._transforms(x)
        return x, y


class EvalDataset(BaseDataset):

    def __getitem__(self, idx):
        image_path = self._create_image_path(idx)   
        x, y = load_eval_data(image_path)
        x = self._transforms(x)
        return x, y


class VisualDataset(BaseDataset):

    def __getitem__(self, idx):
        image_path = self._create_image_path(idx)   
        x, l, image_rgb = load_visual_data(image_path)
        x = self._transforms(x)
        return x, l, image_rgb
