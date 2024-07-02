import cv2
import numpy as np
import torch

from typing import Tuple

from .image_utils import SoftEncoder


IMAGE_SIZE = (224, 224)
Z_SIZE = 64


def convert_lab_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)


def convert_rgb_to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def get_ab_components(image_lab: np.ndarray) -> np.ndarray:
    """Get ab components of CIE Lab color space."""
    return image_lab[:, :, 1:].astype(int) - 128


def get_l_component(image_lab: np.ndarray) -> np.ndarray:
    """Get L component of CIE Lab color space."""
    return image_lab[:, :, 0].astype(int)


def get_input_tensor(image_lab: np.ndarray) -> torch.Tensor:
    """Prepares CIE Lab image to model input"""
    input_tensor = torch.from_numpy(image_lab[:, :, 0]).to(torch.float32) / 100
    return input_tensor.unsqueeze(0)


def get_output_train_tensor(image_lab: np.ndarray, soft_encoder: SoftEncoder) -> torch.Tensor:
    image_lab = cv2.resize(image_lab, (Z_SIZE, Z_SIZE), interpolation=cv2.INTER_AREA)
    ab = get_ab_components(image_lab)
    return soft_encoder(ab)


def get_output_eval_tensor(image_lab: np.ndarray, soft_encoder: SoftEncoder) -> torch.Tensor:
    image_lab = cv2.resize(image_lab, (Z_SIZE, Z_SIZE), interpolation=cv2.INTER_AREA)
    ab = get_ab_components(image_lab)
    return soft_encoder.get_one_hot_encoded(ab)


def load_image(image_path: str) -> np.ndarray:
    """Loads RGB image."""
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_for_model(image: np.ndarray) -> np.ndarray:
    """Resizes image to (224, 224) image size"""
    return cv2.resize(image, IMAGE_SIZE)


def convert_image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1)


def load_lab_image(image_path: str) -> np.ndarray:
    """Loads image, resize it and returns image in CIE Lab color space."""
    image_rgb = load_image(image_path)
    image_rgb = resize_for_model(image_rgb)
    return convert_rgb_to_lab(image_rgb)

  
def load_train_data(image_path: str, soft_encoder: SoftEncoder) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares data from training."""
    image_lab = load_lab_image(image_path)

    x = get_input_tensor(image_lab)
    y = get_output_train_tensor(image_lab, soft_encoder)

    return x, y


def load_eval_data(image_path: str, soft_encoder: SoftEncoder) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns image prepared for model input and one hot encoded classes."""
    image_lab = load_lab_image(image_path)

    x = get_input_tensor(image_lab)
    y = get_output_eval_tensor(image_lab, soft_encoder)

    return x, y


def load_visual_data(image_path: str) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    image_rgb = load_image(image_path)
    image_rgb = resize_for_model(image_rgb)
    image_lab = convert_rgb_to_lab(image_rgb)

    x = get_input_tensor(image_lab)
    l = get_l_component(image_lab)

    return x, l, image_rgb
