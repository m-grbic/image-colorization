
import torch
import torch.nn.functional as F
import numpy as np
from .anealed_mean import AnealedMean
from data.image_utils import load_ab_pair_counts
from data.image_loader import convert_lab_to_rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Z_SIZE = 64


class ImageColorizer:

    def __init__(self, model: torch.nn.Module, approach: str, t: float = 0.38) -> None:
        self._model = model.eval().to(device)
        self._t = t
        self._approach = approach
        self._classes = torch.Tensor(list(load_ab_pair_counts()))
        self._a_values = self._classes[:, 0].view(1, -1, 1, 1).expand(1, -1, Z_SIZE, Z_SIZE)
        self._b_values = self._classes[:, 1].view(1, -1, 1, 1).expand(1, -1, Z_SIZE, Z_SIZE)
        self._anealed_mean = AnealedMean(a_dist=self._a_values, b_dist=self._b_values, t=self._t)

    def set_temperature(self, t: float):
        self._t = t

    def __call__(self, x: torch.Tensor, resize: bool = False):
        """Predict ab components in (0, 255) range from input image."""
        is_unsqueezed = False
        if x.dim() == 3:
            is_unsqueezed = True
            x = x.unsqueeze(0)

        with torch.no_grad():
            model_out = self._model(x)

        if self._approach == 'classification':
            ab_predictions = self._anealed_mean(model_out) + 128  # (BS, 2, 64, 64)
        else:
            ab_predictions = model_out * 255

        if resize:
            ab_predictions = F.interpolate(ab_predictions, size=(224, 224), mode='bicubic', align_corners=False)

        ab_predictions = torch.clamp(ab_predictions, min=0, max=255)

        if is_unsqueezed:
            ab_predictions = ab_predictions.squeeze(0)

        return ab_predictions
    
    def reconstruct_image(self, x: torch.Tensor, l: np.ndarray) -> None:
        """Reconstruct the image from lumination component and ab predictions."""
        ab_prediction = self(x, resize=True)

        l_tensor = torch.from_numpy(l).unsqueeze(0)

        image_lab = torch.cat((l_tensor, ab_prediction), dim=0)
        image_lab = image_lab.permute(1, 2, 0).detach().numpy().astype(np.uint8)

        return convert_lab_to_rgb(image_lab)
