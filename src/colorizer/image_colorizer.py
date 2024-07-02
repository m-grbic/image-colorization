
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

    def __call__(self, x: torch.Tensor):
        is_unsqueezed = False

        if x.dim() == 3:
            is_unsqueezed = True
            x = x.unsqueeze(0)

        if self._approach == 'classification':
            pred_prob = self._model(x)  # (BS, 265, 64, 64)
            ab_predictions = self._anealed_mean(pred_prob)  # (BS, 2, 64, 64)
            ab_predictions = F.interpolate(ab_predictions, size=(224, 224), mode='bicubic', align_corners=False)
            ab_predictions += 128
        else:
            ab_predictions = self._model(x)
            ab_predictions = F.interpolate(ab_predictions, size=(224, 224), mode='bicubic', align_corners=False)
            ab_predictions *= 255
        
        ab_predictions = torch.clamp(ab_predictions * 255, min=0, max=255)
        
        if is_unsqueezed:
            ab_predictions = ab_predictions.squeeze(0)

        return ab_predictions
    
    def reconstruct_image(self, l: np.ndarray, ab: torch.Tensor) -> None:
        """Reconstruct the image from lumination component and ab predictions."""
        l_tensor = torch.from_numpy(l).unsqueeze(0)

        image_lab = torch.cat((l_tensor, ab), dim=0)
        image_lab = image_lab.permute(1, 2, 0).detach().numpy().astype(np.uint8)

        return convert_lab_to_rgb(image_lab)
