import torch


class AnealedMean:

    tolerance = 1e-1

    def __init__(self, a_dist: torch.Tensor, b_dist: torch.Tensor, t: float = 0.38) -> None:
        self._T = t if t != 0 else t + self.tolerance
        self._index_matrix = torch.arange(265).view(1, -1, 1, 1).expand(1, -1, 64, 64)
        self._a_dist = a_dist
        self._b_dist = b_dist

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        z_exp = torch.exp(torch.log(z) / self._T)  # (BS, 256, 64, 64)
        z_exp_sum = torch.sum(z_exp, dim=1, keepdim=True)  # (BS, 1, 64, 64)
        f_t = z_exp / z_exp_sum  # (BS, 256, 64, 64)

        a_pred = torch.sum(f_t * self._a_dist, dim=1, keepdim=True)  # (BS, 1, 64, 64)
        b_pred = torch.sum(f_t * self._b_dist, dim=1, keepdim=True)  # (BS, 1, 64, 64)

        ab_pred = torch.cat((a_pred, b_pred), dim=1)  # (BS, 2, 64, 64)

        return ab_pred
