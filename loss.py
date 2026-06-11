from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrelationFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.flatten(start_dim=1)
    y = y.flatten(start_dim=1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = torch.sum(x * y, dim=1)
    denominator = torch.sqrt(torch.sum(x * x, dim=1) * torch.sum(y * y, dim=1) + eps)
    return numerator / (denominator + eps)


class DDICLoss(nn.Module):
    """Loss for Dual EDM + Correlation training.

    EDM denoise loss uses the Karras weighting:
        L_denoise = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2 * ||F_theta - target||^2
    where target = noise (the added noise), and F_theta is the raw network output.

    For backward compatibility, the interface accepts pred_noise and true_noise
    which in the EDM context are F_theta and the noise target respectively.
    """

    def __init__(self, lambda_corr: float = 0.1, sigma_data: float = 0.5) -> None:
        super().__init__()
        self.lambda_corr = lambda_corr
        self.sigma_data = sigma_data
        self.mr_corr_encoder = CorrelationFeatureExtractor(in_channels=1)
        self.us_corr_encoder = CorrelationFeatureExtractor(in_channels=3)

    def correlation_loss(
        self,
        x0_hat_mr: torch.Tensor,
        us_condition: torch.Tensor,
        overlap_mask: torch.Tensor,
    ) -> torch.Tensor:
        mr_features = self.mr_corr_encoder(x0_hat_mr)
        us_features = self.us_corr_encoder(us_condition)
        feature_mask = F.interpolate(overlap_mask, size=mr_features.shape[-2:], mode="nearest")
        mr_features = mr_features * feature_mask
        us_features = us_features * feature_mask
        rho = pearson_corrcoef(mr_features, us_features)
        return 1.0 - rho.mean()

    def _edm_loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """EDM loss weighting: (σ² + σ_data²) / (σ · σ_data)²"""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def forward(
        self,
        us_pred_noise: torch.Tensor,
        us_true_noise: torch.Tensor,
        mr_pred_noise: torch.Tensor,
        mr_true_noise: torch.Tensor,
        x0_hat_mr: torch.Tensor,
        us_condition: torch.Tensor,
        overlap_mask: torch.Tensor,
        us_sigma: torch.Tensor = None,
        mr_sigma: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # EDM-weighted denoise loss
        if us_sigma is not None:
            us_weight = self._edm_loss_weight(us_sigma).view(-1, *([1] * (us_pred_noise.ndim - 1)))
            us_denoise_loss = (us_weight * (us_pred_noise - us_true_noise) ** 2).mean()
        else:
            us_denoise_loss = F.mse_loss(us_pred_noise, us_true_noise)

        if mr_sigma is not None:
            mr_weight = self._edm_loss_weight(mr_sigma).view(-1, *([1] * (mr_pred_noise.ndim - 1)))
            mr_denoise_loss = (mr_weight * (mr_pred_noise - mr_true_noise) ** 2).mean()
        else:
            mr_denoise_loss = F.mse_loss(mr_pred_noise, mr_true_noise)

        corr_loss = self.correlation_loss(x0_hat_mr, us_condition, overlap_mask)
        total_loss = us_denoise_loss + mr_denoise_loss + self.lambda_corr * corr_loss
        return {
            "us_denoise_loss": us_denoise_loss,
            "mr_denoise_loss": mr_denoise_loss,
            "corr_loss": corr_loss,
            "total_loss": total_loss,
        }
