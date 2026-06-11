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
    def __init__(self, lambda_corr: float = 0.1) -> None:
        super().__init__()
        self.lambda_corr = lambda_corr
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

    def forward(
        self,
        us_pred_noise: torch.Tensor,
        us_true_noise: torch.Tensor,
        mr_pred_noise: torch.Tensor,
        mr_true_noise: torch.Tensor,
        x0_hat_mr: torch.Tensor,
        us_condition: torch.Tensor,
        overlap_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        us_denoise_loss = F.mse_loss(us_pred_noise, us_true_noise)
        mr_denoise_loss = F.mse_loss(mr_pred_noise, mr_true_noise)
        corr_loss = self.correlation_loss(x0_hat_mr, us_condition, overlap_mask)
        total_loss = us_denoise_loss + mr_denoise_loss + self.lambda_corr * corr_loss
        return {
            "us_denoise_loss": us_denoise_loss,
            "mr_denoise_loss": mr_denoise_loss,
            "corr_loss": corr_loss,
            "total_loss": total_loss,
        }
