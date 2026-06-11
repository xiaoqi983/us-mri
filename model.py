import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EDM Preconditioning (Karras et al., 2022)
# ---------------------------------------------------------------------------

class EDMPreconditioning(nn.Module):
    """EDM preconditioning wrappers and σ utilities.

    Key hyper-parameters (Table 1 of the paper):
        σ_data = 0.5   (assumed data std, our data is in [-1,1])
        σ_min  = 0.002
        σ_max  = 80.0
    """

    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    # ---- preconditioning coefficients (scalar, per-sample) ----

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.log(sigma + 1e-20)

    # ---- forward diffusion: add noise ----

    def q_sample(self, x: torch.Tensor, sigma: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x)
        return x + sigma.view(-1, *([1] * (x.ndim - 1))) * noise

    # ---- denoise: network output → x0 prediction ----

    def denoise(self, F_theta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Apply preconditioning to recover x0 from raw network output F_theta."""
        c_skip = self.c_skip(sigma).view(-1, *([1] * (F_theta.ndim - 1)))
        c_out = self.c_out(sigma).view(-1, *([1] * (F_theta.ndim - 1)))
        return c_skip * F_theta + c_out * F_theta

    # ---- sample σ for training (log-normal) ----

    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample σ from log-normal distribution (mean=-1.2, std=1.2)."""
        log_sigma = torch.randn(batch_size, device=device) * 1.2 - 1.2
        sigma = log_sigma.exp().clamp(self.sigma_min, self.sigma_max)
        return sigma


# ---------------------------------------------------------------------------
# Building blocks (unchanged from DDPM version)
# ---------------------------------------------------------------------------

def make_group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class SigmaEmbedding(nn.Module):
    """Embed continuous σ for the UNet (replaces SinusoidalTimeEmbedding)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (B,) → (B, dim)
        return self.mlp(sigma.unsqueeze(-1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = make_group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = make_group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = make_group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x).view(b, c, h * w)
        q, k, v = self.qkv(x).chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        attention = torch.einsum("bhdl,bhdm->bhlm", q, k) * (head_dim ** -0.5)
        attention = attention.softmax(dim=-1)
        out = torch.einsum("bhlm,bhdm->bhdl", attention, v).reshape(b, c, h * w)
        out = self.proj(out).view(b, c, h, w)
        return residual + out


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int,
        use_attention: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = []
        current_channels = in_channels
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(current_channels, out_channels, time_dim, dropout=dropout))
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, time_emb)
        return self.attention(x)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_res_blocks: int,
        use_attention: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks = [ResidualBlock(in_channels, out_channels, time_dim, dropout=dropout)]
        for _ in range(num_res_blocks - 1):
            blocks.append(ResidualBlock(out_channels, out_channels, time_dim, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.attention = AttentionBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, time_emb)
        return self.attention(x)


class DenoisingUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: Sequence[int] = (2, 3),
        dropout: float = 0.0,
        enable_source_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.enable_source_fusion = enable_source_fusion
        time_dim = base_channels * 4
        # EDM: use sigma embedding instead of sinusoidal timestep embedding
        self.sigma_mlp = nn.Sequential(
            SigmaEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        first_channels = base_channels * channel_mults[0]
        self.input_proj = nn.Conv2d(in_channels, first_channels, kernel_size=3, padding=1)

        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: List[int] = []

        current_channels = first_channels
        stage_out_channels: List[int] = []
        for level, mult in enumerate(channel_mults):
            out_channels_level = base_channels * mult
            stage = EncoderStage(
                in_channels=current_channels,
                out_channels=out_channels_level,
                time_dim=time_dim,
                num_res_blocks=num_res_blocks,
                use_attention=level in attention_levels,
                dropout=dropout,
            )
            self.encoder_stages.append(stage)
            stage_out_channels.append(out_channels_level)
            self.skip_channels.append(out_channels_level)
            current_channels = out_channels_level
            if level != len(channel_mults) - 1:
                self.downsamples.append(Downsample(current_channels))

        self.mid_block1 = ResidualBlock(current_channels, current_channels, time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(current_channels)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, time_dim, dropout=dropout)

        self.fusion_convs = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_stage_channels = list(reversed(stage_out_channels))
        for level, skip_channels in enumerate(reversed_stage_channels):
            if enable_source_fusion:
                self.fusion_convs.append(
                    nn.Sequential(
                        nn.Conv2d(skip_channels * 2, skip_channels, kernel_size=1),
                        nn.SiLU(),
                    )
                )
            else:
                self.fusion_convs.append(nn.Identity())

            decoder_stage = DecoderStage(
                in_channels=current_channels + skip_channels,
                out_channels=skip_channels,
                time_dim=time_dim,
                num_res_blocks=num_res_blocks,
                use_attention=(len(channel_mults) - 1 - level) in attention_levels,
                dropout=dropout,
            )
            self.decoder_stages.append(decoder_stage)
            current_channels = skip_channels
            if level != len(reversed_stage_channels) - 1:
                self.upsamples.append(Upsample(current_channels))

        self.out_norm = make_group_norm(current_channels)
        self.out_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor, sigma: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        time_emb = self.sigma_mlp(sigma)
        h = self.input_proj(x)
        encoder_features: List[torch.Tensor] = []
        for level, stage in enumerate(self.encoder_stages):
            h = stage(h, time_emb)
            encoder_features.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)
        return h, encoder_features, time_emb

    def decode(
        self,
        h: torch.Tensor,
        time_emb: torch.Tensor,
        encoder_features: List[torch.Tensor],
        source_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        feature_stack = list(encoder_features)
        source_stack = list(source_features) if source_features is not None else None
        for level, stage in enumerate(self.decoder_stages):
            skip = feature_stack.pop()
            if source_stack is not None:
                source_skip = source_stack.pop()
                skip = self.fusion_convs[level](torch.cat([skip, source_skip], dim=1))
            h = torch.cat([h, skip], dim=1)
            h = stage(h, time_emb)
            if level < len(self.upsamples):
                h = self.upsamples[level](h)
        return self.out_conv(F.silu(self.out_norm(h)))

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        source_features: Optional[List[torch.Tensor]] = None,
        return_encoder_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        h, encoder_features, time_emb = self.encode(x, sigma)
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        out = self.decode(h, time_emb, encoder_features, source_features=source_features)
        if return_encoder_features:
            return out, encoder_features
        return out, None


# ---------------------------------------------------------------------------
# EDM-US: self-supervised denoising on ultrasound
# ---------------------------------------------------------------------------

class EDMUS(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ) -> None:
        super().__init__()
        self.edm = EDMPreconditioning(sigma_data=sigma_data, sigma_min=sigma_min, sigma_max=sigma_max)
        self.unet = DenoisingUNet(
            in_channels=3,
            out_channels=3,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            enable_source_fusion=False,
        )

    def encode_condition(self, us: torch.Tensor) -> List[torch.Tensor]:
        sigma = torch.zeros(us.size(0), device=us.device)
        _, encoder_features, _ = self.unet.encode(us, sigma)
        return encoder_features

    def forward_train(
        self, us: torch.Tensor, sigma: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if sigma is None:
            sigma = self.edm.sample_sigma(us.size(0), us.device)
        if noise is None:
            noise = torch.randn_like(us)

        # Forward diffusion
        x_sigma = self.edm.q_sample(us, sigma, noise)

        # Precondition input
        c_in = self.edm.c_in(sigma).view(-1, *([1] * (us.ndim - 1)))
        c_noise = self.edm.c_noise(sigma)
        # Network takes c_in * x_sigma and c_noise as sigma
        F_theta, encoder_features = self.unet(c_in * x_sigma, c_noise, return_encoder_features=True)

        # Precondition output → x0 prediction
        c_skip = self.edm.c_skip(sigma).view(-1, *([1] * (us.ndim - 1)))
        c_out = self.edm.c_out(sigma).view(-1, *([1] * (us.ndim - 1)))
        x0_hat = (c_skip * x_sigma + c_out * F_theta).clamp(-1.0, 1.0)

        # Target: the raw network should predict the noise (Denoiser score matching)
        # EDM target: F_theta should approximate c_skip*x + c_out*noise → we use
        # the standard EDM loss: ||F_theta - target||^2 where target = (x - c_skip*x_sigma)/c_out
        # Simplified: the loss is on the raw network output vs. the "score" target
        # In practice, EDM uses: loss = (F_theta - target)^2 weighted by (sigma^2 + sigma_data^2) / (sigma^2 * sigma_data^2)
        # We follow Karras: target = (noise) and weight by (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        # But for simplicity and consistency with DDIC, we use the denoised x0 for correlation loss
        # and MSE on raw network output vs. noise target
        target = noise

        return {
            "x_sigma": x_sigma,
            "sigma": sigma,
            "noise": noise,
            "pred_noise": F_theta,
            "target": target,
            "x0_hat": x0_hat,
            "encoder_features": encoder_features,
        }


# ---------------------------------------------------------------------------
# EDM-MR: conditional generation with US + preop MR
# ---------------------------------------------------------------------------

class EDMMR(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ) -> None:
        super().__init__()
        self.edm = EDMPreconditioning(sigma_data=sigma_data, sigma_min=sigma_min, sigma_max=sigma_max)
        self.unet = DenoisingUNet(
            in_channels=5,
            out_channels=1,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            enable_source_fusion=True,
        )

    def _prepare_input(
        self,
        x_sigma: torch.Tensor,
        sigma: torch.Tensor,
        us_condition: torch.Tensor,
        preop_mr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if preop_mr is None:
            preop_mr = torch.zeros(x_sigma.size(0), 1, x_sigma.size(2), x_sigma.size(3), device=x_sigma.device)
        c_in = self.edm.c_in(sigma).view(-1, *([1] * (x_sigma.ndim - 1)))
        c_noise = self.edm.c_noise(sigma)
        model_input = torch.cat([c_in * x_sigma, us_condition, preop_mr], dim=1)
        return model_input, c_noise

    def forward_train(
        self,
        mr: torch.Tensor,
        us_condition: torch.Tensor,
        source_features: List[torch.Tensor],
        sigma: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        preop_mr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if sigma is None:
            sigma = self.edm.sample_sigma(mr.size(0), mr.device)
        if noise is None:
            noise = torch.randn_like(mr)

        x_sigma = self.edm.q_sample(mr, sigma, noise)
        model_input, c_noise = self._prepare_input(x_sigma, sigma, us_condition, preop_mr)
        F_theta, _ = self.unet(model_input, c_noise, source_features=source_features)

        c_skip = self.edm.c_skip(sigma).view(-1, *([1] * (mr.ndim - 1)))
        c_out = self.edm.c_out(sigma).view(-1, *([1] * (mr.ndim - 1)))
        x0_hat = (c_skip * x_sigma + c_out * F_theta).clamp(-1.0, 1.0)

        target = noise

        return {
            "x_sigma": x_sigma,
            "sigma": sigma,
            "noise": noise,
            "pred_noise": F_theta,
            "target": target,
            "x0_hat": x0_hat,
        }

    @torch.no_grad()
    def denoise_step(
        self,
        x_sigma: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        us_condition: torch.Tensor,
        source_features: List[torch.Tensor],
        preop_mr: Optional[torch.Tensor] = None,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
    ) -> torch.Tensor:
        """Heun's 2nd-order deterministic/stochastic sampler step (Karras et al.)."""
        # Optional stochastic noise injection
        if s_churn > 0 and sigma.min().item() > s_tmin and sigma.max().item() < s_tmax:
            gamma = min(s_churn / len(sigma), 1.0)
            noise_inject = torch.randn_like(x_sigma) * s_noise
            sigma_hat = sigma + gamma * sigma
            x_hat = x_sigma + torch.sqrt(sigma_hat ** 2 - sigma ** 2).view(-1, *([1] * (x_sigma.ndim - 1))) * noise_inject
        else:
            sigma_hat = sigma
            x_hat = x_sigma

        # Denoise to get x0 estimate
        model_input, c_noise = self._prepare_input(x_hat, sigma_hat, us_condition, preop_mr)
        F_theta, _ = self.unet(model_input, c_noise, source_features=source_features)

        c_skip = self.edm.c_skip(sigma_hat).view(-1, *([1] * (x_sigma.ndim - 1)))
        c_out = self.edm.c_out(sigma_hat).view(-1, *([1] * (x_sigma.ndim - 1)))
        x0_hat = c_skip * x_hat + c_out * F_theta

        # Score / derivative
        d_cur = (x_hat - x0_hat) / sigma_hat.view(-1, *([1] * (x_sigma.ndim - 1)))

        # Euler step
        x_next = x_hat + (sigma_next - sigma_hat).view(-1, *([1] * (x_sigma.ndim - 1))) * d_cur

        # 2nd-order correction (Heun)
        if sigma_next.min().item() > 0:
            model_input_next, c_noise_next = self._prepare_input(x_next, sigma_next, us_condition, preop_mr)
            F_theta_next, _ = self.unet(model_input_next, c_noise_next, source_features=source_features)

            c_skip_next = self.edm.c_skip(sigma_next).view(-1, *([1] * (x_sigma.ndim - 1)))
            c_out_next = self.edm.c_out(sigma_next).view(-1, *([1] * (x_sigma.ndim - 1)))
            x0_hat_next = c_skip_next * x_next + c_out_next * F_theta_next

            d_next = (x_next - x0_hat_next) / sigma_next.view(-1, *([1] * (x_sigma.ndim - 1)))
            x_next = x_hat + (sigma_next - sigma_hat).view(-1, *([1] * (x_sigma.ndim - 1))) * (d_cur + d_next) / 2.0

        return x_next.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Dual EDM + Correlation Model
# ---------------------------------------------------------------------------

class DualEDMCorrelationModel(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.us_edm = EDMUS(
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        self.mr_edm = EDMMR(
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.us_edm.edm.sample_sigma(batch_size, device)

    def forward_train(
        self,
        us: torch.Tensor,
        mr: torch.Tensor,
        sigma_us: Optional[torch.Tensor] = None,
        sigma_mr: Optional[torch.Tensor] = None,
        preop_mr: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        batch_size = us.size(0)
        device = us.device
        if sigma_us is None:
            sigma_us = self.sample_sigma(batch_size, device)
        if sigma_mr is None:
            sigma_mr = self.sample_sigma(batch_size, device)

        us_outputs = self.us_edm.forward_train(us, sigma=sigma_us)
        source_features = self.us_edm.encode_condition(us)
        mr_outputs = self.mr_edm.forward_train(mr, us, source_features, sigma=sigma_mr, preop_mr=preop_mr)
        return {"us": us_outputs, "mr": mr_outputs, "sigmas": {"us": sigma_us, "mr": sigma_mr}}

    @torch.no_grad()
    def sample(
        self,
        us: torch.Tensor,
        num_steps: int = 18,
        s_churn: float = 40.0,
        s_tmin: float = 0.05,
        s_tmax: float = 50.0,
        s_noise: float = 1.003,
        preop_mr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stochastic sampler with Heun's 2nd-order correction (Algorithm 2 of Karras et al.)."""
        batch_size = us.size(0)
        device = us.device
        source_features = self.us_edm.encode_condition(us)

        # Build σ schedule (geometric progression)
        sigmas = torch.exp(
            torch.linspace(math.log(self.sigma_max), math.log(self.sigma_min), num_steps + 1, device=device)
        )

        # Start from pure noise
        x = torch.randn(batch_size, 1, us.size(2), us.size(3), device=device) * self.sigma_max

        for i in range(num_steps):
            sigma_cur = sigmas[i].expand(batch_size)
            sigma_next = sigmas[i + 1].expand(batch_size)
            x = self.mr_edm.denoise_step(
                x, sigma_cur, sigma_next, us, source_features,
                preop_mr=preop_mr,
                s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise,
            )

        return x.clamp(-1.0, 1.0)
