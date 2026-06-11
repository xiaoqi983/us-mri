import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def extract(buffer: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    values = buffer.gather(0, timesteps)
    return values.view(-1, *([1] * (len(target_shape) - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10_000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
        embeddings = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


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
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
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

    def encode(self, x: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        time_emb = self.time_mlp(timesteps)
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
        timesteps: torch.Tensor,
        source_features: Optional[List[torch.Tensor]] = None,
        return_encoder_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        h, encoder_features, time_emb = self.encode(x, timesteps)
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        out = self.decode(h, time_emb, encoder_features, source_features=source_features)
        if return_encoder_features:
            return out, encoder_features
        return out, None


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000) -> None:
        super().__init__()
        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod), min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_x0_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, timesteps, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance


class DDPMUS(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.diffusion = GaussianDiffusion(timesteps=timesteps)
        self.unet = DenoisingUNet(
            in_channels=3,
            out_channels=3,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            enable_source_fusion=False,
        )

    def encode_condition(self, us: torch.Tensor) -> List[torch.Tensor]:
        timesteps = torch.zeros(us.size(0), device=us.device, dtype=torch.long)
        _, encoder_features, _ = self.unet.encode(us, timesteps)
        return encoder_features

    def forward_train(
        self, us: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(us)
        x_t = self.diffusion.q_sample(us, timesteps, noise)
        pred_noise, encoder_features = self.unet(x_t, timesteps, return_encoder_features=True)
        x0_hat = self.diffusion.predict_x0_from_noise(x_t, timesteps, pred_noise).clamp(-1.0, 1.0)
        return {
            "x_t": x_t,
            "noise": noise,
            "pred_noise": pred_noise,
            "x0_hat": x0_hat,
            "encoder_features": encoder_features,
        }


class DDPMMR(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.diffusion = GaussianDiffusion(timesteps=timesteps)
        self.unet = DenoisingUNet(
            in_channels=4,
            out_channels=1,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            enable_source_fusion=True,
        )

    def predict_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        us_condition: torch.Tensor,
        source_features: List[torch.Tensor],
    ) -> torch.Tensor:
        model_input = torch.cat([x_t, us_condition], dim=1)
        pred_noise, _ = self.unet(model_input, timesteps, source_features=source_features)
        return pred_noise

    def forward_train(
        self,
        mr: torch.Tensor,
        us_condition: torch.Tensor,
        source_features: List[torch.Tensor],
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(mr)
        x_t = self.diffusion.q_sample(mr, timesteps, noise)
        pred_noise = self.predict_noise(x_t, timesteps, us_condition, source_features)
        x0_hat = self.diffusion.predict_x0_from_noise(x_t, timesteps, pred_noise).clamp(-1.0, 1.0)
        return {
            "x_t": x_t,
            "noise": noise,
            "pred_noise": pred_noise,
            "x0_hat": x0_hat,
        }

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        us_condition: torch.Tensor,
        source_features: List[torch.Tensor],
    ) -> torch.Tensor:
        pred_noise = self.predict_noise(x_t, timesteps, us_condition, source_features)
        x0_hat = self.diffusion.predict_x0_from_noise(x_t, timesteps, pred_noise).clamp(-1.0, 1.0)
        posterior_mean, _, posterior_log_variance = self.diffusion.q_posterior(x0_hat, x_t, timesteps)
        nonzero_mask = (timesteps != 0).float().view(-1, 1, 1, 1)
        noise = torch.randn_like(x_t)
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        next_timesteps: torch.Tensor,
        us_condition: torch.Tensor,
        source_features: List[torch.Tensor],
        eta: float = 0.0,
    ) -> torch.Tensor:
        pred_noise = self.predict_noise(x_t, timesteps, us_condition, source_features)
        alpha_t = extract(self.diffusion.alphas_cumprod, timesteps, x_t.shape)
        alpha_next = extract(self.diffusion.alphas_cumprod, next_timesteps.clamp(min=0), x_t.shape)
        x0_hat = self.diffusion.predict_x0_from_noise(x_t, timesteps, pred_noise).clamp(-1.0, 1.0)

        sigma = (
            eta
            * torch.sqrt((1 - alpha_next) / (1 - alpha_t))
            * torch.sqrt(torch.clamp(1 - alpha_t / alpha_next, min=0.0))
        )
        direction = torch.sqrt(torch.clamp(1 - alpha_next - sigma ** 2, min=0.0)) * pred_noise
        noise = torch.randn_like(x_t)
        x_next = torch.sqrt(alpha_next) * x0_hat + direction + sigma * noise
        zero_mask = (next_timesteps < 0).float().view(-1, 1, 1, 1)
        return x_next * (1.0 - zero_mask) + x0_hat * zero_mask


class DualDDPMCorrelationModel(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.us_ddpm = DDPMUS(
            timesteps=timesteps,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
        )
        self.mr_ddpm = DDPMMR(
            timesteps=timesteps,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def forward_train(
        self,
        us: torch.Tensor,
        mr: torch.Tensor,
        t_us: Optional[torch.Tensor] = None,
        t_mr: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        batch_size = us.size(0)
        device = us.device
        if t_us is None:
            t_us = self.sample_timesteps(batch_size, device)
        if t_mr is None:
            t_mr = self.sample_timesteps(batch_size, device)

        us_outputs = self.us_ddpm.forward_train(us, t_us)
        source_features = self.us_ddpm.encode_condition(us)
        mr_outputs = self.mr_ddpm.forward_train(mr, us, source_features, t_mr)
        return {"us": us_outputs, "mr": mr_outputs, "timesteps": {"us": t_us, "mr": t_mr}}

    @torch.no_grad()
    def sample(
        self,
        us: torch.Tensor,
        sampling: str = "ddpm",
        ddim_steps: int = 1000,
        eta: float = 0.0,
    ) -> torch.Tensor:
        batch_size = us.size(0)
        device = us.device
        source_features = self.us_ddpm.encode_condition(us)
        x_t = torch.randn(batch_size, 1, us.size(2), us.size(3), device=device)

        if sampling.lower() == "ddim":
            ddim_steps = min(ddim_steps, self.timesteps)
            schedule = torch.linspace(self.timesteps - 1, 0, ddim_steps, device=device).long()
            next_schedule = torch.cat([schedule[1:], torch.full((1,), -1, device=device, dtype=torch.long)], dim=0)
            for current_t, next_t in zip(schedule, next_schedule):
                t = torch.full((batch_size,), int(current_t.item()), device=device, dtype=torch.long)
                nt = torch.full((batch_size,), int(next_t.item()), device=device, dtype=torch.long)
                x_t = self.mr_ddpm.ddim_step(x_t, t, nt, us, source_features, eta=eta)
            return x_t.clamp(-1.0, 1.0)

        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x_t = self.mr_ddpm.p_sample(x_t, timesteps, us, source_features)
        return x_t.clamp(-1.0, 1.0)
