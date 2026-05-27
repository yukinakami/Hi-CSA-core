import torch
import torch.nn as nn
import torch.nn.functional as F

from .macro_moe import (
    GenericMacroExpert,
    RegimeVolatilityExpert,
    ScaleExpert,
    SeasonalExpert,
    TrendExpert,
)


EXPERT_NAMES = ["trend", "seasonal", "scale", "regime"]


def _odd_kernel(kernel_size):
    kernel_size = int(kernel_size)
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size


def _moving_average(sequence, kernel_size):
    kernel_size = max(1, _odd_kernel(kernel_size))
    if kernel_size <= 1:
        return sequence
    return F.avg_pool1d(
        sequence.transpose(0, 1).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    ).squeeze(0).transpose(0, 1)


def _multi_average(sequence, kernel_sizes):
    smoothed = [_moving_average(sequence, kernel_size) for kernel_size in kernel_sizes]
    return torch.stack(smoothed, dim=0).mean(dim=0)


def _seasonal_reconstruct(sequence, top_k):
    spectrum = torch.fft.rfft(sequence, dim=0)
    magnitude = spectrum.abs().mean(dim=-1)
    if magnitude.shape[0] <= 1:
        return torch.zeros_like(sequence)

    magnitude = magnitude.clone()
    magnitude[0] = 0.0
    top_k = min(int(top_k), magnitude.shape[0] - 1)
    _, top_indices = torch.topk(magnitude, top_k, dim=0)
    mask = torch.zeros_like(magnitude)
    mask.scatter_(dim=0, index=top_indices, value=1.0)
    return torch.fft.irfft(spectrum * mask[:, None], n=sequence.shape[0], dim=0)


def _normalize_target(target):
    mean = target.mean(dim=0, keepdim=True)
    std = target.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    return (target - mean) / std


def _compress_time(target, seq_len):
    return F.adaptive_avg_pool1d(
        target.transpose(0, 1).unsqueeze(0),
        output_size=seq_len,
    ).squeeze(0).transpose(0, 1)


def build_global_macro_targets(
    train_data,
    seq_len,
    d_model,
    seasonal_top_k=8,
    trend_kernels=(25, 97, 193),
    scale_kernels=(7, 25, 97),
    regime_kernel=25,
    projection_mode="random",
    projection_layer=None,
    projection_seed=2026,
    device=None,
):
    """Build [num_experts, seq_len, d_model] targets from train split only."""
    if not torch.is_tensor(train_data):
        train_data = torch.as_tensor(train_data, dtype=torch.float32)
    train_data = train_data.float()
    device = train_data.device if device is None else torch.device(device)
    x = train_data.to(device)

    trend = _multi_average(x, trend_kernels)
    season = _seasonal_reconstruct(x, seasonal_top_k)

    centered = x - trend
    scale_parts = []
    for kernel_size in scale_kernels:
        local_var = _moving_average(centered.pow(2), kernel_size)
        scale_parts.append(torch.sqrt(local_var.clamp_min(1e-6)))
    scale = torch.stack(scale_parts, dim=0).mean(dim=0)

    residual = x - trend - season
    diff = torch.zeros_like(residual)
    diff[1:] = residual[1:] - residual[:-1]
    residual_energy = torch.sqrt(_moving_average(residual.pow(2), regime_kernel).clamp_min(1e-6))
    diff_energy = torch.sqrt(_moving_average(diff.pow(2), regime_kernel).clamp_min(1e-6))
    regime = 0.5 * residual_energy + 0.5 * diff_energy

    channel_targets = [trend, season, scale, regime]
    channel_targets = [_normalize_target(target) for target in channel_targets]
    channel_targets = [_compress_time(target, seq_len) for target in channel_targets]

    macro_targets = []
    for target in channel_targets:
        if projection_mode == "random":
            in_channels = x.shape[-1]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(projection_seed))
            projection = torch.randn(in_channels, d_model, generator=generator, dtype=torch.float32)
            projection = F.normalize(projection, dim=0).to(device)
            hidden_target = target @ projection
        elif projection_mode == "repeat":
            repeats = (d_model + target.shape[-1] - 1) // target.shape[-1]
            hidden_target = target.repeat(1, repeats)[:, :d_model]
        elif projection_mode == "residual":
            if projection_layer is None:
                raise ValueError("projection_layer is required when projection_mode='residual'.")
            with torch.no_grad():
                hidden_target = projection_layer(target)
        else:
            raise ValueError(
                "projection_mode must be one of ['random', 'repeat', 'residual'], "
                f"got {projection_mode}."
            )
        macro_targets.append(_normalize_target(hidden_target))
    return torch.stack(macro_targets, dim=0)


class PretrainedMacroRouter(nn.Module):
    def __init__(self, d_model, num_experts, hidden_dim, dropout):
        super().__init__()
        self.router = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, h_micro):
        mean = h_micro.mean(dim=1)
        std = h_micro.std(dim=1, unbiased=False)
        slope = h_micro[:, -1] - h_micro[:, 0]
        max_abs = h_micro.abs().amax(dim=1)
        stats = torch.cat([mean, std, slope, max_abs], dim=-1)
        return self.router(stats)


class PretrainedMacroSeedMoE(nn.Module):
    def __init__(
        self,
        d_model,
        seq_len,
        num_experts=4,
        top_k=2,
        hidden_dim=None,
        router_hidden_dim=None,
        seasonal_top_k=8,
        router_temperature=1.0,
        dropout=0.0,
        gamma_init=0.01,
        gamma_max=0.1,
        macro_proj_init="random",
        use_expert_adapters=True,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        if gamma_max <= 0:
            raise ValueError(f"gamma_max must be positive, got {gamma_max}.")

        hidden_dim = hidden_dim or d_model * 2
        router_hidden_dim = router_hidden_dim or hidden_dim
        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)
        self.gamma_max = float(gamma_max)
        self.router_temperature = float(router_temperature)
        if self.router_temperature <= 0:
            raise ValueError(f"router_temperature must be positive, got {router_temperature}.")

        self.seed_tokens = nn.Parameter(torch.randn(self.num_experts, seq_len, d_model) * 0.02)
        self.router = PretrainedMacroRouter(d_model, self.num_experts, router_hidden_dim, dropout)
        self.experts = nn.ModuleList(
            self._build_experts(d_model, hidden_dim, dropout, seasonal_top_k, seq_len)
        )
        if use_expert_adapters:
            self.expert_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, d_model),
                    )
                    for _ in range(self.num_experts)
                ]
            )
            for adapter in self.expert_adapters:
                nn.init.eye_(adapter[1].weight)
                nn.init.zeros_(adapter[1].bias)
        else:
            self.expert_adapters = nn.ModuleList([nn.Identity() for _ in range(self.num_experts)])
        self.macro_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        if macro_proj_init == "zero":
            nn.init.zeros_(self.macro_proj[1].weight)
            nn.init.zeros_(self.macro_proj[1].bias)
        elif macro_proj_init == "identity":
            nn.init.eye_(self.macro_proj[1].weight)
            nn.init.zeros_(self.macro_proj[1].bias)
        elif macro_proj_init != "random":
            raise ValueError(
                "macro_proj_init must be one of ['random', 'zero', 'identity'], "
                f"got {macro_proj_init}."
            )

        gamma_init = min(max(float(gamma_init), 1e-5), self.gamma_max - 1e-5)
        gamma_ratio = gamma_init / self.gamma_max
        self.gamma_logit = nn.Parameter(torch.logit(torch.tensor(gamma_ratio)))

    def _build_experts(self, d_model, hidden_dim, dropout, seasonal_top_k, seq_len):
        experts = []
        for idx in range(self.num_experts):
            if idx == 0:
                experts.append(TrendExpert(d_model, hidden_dim, dropout))
            elif idx == 1:
                experts.append(SeasonalExpert(d_model, hidden_dim, dropout, top_k=seasonal_top_k))
            elif idx == 2:
                experts.append(ScaleExpert(d_model, hidden_dim, dropout, seq_len))
            elif idx == 3:
                experts.append(RegimeVolatilityExpert(d_model, hidden_dim, dropout))
            else:
                experts.append(GenericMacroExpert(d_model, hidden_dim, dropout))
        return experts

    def expert_outputs(self):
        outputs = []
        for idx, expert in enumerate(self.experts):
            seed = self.seed_tokens[idx : idx + 1]
            outputs.append(self.expert_adapters[idx](expert(seed)).squeeze(0))
        return torch.stack(outputs, dim=0)

    def pretrain_loss(self, macro_targets, cosine_weight=0.1):
        outputs = self.expert_outputs()
        macro_targets = macro_targets[: self.num_experts].to(outputs.device)
        mse_loss = F.mse_loss(outputs, macro_targets)
        cosine_loss = 1.0 - F.cosine_similarity(
            outputs.flatten(start_dim=1),
            macro_targets.flatten(start_dim=1),
            dim=-1,
        ).mean()
        per_expert_mse = (outputs - macro_targets).pow(2).mean(dim=(1, 2)).detach()
        return mse_loss + cosine_weight * cosine_loss, mse_loss.detach(), cosine_loss.detach(), per_expert_mse

    def freeze_expert_bank(self, finetune_mode="none"):
        valid_modes = {"none", "seed", "seed_last", "all"}
        if finetune_mode not in valid_modes:
            raise ValueError(f"finetune_mode must be one of {sorted(valid_modes)}, got {finetune_mode}.")

        self.seed_tokens.requires_grad_(finetune_mode in {"seed", "seed_last", "all"})
        for expert in self.experts:
            for parameter in expert.parameters():
                parameter.requires_grad_(finetune_mode == "all")
        for adapter in self.expert_adapters:
            for parameter in adapter.parameters():
                parameter.requires_grad_(finetune_mode == "all")

        if finetune_mode == "seed_last":
            for module_group in (self.experts, self.expert_adapters):
                for module_container in module_group:
                    last_linear = None
                    for module in module_container.modules():
                        if isinstance(module, nn.Linear):
                            last_linear = module
                    if last_linear is not None:
                        last_linear.weight.requires_grad_(True)
                        if last_linear.bias is not None:
                            last_linear.bias.requires_grad_(True)
            for expert in self.experts:
                last_linear = None
                for module in expert.modules():
                    if isinstance(module, nn.Linear):
                        last_linear = module
                if last_linear is not None:
                    last_linear.weight.requires_grad_(True)
                    if last_linear.bias is not None:
                        last_linear.bias.requires_grad_(True)

    def forward(self, h_micro, return_aux=False):
        batch_size = h_micro.shape[0]
        router_logits = self.router(h_micro)
        router_probs = F.softmax(router_logits / self.router_temperature, dim=-1)
        top_values, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        expert_weights = router_probs.new_zeros(router_probs.shape)
        expert_weights.scatter_(dim=1, index=top_indices, src=top_weights)

        candidates = self.expert_outputs().unsqueeze(0).expand(batch_size, -1, -1, -1)
        macro_context = torch.sum(candidates * expert_weights[:, :, None, None], dim=1)
        macro_update = self.macro_proj(macro_context)
        h_fused = h_micro + self.effective_gamma() * macro_update

        if not return_aux:
            return h_fused

        mean_router_probs = router_probs.mean(dim=0)
        mean_expert_weights = expert_weights.mean(dim=0)
        top1_indices = torch.argmax(router_probs, dim=-1)
        top1_usage = F.one_hot(top1_indices, num_classes=self.num_experts).float().mean(dim=0)
        topk_usage = torch.zeros_like(router_probs)
        topk_usage.scatter_(dim=1, index=top_indices, value=1.0)
        topk_usage = topk_usage.mean(dim=0)
        load_balance_loss = self.num_experts * (mean_router_probs * mean_expert_weights).sum()
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        router_entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        flat_candidates = F.normalize(candidates[:, : self.num_experts].flatten(start_dim=2), dim=-1)
        cosine = torch.matmul(flat_candidates, flat_candidates.transpose(1, 2))
        off_diag = 1.0 - torch.eye(self.num_experts, device=cosine.device, dtype=cosine.dtype)
        expert_similarity = (cosine.abs() * off_diag).sum(dim=(1, 2)) / off_diag.sum().clamp_min(1.0)
        macro_update_ratio = macro_update.norm(dim=-1).mean() / h_micro.norm(dim=-1).mean().clamp_min(1e-8)

        aux = {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
            "router_entropy": router_entropy,
            "expert_similarity": expert_similarity.mean(),
            "macro_context": macro_context,
            "macro_candidates": candidates[:, : self.num_experts],
            "macro_update": macro_update,
            "macro_update_ratio": macro_update_ratio.detach(),
            "effective_macro_ratio": (self.effective_gamma() * macro_update_ratio).detach(),
            "expert_weights": mean_expert_weights.detach(),
            "expert_weights_batch": expert_weights,
            "router_probs": mean_router_probs.detach(),
            "router_probs_batch": router_probs,
            "top1_usage": top1_usage.detach(),
            "topk_usage": topk_usage.detach(),
        }
        return h_fused, aux

    def effective_gamma(self):
        return self.gamma_max * torch.sigmoid(self.gamma_logit)

    def expert_names(self):
        names = list(EXPERT_NAMES)
        if self.num_experts > len(names):
            names.extend(f"generic_{idx}" for idx in range(len(names), self.num_experts))
        return names[: self.num_experts]
