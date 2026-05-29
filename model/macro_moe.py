import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroRouter(nn.Module):
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


class MacroConditioner(nn.Module):
    def __init__(self, d_model, num_experts, hidden_dim, dropout):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.conditioner = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts * d_model * 2),
        )

    def forward(self, h_micro):
        mean = h_micro.mean(dim=1)
        std = h_micro.std(dim=1, unbiased=False)
        slope = h_micro[:, -1] - h_micro[:, 0]
        max_abs = h_micro.abs().amax(dim=1)
        stats = torch.cat([mean, std, slope, max_abs], dim=-1)
        params = self.conditioner(stats).view(-1, self.num_experts, 2, self.d_model)
        scale, shift = params[:, :, 0], params[:, :, 1]
        return scale, shift


class TrendExpert(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.0, kernel_sizes=(3, 7, 15, 31)):
        super().__init__()
        self.kernel_sizes = tuple(int(kernel_size) for kernel_size in kernel_sizes)
        if any(kernel_size <= 0 or kernel_size % 2 == 0 for kernel_size in self.kernel_sizes):
            raise ValueError("TrendExpert kernel sizes must be positive odd integers.")
        self.scale_logits = nn.Parameter(torch.zeros(len(self.kernel_sizes)))

    def forward(self, seed):
        x = seed.transpose(1, 2)
        trends = []
        for kernel_size in self.kernel_sizes:
            trend = F.avg_pool1d(
                x,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                count_include_pad=False,
            )
            trends.append(trend.transpose(1, 2))

        weights = F.softmax(self.scale_logits, dim=0)
        stacked = torch.stack(trends, dim=1)
        return torch.sum(stacked * weights.view(1, -1, 1, 1), dim=1)


class SeasonalExpert(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.0, top_k=8):
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        self.top_k = int(top_k)

    def forward(self, seed):
        seq_len = seed.shape[1]
        spectrum = torch.fft.rfft(seed, dim=1)
        magnitude = spectrum.abs().mean(dim=-1)

        if magnitude.shape[1] <= 1:
            return seed.new_zeros(seed.shape)

        magnitude = magnitude.clone()
        magnitude[:, 0] = 0.0
        top_k = min(self.top_k, magnitude.shape[1] - 1)
        _, top_indices = torch.topk(magnitude, top_k, dim=1)

        mask = torch.zeros_like(magnitude)
        mask.scatter_(dim=1, index=top_indices, value=1.0)
        filtered = spectrum * mask.unsqueeze(-1)
        return torch.fft.irfft(filtered, n=seq_len, dim=1)


class ScaleExpert(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout, seq_len):
        super().__init__()
        self.channel_log_scale = nn.Parameter(torch.zeros(1, 1, d_model))
        self.temporal_log_scale = nn.Parameter(torch.zeros(1, seq_len, 1))
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, seed):
        center = seed.mean(dim=1, keepdim=True)
        scale = seed.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        normalized = (seed - center) / scale
        channel_scale = F.softplus(self.channel_log_scale) + 0.5
        temporal_scale = F.softplus(self.temporal_log_scale) + 0.5
        scaled = center + normalized * channel_scale * temporal_scale
        return seed + self.proj(scaled)


class RegimeVolatilityExpert(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()
        self.local_state = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=5,
            padding=2,
            groups=d_model,
        )
        self.regime_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, seed):
        state = self.local_state(seed.transpose(1, 2)).transpose(1, 2)
        gate = self.regime_gate(state)
        return seed + self.proj(state * gate)


class GenericMacroExpert(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, seed):
        return seed + self.block(seed)


class MacroSeedMoE(nn.Module):
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
        gamma_init=0.0,
        gamma_max=0.1,
        condition_max=0.3,
        macro_proj_init="zero",
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
        self.seasonal_top_k = int(seasonal_top_k)
        self.router_temperature = float(router_temperature)
        self.condition_max = float(condition_max)
        if self.router_temperature <= 0:
            raise ValueError(f"router_temperature must be positive, got {router_temperature}.")
        if self.condition_max <= 0:
            raise ValueError(f"condition_max must be positive, got {condition_max}.")

        self.seed_tokens = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.router = MacroRouter(d_model, self.num_experts, router_hidden_dim, dropout)
        self.conditioner = MacroConditioner(d_model, self.num_experts, router_hidden_dim, dropout)
        self.experts = nn.ModuleList(self._build_experts(d_model, hidden_dim, dropout, self.seasonal_top_k, seq_len))
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
        gamma_logit = torch.logit(torch.tensor(gamma_ratio))
        self.gamma_logit = nn.Parameter(gamma_logit)

    def _build_experts(self, d_model, hidden_dim, dropout, seasonal_top_k, seq_len):
        expert_factories = [
            TrendExpert,
            SeasonalExpert,
            ScaleExpert,
            RegimeVolatilityExpert,
        ]
        experts = []
        for idx in range(self.num_experts):
            if idx == 1:
                experts.append(SeasonalExpert(d_model, hidden_dim, dropout, top_k=seasonal_top_k))
            elif idx == 2:
                experts.append(ScaleExpert(d_model, hidden_dim, dropout, seq_len))
            else:
                expert_cls = expert_factories[idx] if idx < len(expert_factories) else GenericMacroExpert
                experts.append(expert_cls(d_model, hidden_dim, dropout))
        return experts

    def forward(self, h_micro, return_aux=False):
        batch_size = h_micro.shape[0]
        router_logits = self.router(h_micro)
        router_probs = F.softmax(router_logits / self.router_temperature, dim=-1)
        cond_scale, cond_shift = self.conditioner(h_micro)
        cond_scale = self.condition_max * torch.tanh(cond_scale)
        cond_shift = self.condition_max * torch.tanh(cond_shift)

        top_values, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        expert_weights = router_probs.new_zeros(router_probs.shape)
        expert_weights.scatter_(dim=1, index=top_indices, src=top_weights)

        seed = self.seed_tokens.expand(batch_size, -1, -1)
        base_candidates = torch.stack([expert(seed) for expert in self.experts], dim=1)
        candidates = base_candidates * (1.0 + cond_scale[:, :, None, :]) + cond_shift[:, :, None, :]
        macro_context = torch.sum(candidates * expert_weights[:, :, None, None], dim=1)
        macro_update = self.macro_proj(macro_context)

        gamma = self.effective_gamma()
        h_fused = h_micro + gamma * macro_update
        if not return_aux:
            return h_fused

        mean_router_probs = router_probs.mean(dim=0)
        mean_expert_weights = expert_weights.mean(dim=0)
        top1_indices = torch.argmax(router_probs, dim=-1)
        top1_usage = F.one_hot(top1_indices, num_classes=self.num_experts).float().mean(dim=0)
        topk_usage = torch.zeros_like(router_probs)
        topk_usage.scatter_(dim=1, index=top_indices, value=1.0)
        topk_usage = topk_usage.mean(dim=0)
        min_expert_prob = min(0.08, 1.0 / self.num_experts)
        load_balance_loss = self.num_experts * F.relu(min_expert_prob - mean_router_probs).pow(2).sum()
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        router_entropy = -(router_probs * torch.log(router_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        flat_candidates = F.normalize(base_candidates.flatten(start_dim=2), dim=-1)
        cosine = torch.matmul(flat_candidates, flat_candidates.transpose(1, 2))
        off_diag = 1.0 - torch.eye(self.num_experts, device=cosine.device, dtype=cosine.dtype)
        expert_similarity = (cosine.abs() * off_diag).sum(dim=(1, 2)) / off_diag.sum().clamp_min(1.0)
        macro_update_ratio = macro_update.norm(dim=-1).mean() / h_micro.norm(dim=-1).mean().clamp_min(1e-8)
        condition_scale_mean = cond_scale.abs().mean()
        condition_shift_mean = cond_shift.abs().mean()
        aux = {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
            "router_entropy": router_entropy,
            "expert_similarity": expert_similarity.mean(),
            "macro_update": macro_update,
            "macro_update_ratio": macro_update_ratio.detach(),
            "condition_scale_mean": condition_scale_mean.detach(),
            "condition_shift_mean": condition_shift_mean.detach(),
            "expert_weights": mean_expert_weights.detach(),
            "router_probs": mean_router_probs.detach(),
            "top1_usage": top1_usage.detach(),
            "topk_usage": topk_usage.detach(),
        }
        return h_fused, aux

    def effective_gamma(self):
        return self.gamma_max * torch.sigmoid(self.gamma_logit)

    def expert_names(self):
        names = ["trend", "seasonal", "scale", "regime"]
        if self.num_experts > len(names):
            names.extend(f"generic_{idx}" for idx in range(len(names), self.num_experts))
        return names[:self.num_experts]
