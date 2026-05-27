import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .ContinousScalePredictor import CScalePredictor


class MicroScaleEmbedding(nn.Module):
    def __init__(self, input_dim, kernel_size, num_gaussians, num_base, max_sigma, in_channels, pred_len, flourier_k, gmm_k, hidden_dim, dropout=0.0):
        super().__init__()
        if num_gaussians != gmm_k:
            raise ValueError(
                "num_gaussians and gmm_k both describe the GMM component count; "
                f"got num_gaussians={num_gaussians}, gmm_k={gmm_k}."
            )
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_gaussians = num_gaussians
        self.num_base = num_base
        self.padding = kernel_size // 2

        self.gmm = CScalePredictor(in_channels, pred_len, flourier_k, gmm_k)
        self.equal_kernel = EqualDyconvCalculation(kernel_size, num_base, max_sigma)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.raw_proj = nn.Linear(input_dim, hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, L, D = x.shape

        mus, sigma, weights = self.gmm(x)   # (B, D, N)
        if mus.shape[1] != D:
            mus = mus.mean(dim=1, keepdim=True).expand(-1, D, -1)
            sigma = sigma.mean(dim=1, keepdim=True).expand(-1, D, -1)
            weights = weights.mean(dim=1, keepdim=True).expand(-1, D, -1)

        mus = torch.nan_to_num(mus, nan=1.0, posinf=self.equal_kernel.max_sigma, neginf=1e-4)
        sigma = torch.nan_to_num(sigma, nan=1.0, posinf=self.equal_kernel.max_sigma, neginf=1e-4)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)

        mus = mus.clamp(1e-3, self.equal_kernel.max_sigma)
        sigma = sigma.clamp(1e-2, self.equal_kernel.max_sigma)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        kernels = self.equal_kernel(mus, sigma, weights) # (B, D, K)
        
        x_perm = x.permute(0 ,2, 1) # (B, D, L)

        # dynamic conv kernel
        x_unfold = F.unfold(
            x_perm.unsqueeze(-1),
            kernel_size=(self.kernel_size, 1),
            padding=(self.padding, 0)
        )

        x_unfold = x_unfold.view(B, D, self.kernel_size, -1)
        kernels_exp = kernels.unsqueeze(-1)
        out_fold = (x_unfold * kernels_exp).sum(dim=2, keepdim=True)
        out_fold = out_fold[..., :L]

        #out_fold = (x_unfold * kernels_exp).sum(dim=2, keepdim=True)

        micro_features = out_fold.squeeze(2).permute(0, 2, 1)
        micro_features = torch.nan_to_num(micro_features, nan=0.0, posinf=1e3, neginf=-1e3)
        micro_features = torch.clamp(micro_features, -1e3, 1e3)
        filtered_features = self.proj(micro_features)
        raw_features = self.raw_proj(x)
        micro_features = self.feature_norm(filtered_features + raw_features)
        micro_features = torch.nan_to_num(micro_features, nan=0.0, posinf=1e3, neginf=-1e3)
        micro_features = self.dropout(micro_features)

        return micro_features


class EqualDyconvCalculation(nn.Module):
    def __init__(self, kernel_size, num_base, max_sigma):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
        if num_base <= 0:
            raise ValueError(f"num_base must be positive, got {num_base}.")
        if max_sigma <= 0:
            raise ValueError(f"max_sigma must be positive, got {max_sigma}.")

        self.kernel_size = kernel_size
        self.num_base = num_base
        self.max_sigma = max_sigma
        min_sigma = min(1.0, float(max_sigma))
        base_sigmas = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(min_sigma)),
                torch.log(torch.tensor(float(max_sigma))),
                num_base
            )
        )
        self.register_buffer('base_sigmas', base_sigmas)

    def forward(self, mus, sigmas, weights):

        equal_kernel = self.EqualKernelGenerator(mus, sigmas, weights)

        return equal_kernel

    def EqualKernelGenerator(self, mus, sigmas, weights):
        
        device = mus.device
        dtype = mus.dtype
        t_grid = torch.arange(self.kernel_size, device=device, dtype=dtype)
        t_grid = t_grid - (self.kernel_size - 1) / 2
        grids = t_grid.unsqueeze(0).expand(self.num_base, -1)  # [num_base, K]
        sigma_grid = self.base_sigmas.unsqueeze(1).to(device=device, dtype=dtype)  # [num_base, 1]

        base_kernels = torch.exp(-grids**2 / (2 * sigma_grid ** 2))
        base_kernels = base_kernels / (base_kernels.sum(dim=1, keepdim=True) + 1e-8)

        fusion_weight = self._integrate_scale_distribution(mus, sigmas, weights)
        fusion_weight = torch.nan_to_num(fusion_weight, nan=0.0, posinf=1.0, neginf=0.0)

        final_kernels = torch.einsum('bdn,nk -> bdk', fusion_weight, base_kernels)
        final_kernels = torch.nan_to_num(final_kernels, nan=0.0, posinf=1.0, neginf=0.0)
        final_kernels = final_kernels / (final_kernels.sum(dim=-1, keepdim=True) + 1e-8)

        return final_kernels

    def _integrate_scale_distribution(self, mus, sigmas, weights):
        device = mus.device
        dtype = mus.dtype
        base_sigmas = self.base_sigmas.to(device=device, dtype=dtype)

        if self.num_base == 1:
            lower = base_sigmas.new_tensor([0.0])
            upper = base_sigmas.new_tensor([float(self.max_sigma)])
        else:
            boundaries = torch.sqrt(base_sigmas[:-1] * base_sigmas[1:])
            lower = torch.cat([base_sigmas.new_tensor([0.0]), boundaries])
            upper = torch.cat([boundaries, base_sigmas.new_tensor([float(self.max_sigma)])])

        lower = lower.view(1, 1, 1, self.num_base)
        upper = upper.view(1, 1, 1, self.num_base)
        mus = mus.unsqueeze(-1)
        sigmas = sigmas.unsqueeze(-1).clamp_min(1e-4)
        weights = weights.unsqueeze(-1)

        denom = sigmas * math.sqrt(2.0)
        lower_z = torch.clamp((lower - mus) / denom, -10.0, 10.0)
        upper_z = torch.clamp((upper - mus) / denom, -10.0, 10.0)
        component_mass = 0.5 * (torch.erf(upper_z) - torch.erf(lower_z))
        component_mass = torch.clamp(component_mass, min=0.0)

        fusion_weight = torch.sum(component_mass * weights, dim=2)
        mass_sum = fusion_weight.sum(dim=-1, keepdim=True)
        normalized = fusion_weight / (mass_sum + 1e-8)
        uniform = torch.full_like(fusion_weight, 1.0 / self.num_base)
        return torch.where(mass_sum > 1e-8, normalized, uniform)
