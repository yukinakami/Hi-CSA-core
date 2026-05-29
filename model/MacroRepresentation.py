import math

import torch
import torch.nn as nn
from einops import rearrange

from .ContinousScalePredictor import Fourier


class GlobalSineWaveMixin:
    def build_global_waves(self, x, output_len=None):
        _, time_len, channels = x.shape
        _, frequency, amplitude, phase = self.fourier.components(x)

        k = min(self.macro_k, frequency.shape[1] // 2)
        frequency = frequency[:, :k]
        amplitude = amplitude[:, :k] * 2.0
        phase = phase[:, :k]

        if output_len is None:
            output_len = time_len + self.fourier.pred_len

        timeline = torch.arange(output_len, dtype=amplitude.dtype, device=x.device)
        timeline = rearrange(timeline, "t -> () () t ()")
        macro_waves = amplitude * torch.cos(2 * math.pi * frequency * timeline + phase)
        macro_waves = torch.clamp(torch.nan_to_num(macro_waves, nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)

        if k < self.macro_k:
            padding = macro_waves.new_zeros(macro_waves.shape[0], self.macro_k - k, macro_waves.shape[2], channels)
            macro_waves = torch.cat([macro_waves, padding], dim=1)
        return macro_waves


class IndividualLinearMacro(GlobalSineWaveMixin, nn.Module):
    def __init__(self, input_dim, final_dim, pred_len, macro_k, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.fourier = Fourier(pred_len, macro_k, low_freq=1)
        self.macro_k = macro_k
        self.shared_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, final_dim),
        )
        self.personal_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(final_dim),
                    nn.Linear(final_dim, final_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(final_dim, final_dim),
                )
                for _ in range(macro_k)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward_from_waves(self, macro_waves):
        shared = self.shared_embedding(macro_waves)
        shared = torch.clamp(torch.nan_to_num(shared, nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)
        personalized = [layer(shared[:, idx]) for idx, layer in enumerate(self.personal_embeddings)]
        macro_features = torch.stack(personalized, dim=1)
        macro_features = torch.clamp(torch.nan_to_num(macro_features, nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)
        return self.dropout(macro_features).mean(dim=2)
