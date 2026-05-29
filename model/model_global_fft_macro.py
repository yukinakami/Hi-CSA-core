import math

import torch
import torch.nn as nn

from .CrossAttention import CrossAttention
from .DynamicConv import MicroScaleEmbedding
from .MacroRepresentation import IndividualLinearMacro
from .model import TemporalDecoder


class MicroResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout):
        super().__init__()
        self.raw_proj = nn.Linear(in_channels, hidden_dim)
        self.local_proj = nn.Linear(in_channels, hidden_dim)
        self.freq_proj = nn.Linear(in_channels, hidden_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_norm, h_micro):
        raw_residual = self.raw_proj(x_norm)
        local_residual = self.local_proj(x_norm - self._moving_average(x_norm, kernel_size=5))
        freq_residual = self.freq_proj(self._dominant_frequency_residual(x_norm, top_k=3))

        gates = torch.sigmoid(self.gate(h_micro.mean(dim=1))).view(x_norm.shape[0], 3, 1, h_micro.shape[-1])
        residual = torch.stack([raw_residual, local_residual, freq_residual], dim=1)
        residual = (residual * gates).sum(dim=1)
        return torch.clamp(self.gamma, 0.0, 1.0) * self.dropout(self.norm(residual))

    def _moving_average(self, x, kernel_size):
        padding = kernel_size // 2
        x_t = x.transpose(1, 2)
        smooth = torch.nn.functional.avg_pool1d(x_t, kernel_size=kernel_size, stride=1, padding=padding)
        return smooth.transpose(1, 2)

    def _dominant_frequency_residual(self, x, top_k):
        _, seq_len, _ = x.shape
        spectrum = torch.fft.rfft(x, dim=1)
        amplitudes = spectrum.abs()
        if amplitudes.shape[1] <= 1:
            return torch.zeros_like(x)

        amplitudes = amplitudes.clone()
        amplitudes[:, 0, :] = 0.0
        k = min(top_k, amplitudes.shape[1] - 1)
        top_indices = torch.topk(amplitudes, k=k, dim=1).indices
        mask = torch.zeros_like(spectrum, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)
        filtered = torch.where(mask, spectrum, torch.zeros_like(spectrum))
        reconstructed = torch.fft.irfft(filtered, n=seq_len, dim=1)
        return reconstructed


class ForecastResidualHead(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        seq_len,
        pred_len,
        period=24,
        fourier_top_k=1,
        gamma_init=0.16,
        gamma_max=0.8,
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period = int(period)
        self.fourier_top_k = int(fourier_top_k)
        self.gamma_max = float(gamma_max)
        self.linear_projection = nn.Linear(seq_len, pred_len)
        nn.init.zeros_(self.linear_projection.weight)
        nn.init.zeros_(self.linear_projection.bias)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channels * 3),
        )
        gamma_init = min(max(float(gamma_init), 1e-5), self.gamma_max - 1e-5)
        self.gamma_logit = nn.Parameter(torch.logit(torch.tensor(gamma_init / self.gamma_max)))

    def forward(self, x_norm, h_fused, macro_features):
        last_value = x_norm[:, -1:, :].repeat(1, self.pred_len, 1)
        seasonal = self._seasonal_repeat(x_norm) - last_value
        fourier = self._fourier_extrapolate(x_norm) - last_value
        linear = self.linear_projection((x_norm - x_norm[:, -1:, :]).transpose(1, 2)).transpose(1, 2)
        basis = torch.stack([seasonal, fourier, linear], dim=1)

        gate_input = torch.cat([h_fused.mean(dim=1), macro_features.mean(dim=1)], dim=-1)
        gates = torch.tanh(self.gate(gate_input)).view(x_norm.shape[0], 3, 1, self.in_channels)
        return self.effective_gamma() * (basis * gates).sum(dim=1)

    def _seasonal_repeat(self, x_norm):
        period = max(1, min(self.period, x_norm.shape[1]))
        seasonal_source = x_norm[:, -period:, :]
        repeats = (self.pred_len + period - 1) // period
        return seasonal_source.repeat(1, repeats, 1)[:, : self.pred_len, :]

    def _fourier_extrapolate(self, x_norm):
        _, seq_len, _ = x_norm.shape
        spectrum = torch.fft.rfft(x_norm, dim=1) / float(seq_len)
        amplitudes = spectrum.abs()
        if amplitudes.shape[1] <= 1 or self.fourier_top_k <= 0:
            return x_norm.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)

        amplitudes = amplitudes.clone()
        amplitudes[:, 0, :] = 0.0
        k = min(self.fourier_top_k, amplitudes.shape[1] - 1)
        top_indices = torch.topk(amplitudes, k=k, dim=1).indices
        top_coefficients = torch.gather(spectrum, dim=1, index=top_indices)
        future_steps = torch.arange(seq_len, seq_len + self.pred_len, device=x_norm.device, dtype=x_norm.dtype).view(1, self.pred_len, 1, 1)
        freq_indices = top_indices.to(dtype=x_norm.dtype).unsqueeze(1)
        phase = 2.0 * math.pi * future_steps * freq_indices / float(seq_len)
        exponent = torch.complex(torch.cos(phase), torch.sin(phase))
        seasonal = 2.0 * torch.real(top_coefficients.unsqueeze(1) * exponent).sum(dim=2)
        mean = spectrum[:, :1, :].real.repeat(1, self.pred_len, 1)
        return mean + seasonal

    def effective_gamma(self):
        return self.gamma_max * torch.sigmoid(self.gamma_logit)


class Hi_CSAGlobalFFTMacro(nn.Module):
    def __init__(
        self,
        in_channels,
        final_dim,
        seq_len,
        pred_len,
        flourier_k,
        gmm_k,
        num_gaussians,
        num_base,
        max_sigma,
        dropout,
        kernel_size,
        macro_k,
        num_heads,
        macro_dropout=None,
        cross_dropout=None,
        cross_gamma_init=0.0,
        cross_gamma_limit=0.0,
        use_revin=True,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.use_revin = use_revin
        self.revin_eps = 1e-5
        self.register_buffer("global_macro_waves", None, persistent=False)

        macro_dropout = dropout if macro_dropout is None else macro_dropout
        cross_dropout = dropout if cross_dropout is None else cross_dropout

        self.input_dropout = nn.Dropout(dropout)
        self.micro_embedding = MicroScaleEmbedding(
            in_channels,
            kernel_size,
            num_gaussians,
            num_base,
            max_sigma,
            in_channels,
            pred_len,
            flourier_k,
            gmm_k,
            hidden_dim=final_dim,
            dropout=dropout,
        )
        self.micro_position_embedding = nn.Parameter(torch.randn(1, seq_len, final_dim) * 0.02)
        self.micro_residual = MicroResidualBlock(in_channels, final_dim, dropout)
        self.original_residual_proj = nn.Linear(in_channels, final_dim)
        self.macro_encoder = IndividualLinearMacro(
            in_channels,
            final_dim,
            pred_len,
            macro_k,
            hidden_dim=128,
            dropout=macro_dropout,
        )
        self.cross_attention = CrossAttention(
            final_dim,
            num_heads,
            cross_dropout,
            macro_k,
            gamma_init=cross_gamma_init,
            gamma_limit=cross_gamma_limit,
        )
        self.decoder = TemporalDecoder(final_dim, seq_len, pred_len, in_channels, dropout)
        self.micro_aux_decoder = TemporalDecoder(final_dim, seq_len, pred_len, in_channels, dropout)
        self.forecast_residual_head = ForecastResidualHead(
            in_channels,
            final_dim,
            seq_len,
            pred_len,
            period=24,
            fourier_top_k=1,
            gamma_init=0.16,
            gamma_max=0.8,
            dropout=dropout,
        )

    @torch.no_grad()
    def set_global_macro(self, global_macro):
        if global_macro is None:
            self.global_macro_waves = None
            return
        if global_macro.dim() != 2:
            raise ValueError("global_macro must have shape [time, channels]")
        macro_series = global_macro.to(
            device=self.micro_position_embedding.device,
            dtype=self.micro_position_embedding.dtype,
        )
        self.global_macro_waves = self.macro_encoder.build_global_waves(
            macro_series.unsqueeze(0),
            output_len=macro_series.shape[0] + self.pred_len,
        ).detach()

    def forward(self, x_micro):
        if self.global_macro_waves is None:
            raise RuntimeError("Global FFT macro waves are not set. Call set_global_macro() before training/eval.")

        if self.use_revin:
            micro_mean = x_micro.mean(dim=1, keepdim=True).detach()
            micro_std = torch.sqrt(x_micro.var(dim=1, keepdim=True, unbiased=False) + self.revin_eps).detach()
            x_micro_in = (x_micro - micro_mean) / micro_std
        else:
            x_micro_in = x_micro
            micro_mean = None
            micro_std = None

        x_micro_feat = self.input_dropout(x_micro_in)
        micro_features = self.micro_embedding(x_micro_feat) + self.micro_position_embedding
        micro_features = micro_features + self.micro_residual(x_micro_in, micro_features)
        aux_prediction = self.micro_aux_decoder(micro_features)

        macro_features = self.macro_encoder.forward_from_waves(self.global_macro_waves)
        macro_features = macro_features.expand(x_micro.shape[0], -1, -1)
        fused_features = self.cross_attention(micro_features, macro_features)
        fused_features = fused_features + self.original_residual_proj(x_micro_in)

        base_prediction = self.decoder(fused_features)
        forecast_residual = self.forecast_residual_head(x_micro_in, fused_features, macro_features)
        prediction = base_prediction + forecast_residual

        if self.use_revin:
            base_prediction = base_prediction * micro_std + micro_mean
            prediction = prediction * micro_std + micro_mean
            aux_prediction = aux_prediction * micro_std + micro_mean
            forecast_residual = forecast_residual * micro_std

        return prediction, {
            "forecast": aux_prediction,
            "base_prediction": base_prediction.detach(),
            "forecast_residual": forecast_residual,
        }

    def effective_cross_gamma(self):
        return self.cross_attention.effective_gamma()

    def effective_forecast_residual_gamma(self):
        return self.forecast_residual_head.effective_gamma()
