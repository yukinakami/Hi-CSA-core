import torch
import torch.nn as nn

from .DynamicConv import MicroScaleEmbedding


class TemporalDecoder(nn.Module):
    def __init__(self, d_model, seq_len, pred_len, out_channels, dropout):
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.temporal_projection = nn.Linear(seq_len, pred_len)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, out_channels)

    def forward(self, context):
        context = self.input_norm(context)
        decoded = self.temporal_projection(context.transpose(1, 2)).transpose(1, 2)
        decoded = decoded + self.ffn(decoded)
        return self.projection(self.output_norm(decoded))


class Hi_CSA(nn.Module):
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
        use_revin=True,
    ):
        super().__init__()

        self.pred_len = pred_len
        self.in_channels = in_channels
        self.final_dim = final_dim
        self.seq_len = seq_len
        self.use_revin = use_revin
        self.revin_eps = 1e-5

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
        self.micro_position_embedding = nn.Parameter(
            torch.randn(1, seq_len, final_dim) * 0.02
        )
        self.decoder = TemporalDecoder(final_dim, seq_len, pred_len, in_channels, dropout)

    def forward(self, x_micro):
        if self.use_revin:
            micro_mean = x_micro.mean(dim=1, keepdim=True).detach()
            micro_std = torch.sqrt(
                x_micro.var(dim=1, keepdim=True, unbiased=False) + self.revin_eps
            ).detach()
            x_micro_in = (x_micro - micro_mean) / micro_std
        else:
            x_micro_in = x_micro
            micro_mean = None
            micro_std = None

        residual = x_micro_in[:, -1:, :].repeat(1, self.pred_len, 1)
        x_micro_feat = self.input_dropout(x_micro_in)
        micro_features = self.micro_embedding(x_micro_feat)
        micro_features = micro_features + self.micro_position_embedding
        prediction = self.decoder(micro_features)
        prediction = prediction + residual

        if self.use_revin:
            prediction = prediction * micro_std + micro_mean

        return prediction
