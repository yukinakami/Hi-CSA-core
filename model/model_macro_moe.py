import torch
import torch.nn as nn

from .DynamicConv import MicroScaleEmbedding
from .macro_moe import MacroSeedMoE
from .model import TemporalDecoder


class Hi_CSAMacroMoE(nn.Module):
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
        macro_num_experts=4,
        macro_top_k=2,
        macro_hidden_dim=None,
        router_hidden_dim=None,
        seasonal_top_k=8,
        router_temperature=1.0,
        macro_dropout=None,
        macro_gamma_init=0.01,
        macro_gamma_max=0.1,
        macro_condition_max=0.3,
        residual_mode="output",
        micro_only=False,
        use_revin=True,
    ):
        super().__init__()

        self.pred_len = pred_len
        self.in_channels = in_channels
        self.final_dim = final_dim
        self.seq_len = seq_len
        self.use_revin = use_revin
        self.revin_eps = 1e-5
        valid_residual_modes = {"none", "output", "feature", "both"}
        if residual_mode not in valid_residual_modes:
            raise ValueError(f"residual_mode must be one of {sorted(valid_residual_modes)}, got {residual_mode}.")
        self.residual_mode = residual_mode
        self.micro_only = micro_only

        self.input_dropout = nn.Dropout(dropout)
        self.input_residual_proj = nn.Linear(in_channels, final_dim)
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

        macro_dropout = dropout if macro_dropout is None else macro_dropout
        self.macro_moe = MacroSeedMoE(
            d_model=final_dim,
            seq_len=seq_len,
            num_experts=macro_num_experts,
            top_k=macro_top_k,
            hidden_dim=macro_hidden_dim,
            router_hidden_dim=router_hidden_dim,
            seasonal_top_k=seasonal_top_k,
            router_temperature=router_temperature,
            dropout=macro_dropout,
            gamma_init=macro_gamma_init,
            gamma_max=macro_gamma_max,
            condition_max=macro_condition_max,
        )
        self.decoder = TemporalDecoder(final_dim, seq_len, pred_len, in_channels, dropout)
        self.macro_decoder = TemporalDecoder(final_dim, seq_len, pred_len, in_channels, dropout)

    def forward(self, x_micro, return_aux=False):
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
        h_micro = self.micro_embedding(x_micro_feat)
        h_micro = h_micro + self.micro_position_embedding
        if self.micro_only:
            h_fused = h_micro
            aux = None
        elif return_aux:
            h_fused, aux = self.macro_moe(h_micro, return_aux=True)
        else:
            h_fused = self.macro_moe(h_micro)
            aux = None
        if self.residual_mode in {"feature", "both"}:
            h_fused = h_fused + self.input_residual_proj(x_micro_in)
        prediction = self.decoder(h_fused)
        if return_aux and not self.micro_only:
            macro_prediction = self.macro_decoder(aux["macro_update"])
        if self.residual_mode in {"output", "both"}:
            prediction = prediction + residual

        if self.use_revin:
            prediction = prediction * micro_std + micro_mean
            if return_aux and not self.micro_only:
                macro_prediction = macro_prediction * micro_std + micro_mean

        if return_aux:
            if self.micro_only:
                aux = self._micro_only_aux(prediction)
                return prediction, aux
            aux["macro_prediction"] = macro_prediction
            return prediction, aux
        return prediction

    def _micro_only_aux(self, prediction):
        zero = prediction.new_tensor(0.0)
        num_experts = self.macro_moe.num_experts
        zeros = prediction.new_zeros(num_experts)
        return {
            "load_balance_loss": zero,
            "router_z_loss": zero,
            "router_entropy": zero,
            "expert_similarity": zero,
            "macro_prediction": prediction.detach(),
            "macro_update_ratio": zero.detach(),
            "condition_scale_mean": zero.detach(),
            "condition_shift_mean": zero.detach(),
            "expert_weights": zeros.detach(),
            "router_probs": zeros.detach(),
            "top1_usage": zeros.detach(),
            "topk_usage": zeros.detach(),
        }

    def effective_macro_gamma(self):
        return self.macro_moe.effective_gamma()

    def macro_expert_names(self):
        return self.macro_moe.expert_names()
