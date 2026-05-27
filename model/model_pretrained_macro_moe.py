import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from .DynamicConv import MicroScaleEmbedding
from .model import TemporalDecoder
from .pretrained_macro_moe import PretrainedMacroSeedMoE


class AlphaScaledCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.scaling = nn.Parameter(torch.empty((self.head_dim,), dtype=torch.float32))
        nn.init.zeros_(self.scaling)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        q_scale = (math.log2(math.e) / math.sqrt(self.head_dim)) * F.softplus(self.scaling)
        q = q * q_scale.view(1, 1, 1, self.head_dim).to(dtype=q.dtype, device=q.device)
        scores = torch.matmul(q, k.transpose(-2, -1))
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        output = self.out_proj(context)

        if need_weights:
            return output, attn.mean(dim=1)
        return output, None

    def alpha_scale(self):
        return (math.log2(math.e) / math.sqrt(self.head_dim)) * F.softplus(self.scaling)


class AlphaScaledTemporalAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        gamma_init=0.02,
        gamma_max=0.2,
        causal=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = AlphaScaledCrossAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.gamma_max = float(gamma_max)
        self.causal = causal
        gamma_init = min(max(float(gamma_init), 1e-5), self.gamma_max - 1e-5)
        gamma_ratio = gamma_init / self.gamma_max
        self.gamma_logit = nn.Parameter(torch.logit(torch.tensor(gamma_ratio)))

    def forward(self, hidden):
        normed = self.norm(hidden)
        attn_mask = None
        if self.causal:
            seq_len = hidden.shape[1]
            attn_mask = torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            attn_mask = torch.triu(attn_mask, diagonal=1).view(1, 1, seq_len, seq_len)
        update, _ = self.attention(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        return hidden + self.effective_gamma() * self.dropout(update)

    def effective_gamma(self):
        return self.gamma_max * torch.sigmoid(self.gamma_logit)

    def alpha_scale(self):
        return self.attention.alpha_scale()


class Hi_CSAPretrainedMacroMoE(nn.Module):
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
        macro_proj_init="random",
        macro_fusion_mode="residual",
        macro_cross_heads=12,
        macro_cross_memory="aggregate",
        macro_cross_gamma_init=0.01,
        macro_cross_gamma_max=0.1,
        use_expert_adapters=True,
        macro_output_residual=False,
        macro_output_gamma_init=0.01,
        macro_output_gamma_max=0.1,
        macro_raw_residual_mode="none",
        macro_raw_period=24,
        macro_raw_fourier_k=4,
        macro_raw_gamma_init=0.01,
        macro_raw_gamma_max=0.1,
        micro_attention_layers=0,
        micro_attention_heads=12,
        micro_attention_gamma_init=0.02,
        micro_attention_gamma_max=0.2,
        micro_attention_causal=False,
        residual_mode="feature",
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
        valid_fusion_modes = {"residual", "cross_attention", "residual_cross_attention"}
        if macro_fusion_mode not in valid_fusion_modes:
            raise ValueError(
                f"macro_fusion_mode must be one of {sorted(valid_fusion_modes)}, "
                f"got {macro_fusion_mode}."
            )
        self.macro_fusion_mode = macro_fusion_mode
        valid_cross_memories = {"aggregate", "experts", "experts_weighted"}
        if macro_cross_memory not in valid_cross_memories:
            raise ValueError(
                f"macro_cross_memory must be one of {sorted(valid_cross_memories)}, "
                f"got {macro_cross_memory}."
            )
        self.macro_cross_memory = macro_cross_memory
        self.macro_cross_gamma_max = float(macro_cross_gamma_max)
        self.macro_output_residual = macro_output_residual
        self.macro_output_gamma_max = float(macro_output_gamma_max)
        valid_raw_modes = {
            "none",
            "seasonal",
            "trend_seasonal",
            "fourier",
            "seasonal_fourier",
            "trend_seasonal_fourier",
            "seasonal_fourier_linear",
        }
        if macro_raw_residual_mode not in valid_raw_modes:
            raise ValueError(
                f"macro_raw_residual_mode must be one of {sorted(valid_raw_modes)}, "
                f"got {macro_raw_residual_mode}."
            )
        self.macro_raw_residual_mode = macro_raw_residual_mode
        self.macro_raw_period = int(macro_raw_period)
        self.macro_raw_fourier_k = int(macro_raw_fourier_k)
        self.macro_raw_gamma_max = float(macro_raw_gamma_max)
        self.micro_attention_layers = int(micro_attention_layers)

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
        self.micro_position_embedding = nn.Parameter(torch.randn(1, seq_len, final_dim) * 0.02)
        if self.micro_attention_layers > 0:
            if final_dim % micro_attention_heads != 0:
                raise ValueError(
                    f"final_dim ({final_dim}) must be divisible by micro_attention_heads ({micro_attention_heads})."
                )
            self.micro_attention_blocks = nn.ModuleList(
                AlphaScaledTemporalAttentionBlock(
                    final_dim,
                    micro_attention_heads,
                    dropout=dropout,
                    gamma_init=micro_attention_gamma_init,
                    gamma_max=micro_attention_gamma_max,
                    causal=micro_attention_causal,
                )
                for _ in range(self.micro_attention_layers)
            )
        else:
            self.micro_attention_blocks = nn.ModuleList()

        macro_dropout = dropout if macro_dropout is None else macro_dropout
        self.macro_moe = PretrainedMacroSeedMoE(
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
            macro_proj_init=macro_proj_init,
            use_expert_adapters=use_expert_adapters,
        )
        if self.macro_fusion_mode in {"cross_attention", "residual_cross_attention"}:
            if final_dim % macro_cross_heads != 0:
                raise ValueError(
                    f"final_dim ({final_dim}) must be divisible by macro_cross_heads ({macro_cross_heads})."
                )
            self.cross_query_norm = nn.LayerNorm(final_dim)
            self.cross_kv_norm = nn.LayerNorm(final_dim)
            self.macro_cross_expert_embedding = nn.Parameter(
                torch.randn(1, macro_num_experts, 1, final_dim) * 0.02
            )
            self.macro_cross_attention = AlphaScaledCrossAttention(
                embed_dim=final_dim,
                num_heads=macro_cross_heads,
                dropout=dropout,
            )
            self.macro_cross_out = nn.Sequential(
                nn.LayerNorm(final_dim),
                nn.Linear(final_dim, final_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(final_dim, final_dim),
            )
            cross_gamma_init = min(max(float(macro_cross_gamma_init), 1e-5), self.macro_cross_gamma_max - 1e-5)
            cross_gamma_ratio = cross_gamma_init / self.macro_cross_gamma_max
            self.macro_cross_gamma_logit = nn.Parameter(torch.logit(torch.tensor(cross_gamma_ratio)))
        else:
            self.cross_query_norm = None
            self.cross_kv_norm = None
            self.macro_cross_expert_embedding = None
            self.macro_cross_attention = None
            self.macro_cross_out = None
            self.macro_cross_gamma_logit = None
        self.decoder = TemporalDecoder(final_dim, seq_len, pred_len, in_channels, dropout)
        if self.macro_output_residual:
            self.macro_decoder = TemporalDecoder(final_dim, seq_len, pred_len, in_channels, dropout)
            gamma_init = min(max(float(macro_output_gamma_init), 1e-5), self.macro_output_gamma_max - 1e-5)
            gamma_ratio = gamma_init / self.macro_output_gamma_max
            self.macro_output_gamma_logit = nn.Parameter(torch.logit(torch.tensor(gamma_ratio)))
        else:
            self.macro_decoder = None
            self.macro_output_gamma_logit = None
        if self.macro_raw_residual_mode != "none":
            raw_components = 1
            if self.macro_raw_residual_mode in {"trend_seasonal", "seasonal_fourier"}:
                raw_components = 2
            elif self.macro_raw_residual_mode == "trend_seasonal_fourier":
                raw_components = 3
            elif self.macro_raw_residual_mode == "seasonal_fourier_linear":
                raw_components = 3
            if self.macro_raw_residual_mode == "seasonal_fourier_linear":
                self.raw_linear_projection = nn.Linear(seq_len, pred_len)
                nn.init.zeros_(self.raw_linear_projection.weight)
                nn.init.zeros_(self.raw_linear_projection.bias)
            else:
                self.raw_linear_projection = None
            self.raw_macro_gate = nn.Sequential(
                nn.LayerNorm(final_dim * 2),
                nn.Linear(final_dim * 2, final_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(final_dim, in_channels * raw_components),
            )
            gamma_init = min(max(float(macro_raw_gamma_init), 1e-5), self.macro_raw_gamma_max - 1e-5)
            gamma_ratio = gamma_init / self.macro_raw_gamma_max
            self.macro_raw_gamma_logit = nn.Parameter(torch.logit(torch.tensor(gamma_ratio)))
        else:
            self.raw_macro_gate = None
            self.macro_raw_gamma_logit = None
            self.raw_linear_projection = None

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
        for block in self.micro_attention_blocks:
            h_micro = block(h_micro)

        if self.micro_only:
            h_fused = h_micro
            aux = None
        elif return_aux or self.macro_output_residual or self.raw_macro_gate is not None:
            h_macro_residual, aux = self.macro_moe(h_micro, return_aux=True)
            h_fused = self._fuse_macro(h_micro, h_macro_residual, aux, return_aux=return_aux)
        else:
            if self.macro_fusion_mode == "residual":
                h_fused = self.macro_moe(h_micro)
            else:
                h_macro_residual, aux = self.macro_moe(h_micro, return_aux=True)
                h_fused = self._fuse_macro(h_micro, h_macro_residual, aux, return_aux=False)
                aux = None
            aux = None

        if self.residual_mode in {"feature", "both"}:
            h_fused = h_fused + self.input_residual_proj(x_micro_in)

        base_prediction = self.decoder(h_fused)
        prediction = base_prediction
        macro_correction = None
        if self.macro_output_residual and not self.micro_only:
            macro_prediction = self.macro_decoder(aux["macro_context"])
            macro_output_gamma = self.effective_macro_output_gamma()
            macro_correction = macro_output_gamma * macro_prediction
            prediction = prediction + macro_correction
            if return_aux:
                macro_prediction_ratio = macro_correction.norm(dim=-1).mean() / base_prediction.norm(dim=-1).mean().clamp_min(1e-8)
                aux["macro_output_ratio"] = macro_prediction_ratio.detach()
                aux["effective_macro_output_ratio"] = macro_prediction_ratio.detach()
        elif return_aux and not self.micro_only:
            zero = prediction.new_tensor(0.0)
            aux["macro_output_ratio"] = zero.detach()
            aux["effective_macro_output_ratio"] = zero.detach()
        raw_macro_correction = None
        if self.raw_macro_gate is not None and not self.micro_only:
            raw_macro_correction, raw_macro_aux = self._raw_macro_residual(
                x_micro_in,
                h_micro,
                aux["macro_context"],
                return_aux=return_aux,
            )
            prediction = prediction + raw_macro_correction
            if return_aux:
                raw_ratio = raw_macro_correction.norm(dim=-1).mean() / base_prediction.norm(dim=-1).mean().clamp_min(1e-8)
                aux["raw_macro_ratio"] = raw_ratio.detach()
                aux["effective_raw_macro_ratio"] = raw_ratio.detach()
                aux.update(raw_macro_aux)
        elif return_aux and not self.micro_only:
            zero = prediction.new_tensor(0.0)
            aux["raw_macro_ratio"] = zero.detach()
            aux["effective_raw_macro_ratio"] = zero.detach()
        if self.residual_mode in {"output", "both"}:
            base_prediction = base_prediction + residual
            prediction = prediction + residual

        if self.use_revin:
            base_prediction = base_prediction * micro_std + micro_mean
            prediction = prediction * micro_std + micro_mean
            if macro_correction is not None:
                macro_correction = macro_correction * micro_std
            if raw_macro_correction is not None:
                raw_macro_correction = raw_macro_correction * micro_std

        if return_aux:
            if self.micro_only:
                return prediction, self._micro_only_aux(prediction)
            aux["base_prediction"] = base_prediction.detach()
            if macro_correction is None:
                aux["macro_correction"] = prediction.new_zeros(prediction.shape)
            else:
                aux["macro_correction"] = macro_correction
            if raw_macro_correction is None:
                aux["raw_macro_correction"] = prediction.new_zeros(prediction.shape)
            else:
                aux["raw_macro_correction"] = raw_macro_correction
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
            "macro_update_ratio": zero.detach(),
            "effective_macro_ratio": zero.detach(),
            "macro_cross_ratio": zero.detach(),
            "effective_macro_cross_ratio": zero.detach(),
            "macro_cross_alpha_mean": zero.detach(),
            "macro_cross_alpha_std": zero.detach(),
            "macro_output_ratio": zero.detach(),
            "effective_macro_output_ratio": zero.detach(),
            "raw_macro_ratio": zero.detach(),
            "effective_raw_macro_ratio": zero.detach(),
            "base_prediction": prediction.detach(),
            "macro_correction": prediction.new_zeros(prediction.shape),
            "raw_macro_correction": prediction.new_zeros(prediction.shape),
            "expert_weights": zeros.detach(),
            "router_probs": zeros.detach(),
            "top1_usage": zeros.detach(),
            "topk_usage": zeros.detach(),
        }

    def _fuse_macro(self, h_micro, h_macro_residual, aux, return_aux=False):
        if self.macro_fusion_mode == "residual":
            if return_aux:
                zero = h_micro.new_tensor(0.0)
                aux["macro_cross_ratio"] = zero.detach()
                aux["effective_macro_cross_ratio"] = zero.detach()
                aux["macro_cross_alpha_mean"] = zero.detach()
                aux["macro_cross_alpha_std"] = zero.detach()
            return h_macro_residual

        macro_context = aux["macro_context"]
        key_value_memory = macro_context
        if self.macro_cross_memory in {"experts", "experts_weighted"} and "macro_candidates" in aux:
            macro_candidates = aux["macro_candidates"]
            key_value_memory = macro_candidates + self.macro_cross_expert_embedding
            if self.macro_cross_memory == "experts_weighted":
                expert_weights = aux["expert_weights_batch"]
                key_value_memory = key_value_memory * expert_weights[:, :, None, None]
            key_value_memory = key_value_memory.flatten(1, 2)

        query = self.cross_query_norm(h_micro)
        key_value = self.cross_kv_norm(key_value_memory)
        cross_context, _ = self.macro_cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )
        cross_update = self.macro_cross_out(cross_context)
        cross_ratio = cross_update.norm(dim=-1).mean() / h_micro.norm(dim=-1).mean().clamp_min(1e-8)
        cross_gamma = self.effective_macro_cross_gamma()

        if self.macro_fusion_mode == "cross_attention":
            h_fused = h_micro + cross_gamma * cross_update
        else:
            h_fused = h_macro_residual + cross_gamma * cross_update

        if return_aux:
            aux["macro_cross_update"] = cross_update
            aux["macro_cross_ratio"] = cross_ratio.detach()
            aux["effective_macro_cross_ratio"] = (cross_gamma * cross_ratio).detach()
            aux["macro_cross_alpha_mean"] = self.macro_cross_attention.alpha_scale().mean().detach()
            aux["macro_cross_alpha_std"] = self.macro_cross_attention.alpha_scale().std(unbiased=False).detach()
        return h_fused

    def _raw_macro_residual(self, x_micro, h_micro, macro_context, return_aux=False):
        components = []
        if self.macro_raw_residual_mode in {
            "seasonal",
            "trend_seasonal",
            "seasonal_fourier",
            "trend_seasonal_fourier",
            "seasonal_fourier_linear",
        }:
            period = max(1, min(self.macro_raw_period, x_micro.shape[1]))
            seasonal_source = x_micro[:, -period:, :]
            repeats = (self.pred_len + period - 1) // period
            seasonal = seasonal_source.repeat(1, repeats, 1)[:, : self.pred_len, :]
            last_value = x_micro[:, -1:, :].repeat(1, self.pred_len, 1)
            components.append(seasonal - last_value)

        if self.macro_raw_residual_mode in {"trend_seasonal", "trend_seasonal_fourier"}:
            half = max(1, x_micro.shape[1] // 2)
            early_mean = x_micro[:, :half, :].mean(dim=1, keepdim=True)
            late_mean = x_micro[:, -half:, :].mean(dim=1, keepdim=True)
            slope = (late_mean - early_mean) / float(max(1, half))
            horizon = torch.arange(
                1,
                self.pred_len + 1,
                device=x_micro.device,
                dtype=x_micro.dtype,
            ).view(1, self.pred_len, 1)
            components.append(slope * horizon)

        if self.macro_raw_residual_mode in {
            "fourier",
            "seasonal_fourier",
            "trend_seasonal_fourier",
            "seasonal_fourier_linear",
        }:
            fourier = self._fourier_extrapolate(x_micro, self.pred_len, self.macro_raw_fourier_k)
            last_value = x_micro[:, -1:, :].repeat(1, self.pred_len, 1)
            components.append(fourier - last_value)

        if self.macro_raw_residual_mode == "seasonal_fourier_linear":
            centered = x_micro - x_micro[:, -1:, :]
            linear_correction = self.raw_linear_projection(centered.transpose(1, 2)).transpose(1, 2)
            components.append(linear_correction)

        basis = torch.stack(components, dim=1)
        gate_input = torch.cat([h_micro.mean(dim=1), macro_context.mean(dim=1)], dim=-1)
        gates = torch.tanh(self.raw_macro_gate(gate_input))
        gates = gates.view(x_micro.shape[0], len(components), 1, self.in_channels)
        weighted_basis = basis * gates
        correction = self.effective_macro_raw_gamma() * weighted_basis.sum(dim=1)
        return correction, {}

    def _fourier_extrapolate(self, x_micro, pred_len, top_k):
        batch_size, seq_len, channels = x_micro.shape
        spectrum = torch.fft.rfft(x_micro, dim=1) / float(seq_len)
        amplitudes = spectrum.abs()
        if amplitudes.shape[1] <= 1 or top_k <= 0:
            return x_micro.mean(dim=1, keepdim=True).repeat(1, pred_len, 1)

        amplitudes = amplitudes.clone()
        amplitudes[:, 0, :] = 0.0
        k = min(int(top_k), amplitudes.shape[1] - 1)
        top_indices = torch.topk(amplitudes, k=k, dim=1).indices
        top_coefficients = torch.gather(spectrum, dim=1, index=top_indices)

        future_steps = torch.arange(
            seq_len,
            seq_len + pred_len,
            device=x_micro.device,
            dtype=x_micro.dtype,
        ).view(1, pred_len, 1, 1)
        freq_indices = top_indices.to(dtype=x_micro.dtype).unsqueeze(1)
        phase = 2.0 * math.pi * future_steps * freq_indices / float(seq_len)
        exponent = torch.complex(torch.cos(phase), torch.sin(phase))
        seasonal = 2.0 * torch.real(top_coefficients.unsqueeze(1) * exponent).sum(dim=2)
        mean = spectrum[:, :1, :].real.repeat(1, pred_len, 1)
        return mean + seasonal

    def effective_macro_gamma(self):
        return self.macro_moe.effective_gamma()

    def effective_macro_output_gamma(self):
        if self.macro_output_gamma_logit is None:
            return self.macro_moe.effective_gamma().new_tensor(0.0)
        return self.macro_output_gamma_max * torch.sigmoid(self.macro_output_gamma_logit)

    def effective_macro_raw_gamma(self):
        if self.macro_raw_gamma_logit is None:
            return self.macro_moe.effective_gamma().new_tensor(0.0)
        return self.macro_raw_gamma_max * torch.sigmoid(self.macro_raw_gamma_logit)

    def effective_macro_cross_gamma(self):
        if self.macro_cross_gamma_logit is None:
            return self.macro_moe.effective_gamma().new_tensor(0.0)
        return self.macro_cross_gamma_max * torch.sigmoid(self.macro_cross_gamma_logit)

    def macro_expert_names(self):
        return self.macro_moe.expert_names()

    def freeze_pretrained_experts(self, finetune_mode="none"):
        self.macro_moe.freeze_expert_bank(finetune_mode=finetune_mode)
