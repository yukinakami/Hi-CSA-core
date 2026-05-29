import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout,
        macro_k,
        gamma_init=0.0,
        gamma_limit=0.0,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.gamma_limit = float(gamma_limit)

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(embedding_dim)
        self.freq_position_embedding = nn.Embedding(macro_k, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, micro_features, macro_features):
        batch_size, query_len, _ = micro_features.shape
        key_len = macro_features.shape[1]

        macro_features = self._add_macro_position(macro_features)
        q = self.q_proj(self.norm_q(micro_features))
        k = self.k_proj(self.norm_kv(macro_features))
        v = self.v_proj(self.norm_kv(macro_features))

        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = torch.clamp(torch.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=-50.0), -20.0, 20.0)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.embedding_dim)
        update = self.out_proj(context)
        update = torch.clamp(torch.nan_to_num(update, nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)
        return micro_features + self.effective_gamma() * update

    def _add_macro_position(self, macro_features):
        _, token_count, _ = macro_features.shape
        num_bins = self.freq_position_embedding.num_embeddings
        freq_ids = torch.arange(token_count, device=macro_features.device)
        freq_ids = (freq_ids * num_bins // token_count).clamp(0, num_bins - 1)
        return macro_features + self.freq_position_embedding(freq_ids).unsqueeze(0)

    def effective_gamma(self):
        if self.gamma_limit > 0:
            return torch.clamp(self.gamma, -self.gamma_limit, self.gamma_limit)
        return self.gamma
