from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用RoPE位置编码。
    输入形状: [batch, seq, dim] 或 [batch*heads, seq, dim]
    """
    # 转为复数域并应用旋转变换
    xq_cis = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) * freqs_cis
    xk_cis = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) * freqs_cis
    
    # 转回实数域并恢复原始形状
    xq_out = torch.view_as_real(xq_cis).flatten(-2).type_as(xq)
    xk_out = torch.view_as_real(xk_cis).flatten(-2).type_as(xk)
    return xq_out, xk_out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLUFeedForward(nn.Module):
    """SwiGLU前馈网络，不包含内部dropout（由外层的resid_dropout统一处理）。"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, dim)
        # Note: dropout参数保留用于接口兼容性，但不使用

    def forward(self, x: Tensor) -> Tensor:
        gated = F.silu(self.gate(x))
        activated = gated * self.value(x)
        return self.proj(activated)


class TransformerXLEncoderLayer(nn.Module):
    """Minimal encoder block with RoPE attention and SwiGLU FFN."""

    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float, max_seq_len: int, rotary_base: float = 10000.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("embedding dimension must be divisible by number of heads.")

        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim % 2 != 0:
            raise ValueError("head dimension must be even to use rotary embeddings.")

        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len, rotary_base)
        self.register_buffer("freqs_cis", freqs_cis)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.ffn = SwiGLUFeedForward(dim, ff_dim, dropout)

    def _shape(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _get_freqs(self, start_pos: int, seq_len: int) -> torch.Tensor:
        cache_len = self.freqs_cis.shape[0]
        if cache_len == 0:
            raise ValueError("RoPE cache has zero length.")
        positions = torch.arange(seq_len, device=self.freqs_cis.device, dtype=torch.long)
        positions = (positions + start_pos) % cache_len
        return self.freqs_cis[positions]

    def _merge_with_cache(
        self,
        new_k: Tensor,
        new_v: Tensor,
        cache: Optional[Tuple[Tensor, Tensor]],
        cache_limit: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if cache is None:
            k_attn, v_attn = new_k, new_v  # [batch, heads, seq, head_dim]
        else:
            k_cached, v_cached = cache
            batch_size = new_k.size(0)
            if k_cached.size(0) != batch_size:
                raise ValueError(
                    f"Cache batch dimension {k_cached.size(0)} does not match new batch {batch_size}"
                )

            k_attn = torch.cat([k_cached, new_k], dim=2)  # [batch, heads, prev + seq, head_dim]
            v_attn = torch.cat([v_cached, new_v], dim=2)  # [batch, heads, prev + seq, head_dim]

        if cache_limit > 0:
            k_attn = k_attn[:, :, -cache_limit:, :]
            v_attn = v_attn[:, :, -cache_limit:, :]

        return k_attn, v_attn, k_attn.detach(), v_attn.detach()

    @staticmethod
    def _causal_mask(seq_len: int, prev_len: int, total_len: int, device: torch.device) -> Optional[Tensor]:
        """创建因果注意力mask。左半部分（缓存）全可见，右半部分（当前序列）应用因果mask。"""
        if seq_len <= 1 or prev_len < 0:
            return None
    
        mask = torch.zeros(seq_len, total_len, device=device, dtype=torch.bool)
        
        if seq_len > 1 and prev_len < total_len:
            future_cols = total_len - prev_len
            mask[:, prev_len:total_len] = torch.triu(
                torch.ones(seq_len, future_cols, dtype=torch.bool, device=device),
                diagonal=1,
            )

        return mask

    def _apply_attention(
        self,
        x: Tensor,
        cache: Optional[Tuple[Tensor, Tensor]],
        cache_limit: int,
        start_pos: int,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """应用带RoPE和缓存的注意力机制."""
        batch_size, seq_len, _ = x.shape
        
        # 生成Q, K, V
        q = self._shape(self.q_proj(x))  # [B, H, T, D_h]
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        
        # 应用RoPE位置编码
        freqs_cis = self._get_freqs(start_pos, seq_len)
        q, k = apply_rotary_emb(
            q.flatten(0, 1), k.flatten(0, 1), freqs_cis
        )
        q = q.view(batch_size, self.heads, seq_len, self.head_dim)
        k = k.view(batch_size, self.heads, seq_len, self.head_dim)
        
        # 合并缓存
        k_attn, v_attn, k_cache, v_cache = self._merge_with_cache(k, v, cache, cache_limit)
        total_len = k_attn.shape[2]
        prev_len = total_len - seq_len
        
        # 计算注意力mask
        attn_mask = self._causal_mask(seq_len, prev_len, total_len, x.device)
        attn_output = F.scaled_dot_product_attention(
            q, k_attn, v_attn,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )
        
        # 重塑输出: [B, H, T, D_h] -> [B, T, D]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().flatten(2)
        return attn_output, (k_cache, v_cache)

    def forward(
        self,
        x: Tensor,
        cache: Optional[Tuple[Tensor, Tensor]],
        cache_limit: int,
        start_pos: int,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """处理序列块（可带注意力缓存）。"""
        # Pre-norm attention with residual connection
        residual = x
        x = self.attn_norm(x)
        attn_output, new_cache = self._apply_attention(x, cache, cache_limit, start_pos)
        x = residual + self.resid_dropout(self.out_proj(attn_output))
        
        # Pre-norm FFN with residual connection
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.resid_dropout(self.ffn(x))
        
        return x, new_cache


TransformerXLState = Tuple[List[Optional[Tuple[Tensor, Tensor]]], int]


class TransformerXL(nn.Module):
    """Lightweight Transformer-XL module with explicit cache management."""

    def __init__(
        self,
        input_size: int,
        *,
        model_dim: int,
        depth: int,
        heads: int,
        ff_multiplier: float,
        dropout: float,
        memory_length: int,
        max_seq_len: int = 10000,
    ) -> None:
        super().__init__()

        if model_dim % heads != 0:
            raise ValueError("model_dim must be divisible by heads")

        self.model_dim = model_dim
        self.memory_length = memory_length

        self.input_proj = nn.Linear(input_size, model_dim)
        ff_dim = int(model_dim * ff_multiplier)
        self.layers = nn.ModuleList(
            [
                TransformerXLEncoderLayer(
                    dim=model_dim,
                    heads=heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = RMSNorm(model_dim)

    def init_state(self) -> TransformerXLState:
        """Create an empty cache state."""
        return [None] * len(self.layers), 0

    def _normalize_state(self, state: Optional[TransformerXLState]) -> TransformerXLState:
        if state is None:
            print("For now the state is none, Please check if it is normal")
            return self.init_state()
        cache_list, abs_pos = state
        if cache_list is None or len(cache_list) != len(self.layers):
            raise ValueError("cache_list is none ot cache_list length is not equal to layers length")
        return cache_list, abs_pos

    def forward(
        self,
        x: Tensor,  # Assumes input is [batch, seq, dim]
        state: Optional[TransformerXLState] = None,
    ) -> Tuple[Tensor [TransformerXLState]]:
        """
        Processes a sequence of observations.
        Assumes input x is already in [batch, seq, dim] format.
        The caller is responsible for permuting/squeezing dimensions.

        Input x: [batch, seq, dim]
        Output: [batch, seq, dim]
        """
        # Input is assumed to be [batch, seq, dim]
        # print(x.shape)
        embeddings = self.input_proj(x)
        seq_len = embeddings.shape[1]

        # Normalize state and process layers
        cache_list, abs_pos = self._normalize_state(state)
        hidden = embeddings
        next_cache: List[Tuple[Tensor, Tensor]] = []

        for layer, layer_cache in zip(self.layers, cache_list):
            hidden, new_cache = layer(hidden, layer_cache, self.memory_length, abs_pos)
            next_cache.append(new_cache)

        # Apply final normalization
        hidden = self.final_norm(hidden)


        cache_len = self.layers[0].freqs_cis.shape[0] if self.layers else 0
        next_pos = (abs_pos + seq_len) % cache_len
        new_state = (next_cache, next_pos)

        cache_shapes = [
            None if layer_cache is None else (layer_cache[0].shape, layer_cache[1].shape)
            for layer_cache in next_cache
        ]
        # print(f"New state cache shapes: {cache_shapes}, next pos: {next_pos}")
        return hidden, new_state
