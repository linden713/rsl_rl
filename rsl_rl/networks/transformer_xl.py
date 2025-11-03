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

def apply_rotary_single(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_cis = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) * freqs_cis
    return torch.view_as_real(x_cis).flatten(-2).type_as(x)
def init_weights(module, depth=None):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        # 残差缩放
        if depth is not None and module.out_features == module.in_features:
            with torch.no_grad():
                module.weight.mul_(1.0 / (depth ** 0.5))

def init_transformer(model, depth):
    nn.init.xavier_uniform_(model.input_proj.weight)
    nn.init.zeros_(model.input_proj.bias)
    for layer in model.layers:
        layer.apply(lambda m: init_weights(m, depth))

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

    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float, max_seq_len: int, rotary_base: float = 10000.0, *, pos_bias_max: float = 2.0, norm_type: str = "rms") -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("embedding dimension must be divisible by number of heads.")

        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim % 2 != 0:
            raise ValueError("head dimension must be even to use rotary embeddings.")

        if norm_type.lower() == "layernorm":
            self.attn_norm = nn.LayerNorm(dim, eps=1e-5)
            self.ffn_norm = nn.LayerNorm(dim, eps=1e-5)
        else:
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
        # 最大相对位置偏置幅度（控制整个可见范围内的最大logit偏置跨度）。
        # 设为0可关闭该偏置；默认约2.0，量级与缩放后的QK logits相当。
        self.pos_bias_max = float(pos_bias_max)

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
        # Cache and return RAW (unrotated) K/V tensors
        if cache is None:
            k_attn_raw, v_attn_raw = new_k, new_v  # [batch, heads, seq, head_dim]
        else:
            k_cached, v_cached = cache
            batch_size = new_k.size(0)
            if k_cached.size(0) != batch_size:
                raise ValueError(
                    f"Cache batch dimension {k_cached.size(0)} does not match new batch {batch_size}"
                )
            k_attn_raw = torch.cat([k_cached, new_k], dim=2)  # [batch, heads, prev + seq, head_dim]
            v_attn_raw = torch.cat([v_cached, new_v], dim=2)  # [batch, heads, prev + seq, head_dim]

        if cache_limit > 0:
            k_attn_raw = k_attn_raw[:, :, -cache_limit:, :]
            v_attn_raw = v_attn_raw[:, :, -cache_limit:, :]

        # Return attention tensors (raw) and updated raw cache
        return k_attn_raw, v_attn_raw, k_attn_raw.detach(), v_attn_raw.detach()

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
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """应用带RoPE和缓存的注意力机制."""
        batch_size, seq_len, _ = x.shape
        
        # 生成Q, K, V (raw)
        q_raw = self._shape(self.q_proj(x))  # [B, H, T, D_h]
        k_raw = self._shape(self.k_proj(x))
        v_raw = self._shape(self.v_proj(x))

        # 合并缓存（保持为未旋转的原始K/V）
        if cache_limit != 0:
            k_attn_raw, v_attn_raw, k_cache_raw, v_cache_raw = self._merge_with_cache(k_raw, v_raw, cache, cache_limit)
        else:
            k_attn_raw, v_attn_raw, k_cache_raw, v_cache_raw = k_raw, v_raw, None, None

        total_len = k_attn_raw.shape[2]
        prev_len = total_len - seq_len

        # 计算Q的RoPE（位置区间：[prev_len, prev_len + seq_len)）
        q_freqs = self._get_freqs(prev_len, seq_len)
        q = apply_rotary_single(q_raw.flatten(0, 1), q_freqs)
        q = q.view(batch_size, self.heads, seq_len, self.head_dim)

        # 计算K的RoPE（合并后整体旋转，位置区间：[prev_len - prev_len, prev_len + seq_len) = [0, total_len)）
        k_freqs = self._get_freqs(prev_len - prev_len, total_len)
        k = apply_rotary_single(k_attn_raw.flatten(0, 1), k_freqs)
        k = k.view(batch_size, self.heads, total_len, self.head_dim)
        v_attn = v_attn_raw  # V 不进行旋转
        
        # 计算注意力mask与相对位置偏置
        # 注：SDPA内部已做1/sqrt(d)缩放，典型QK对数值量级约O(1)。
        # 因此这里将总跨度限制在pos_bias_max，使其与原始logits同量级。
        if self.pos_bias_max > 0.0:
            # 相对位置：对每个query i，bias ~ slope * (j - i_abs)，越接近i越大（更不负）。
            # 这样在因果约束下自然形成“近期偏好”。
            slope = self.pos_bias_max / max(total_len - 1, 1)
            j_idx = torch.arange(total_len, device=x.device).view(1, -1).float()  # [1, total_len]
            i_abs = (prev_len + torch.arange(seq_len, device=x.device).view(-1, 1)).float()  # [seq_len, 1]
            attn_mask = slope * (j_idx - i_abs)  # [seq_len, total_len]
        else:
            attn_mask = torch.zeros(seq_len, total_len, device=x.device)

        # 叠加因果遮罩（未来位置置为 -inf）
        causal = self._causal_mask(seq_len, prev_len, total_len, x.device)              # bool or None
        if causal is not None:
            attn_mask = attn_mask.masked_fill(causal, float("-inf"))   

        attn_output = F.scaled_dot_product_attention(
            q, k, v_attn,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )
        
        # 重塑输出: [B, H, T, D_h] -> [B, T, D]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().flatten(2)
        return attn_output, (k_cache_raw, v_cache_raw)

    def forward(
        self,
        x: Tensor,
        cache: Optional[Tuple[Tensor, Tensor]],
        cache_limit: int,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """处理序列块（可带注意力缓存）。"""
        # Pre-norm attention with residual connection
        residual = x
        x = self.attn_norm(x)
        attn_output, new_cache = self._apply_attention(x, cache, cache_limit)
        x = residual + self.resid_dropout(self.out_proj(attn_output))
        
        # Pre-norm FFN with residual connection
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.resid_dropout(self.ffn(x))
        
        return x, new_cache


TransformerXLState = List[Optional[Tuple[Tensor, Tensor]]]


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
        pos_bias_max: float = 2.0,
        norm_type: str = "rms",
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
                    pos_bias_max=pos_bias_max,
                    norm_type=norm_type,
                )
                for _ in range(depth)
            ]
        )
        if norm_type.lower() == "layernorm":
            self.final_norm = nn.LayerNorm(model_dim, eps=1e-5)
        else:
            self.final_norm = RMSNorm(model_dim)
        init_transformer(self, depth)


    def init_state(self) -> TransformerXLState:
        """Create an empty cache state (per-layer raw KV)."""
        return [None] * len(self.layers)

    def _normalize_state(self, state: Optional[TransformerXLState]) -> TransformerXLState:
        if state is None:
            return self.init_state()
        cache_list = state
        if cache_list is None or len(cache_list) != len(self.layers):
            raise ValueError("cache_list is none ot cache_list length is not equal to layers length")
        return cache_list

    def forward(
        self,
        x: Tensor,  # Assumes input is [batch, seq, dim]
        state: Optional[TransformerXLState] = None,
    ) -> Tuple[Tensor, TransformerXLState]:
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

        # Normalize state and process layers
        cache_list = self._normalize_state(state)
        hidden = embeddings
        next_cache: List[Tuple[Tensor, Tensor]] = []

        for layer, layer_cache in zip(self.layers, cache_list):
            hidden, new_cache = layer(hidden, layer_cache, self.memory_length)
            next_cache.append(new_cache)

        # Apply final normalization
        hidden = self.final_norm(hidden)
        new_state = next_cache

        # cache_shapes = [
        #     None if layer_cache is None else (layer_cache[0].shape, layer_cache[1].shape)
        #     for layer_cache in next_cache
        # ]
        # print(f"New state cache shapes: {cache_shapes}")
        return hidden, new_state
