"""
MIX😌 GPT - 真正的Decoder-only Transformer模型
基于PyTorch实现，支持自监督预训练（Next Token Prediction）
架构参考: GPT-2 / GPT-3 / LLaMA
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len: int):
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_rope = True
        if self.use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, config.block_size)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = self.rope(q, k, T)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))

        return y


class SwiGLUMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(4 * config.n_embd * 2 / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MixGPT(nn.Module):
    """
    MIX😌 GPT - 真正的Decoder-only Transformer语言模型

    核心特性:
    - Rotary Position Embedding (RoPE) 旋转位置编码
    - SwiGLU激活函数 (LLaMA同款)
    - RMSNorm (替代LayerNorm)
    - 权重共享 (token embedding ↔ output head)
    - 因果自注意力 (Causal Self-Attention)
    - 自监督预训练: Next Token Prediction

    参数规模:
    - tiny:   ~2M   (4层, 64维, 2头)
    - small:  ~10M  (6层, 256维, 4头)
    - medium: ~50M  (8层, 512维, 8头)
    - large:  ~150M (12层, 768维, 12头)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"MIX😌 GPT模型初始化: {n_params/1e6:.2f}M参数 | "
            f"架构:{config.model_size.value} | 词表:{config.vocab_size} | "
            f"上下文:{config.block_size} | 层数:{config.n_layer} | "
            f"维度:{config.n_embd} | 头数:{config.n_head}"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"序列长度{T}超过最大上下文{self.config.block_size}"
        )

        x = self.token_embedding(idx)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty > 1.0:
                for token_id in idx[0].tolist():
                    if token_id < logits.size(-1):
                        logits[0, token_id] /= repetition_penalty

            if top_k is not None and top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k_val)
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        fused_available = device_type == "cuda"
        use_fused = fused_available
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )
        return optimizer

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
