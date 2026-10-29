"""V5-shaped transformer with the corrections that matter.

Kept identical to V5 where V5 was right:
    pre-norm blocks, learned positions, 4 heads, GELU feed-forward, tied output head.

Changed where V5 was wrong:
    1. character-level Devanagari vocabulary instead of raw UTF-8 bytes
    2. wider d_model (V5 used 64, which caps capacity hard)
    3. chat template tokens, so user and assistant turns are distinguishable
    4. dropout, because V5 overfit a 12k-record corpus at 60 % next-byte accuracy
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim), nn.Dropout(dropout)
        )
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).split(dim, dim=-1)
        q = q.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        k = k.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        v = v.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        x = x + self.dropout(self.proj(attended))
        return x + self.mlp(self.norm2(x))


class HindiLM(nn.Module):
    def __init__(self, vocab: int, dim: int = 192, layers: int = 4, heads: int = 4,
                 ffn: int = 768, seq_len: int = 192, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, ffn, dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied embeddings
        self.seq_len = seq_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.embed(tokens) + self.position(positions)[None]
        mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))
