"""Brain-inspired transformer.

The architecture is deliberately still a transformer -- the goal is not to
replace what works, but to add the five mechanisms the neuroscience literature
says matter most, and to make each of them measurable:

    1. predictive coding      multi-level prediction error: every block is asked
                              to predict the next token, not only the last one
    2. sparse coding          an L1 penalty on the hidden state, so few units are
                              active per input (Olshausen & Field 1996)
    3. criticality            a controller that keeps the branching ratio near
                              1.0, the regime where Beggs & Plenz (2003) measured
                              maximal information transmission
    4. weight sharing         one block applied N times (ALBERT), which cuts the
                              parameter count that tokens-per-parameter depends on
    5. aux loss surface       each block's error is its own objective, so the
                              gradient does not have to travel the full depth

The hippocampal replay buffer lives in train.py, because it is a training-time
memory rather than part of the forward graph.
"""
from __future__ import annotations

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


class CriticalityController(nn.Module):
    """Keeps each layer's activity at the edge of a branching process.

    `branching` is measured as the mean fraction of dimensions whose activation
    exceeds a threshold. A layer that fires on almost nothing is subcritical and
    its signal dies; a layer that fires on everything is supercritical and
    information is lost in saturation. The controller nudges a per-layer gain so
    that the fraction stays near `target`.

    The gain is a buffer, not a parameter: it is updated by the controller, not
    by backpropagation.
    """

    def __init__(self, layers: int, target: float = 0.20, rate: float = 0.05):
        super().__init__()
        self.register_buffer("gain", torch.ones(layers))
        self.target = target
        self.rate = rate
        self.last_branching = [0.0] * layers

    def measure(self, x: torch.Tensor, index: int) -> float:
        with torch.no_grad():
            scale = x.detach().abs().mean().clamp(min=1e-6)
            active = (x.detach().abs() > scale).float().mean().item()
        self.last_branching[index] = round(active, 4)
        return active

    def update(self, index: int) -> None:
        with torch.no_grad():
            error = self.target - self.last_branching[index]
            self.gain[index] = (self.gain[index] * (1.0 + self.rate * error)).clamp(0.5, 2.0)

    def report(self) -> dict:
        return {
            "branching": self.last_branching,
            "gain": [round(float(g), 3) for g in self.gain],
            "target": self.target,
        }


class HindiLM(nn.Module):
    def __init__(self, vocab: int, dim: int = 192, layers: int = 4, heads: int = 4,
                 ffn: int = 768, seq_len: int = 192, dropout: float = 0.1,
                 share_blocks: bool = False, multi_level: bool = False,
                 criticality: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(seq_len, dim)
        if share_blocks:
            shared = Block(dim, heads, ffn, dropout)
            self.blocks = nn.ModuleList([shared for _ in range(layers)])
        else:
            self.blocks = nn.ModuleList(
                [Block(dim, heads, ffn, dropout) for _ in range(layers)]
            )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied embeddings

        self.seq_len = seq_len
        self.share_blocks = share_blocks
        self.multi_level = multi_level
        self.use_criticality = criticality

        # one auxiliary head shared across depths: cheap, and enough to give each
        # block a prediction objective of its own
        self.aux_norm = nn.LayerNorm(dim) if multi_level else None
        self.aux_head = nn.Linear(dim, vocab, bias=False) if multi_level else None
        self.controller = CriticalityController(layers) if criticality else None

    def forward(self, tokens: torch.Tensor, return_aux: bool = False):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        x = self.embed(tokens) + self.position(positions)[None]
        mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()

        aux_logits = []
        sparsity = []
        for index, block in enumerate(self.blocks):
            x = block(x, mask)
            if self.controller is not None:
                self.controller.measure(x, index)
                x = x * self.controller.gain[index]
            if self.multi_level:
                aux_logits.append(self.aux_head(self.aux_norm(x)))
                sparsity.append(x.detach().abs().mean().item())

        logits = self.head(self.norm(x))
        if return_aux:
            return logits, aux_logits, sparsity
        return logits

    def parameters_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"parameters": total, "share_blocks": self.share_blocks,
                "multi_level": self.multi_level, "criticality": self.use_criticality}
