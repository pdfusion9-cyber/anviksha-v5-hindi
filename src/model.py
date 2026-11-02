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
                 criticality: bool = False, cross_links: bool = False,
                 recurrent: int = 1, elements: int = 0,
                 adapters: int = 0, drift: int = 0, distinct: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(seq_len, dim)
        # Compositional elements. Instead of one block per layer, keep a small
        # set of K shared blocks (the elements) and give each layer its own
        # mixing weights over them. Layer l computes sum_k a[l,k] * element_k(x).
        # The stored parameters are K blocks plus layers*K numbers, but the
        # number of distinct effective layers is combinatorial -- each layer is
        # its own combination, so new behaviour is produced from reused parts
        # rather than from new parameters.
        # A shared fast path with a small distinct correction per layer.
        # The shared block carries the work -- one block-pass per layer, which is
        # what makes it fast -- and each layer adds a low-rank correction of its
        # own, so the layers are still not identical. This is the trade the
        # element mixing got wrong: it spent compute to save parameters, when
        # what we are short of is compute.
        self.adapters = int(adapters)
        if self.adapters > 0:
            self.down = nn.Parameter(torch.randn(layers, dim, self.adapters) * 0.02)
            self.up = nn.Parameter(torch.zeros(layers, self.adapters, dim))
            self.adapter_scale = nn.Parameter(torch.ones(layers) * 0.1)
        else:
            self.down = self.up = self.adapter_scale = None

        # Index-less drift. There is no per-layer weight tensor at all: one shared
        # block, and each layer scales its own input by a factor generated from a
        # few shared waves. The factor is a continuous function of depth,
        #    g(l) = 1 + sum_j a_j * sin(w_j * l + p_j),
        # so every layer is on the same curve and reads the same waves -- which is
        # the "instant communication": there is nothing per-layer to exchange,
        # the waves are shared by construction. Scaling the input before a shared
        # matmul costs O(B*T*dim) instead of a second matmul, so this is as fast
        # as one shared block while the layers are no longer identical.
        self.drift = int(drift)
        if self.drift > 0:
            self.drift_amp = nn.Parameter(torch.randn(self.drift) * 0.05)
            self.drift_freq = nn.Parameter(torch.linspace(0.5, 2.5, self.drift))
            self.drift_phase = nn.Parameter(torch.rand(self.drift) * 6.283)
        else:
            self.drift_amp = self.drift_freq = self.drift_phase = None

        self.elements = int(elements)
        if self.elements > 0:
            self.blocks = nn.ModuleList(
                [Block(dim, heads, ffn, dropout) for _ in range(self.elements)]
            )
            self.mix = nn.Parameter(torch.zeros(layers, self.elements))
            # start near-uniform so no element dominates before training
            with torch.no_grad():
                self.mix.fill_(1.0 / self.elements)
        elif distinct:
            # a separate block per layer, and drift may still add a free gain
            self.blocks = nn.ModuleList(
                [Block(dim, heads, ffn, dropout) for _ in range(layers)]
            )
        elif share_blocks or self.adapters > 0 or self.drift > 0:
            # adapters ride on one shared block: the block does the work, the
            # per-layer correction supplies the difference
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
        self.cross_links = cross_links
        # recurrent depth: apply the block stack N times, told apart only by a
        # small step embedding. Costs recurrent x dim parameters (a few thousand)
        # and buys effective depth that would otherwise cost a whole block.
        self.recurrent = max(1, int(recurrent))
        self.step_embed = (nn.Embedding(self.recurrent, dim)
                           if self.recurrent > 1 else None)
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
        # running average of every earlier layer output: each block sees a summary
        # of the whole depth below it, not only the residual that reaches it. No
        # new parameters -- this is the cross-link.
        memory = torch.zeros_like(x)
        seen = 0
        for step in range(self.recurrent):
            if self.step_embed is not None:
                x = x + self.step_embed.weight[step][None, None, :]
            layer_count = self.mix.shape[0] if self.elements > 0 else len(self.blocks)
            for index in range(layer_count):
                if self.drift > 0 or self.adapters > 0:
                    # both may be on at once: the drift gain is nearly free, the
                    # adapter carries the stronger per-layer difference
                    gain = 1.0
                    if self.drift > 0:
                        depth = torch.tensor(float(index), dtype=x.dtype, device=x.device)
                        wave = torch.sin(self.drift_freq * depth + self.drift_phase)
                        gain = 1.0 + (self.drift_amp * wave).sum()
                    scaled = x * gain
                    # distinct blocks keep their own weights; drift still gives
                    # each of them its own gain, which costs nothing
                    block = self.blocks[0] if len(self.blocks) == 1 else self.blocks[index]
                    out = block(scaled, mask)
                    if self.adapters > 0:
                        corr = (scaled @ self.down[index]) @ self.up[index]
                        x = out + self.adapter_scale[index] * corr
                    else:
                        x = out
                elif self.elements > 0:
                    # every layer is its own weighted combination of the shared
                    # elements; the weights are learned per layer, so six layers
                    # over three elements give six different effective blocks
                    weights = torch.softmax(self.mix[index], dim=0)
                    mixed = None
                    for k, element in enumerate(self.blocks):
                        term = weights[k] * element(x, mask)
                        mixed = term if mixed is None else mixed + term
                    x = mixed
                else:
                    block = self.blocks[index]
                    if self.cross_links:
                        x = block(x + memory, mask)
                        seen += 1
                        memory = memory + (x - memory) / seen
                    else:
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
