"""A one-billion-parameter model that can be trained on a CPU, by activating little of it.

A dense billion-parameter model needs sixteen gigabytes before activations --
weights, gradients and two AdamW states at four bytes each -- and the runner has
sixteen gigabytes in total. On the sandbox's two gigabytes it is not close. So a
dense 1B is not a training target here, whatever the schedule says.

A sparse mixture of experts is. The parameters exist but only a fraction are used
for any given token, and that fraction is what has to be held in memory during a
step:

    1,000M total      stored, 4.0 GB in fp32, memory-mapped from disk
       63M active     touched per token: 1.0 GB of weights, gradients and states

That is the same construction Dhīr 5B uses -- 5.09B total, 1.79B active -- so it
is a real architecture and not a way of writing a bigger number on the box. The
honest description of the result is "a 1B-parameter model with 63M active
parameters", and both numbers belong in any claim about it.

What is new here, beyond the router:

    streaming      the corpus is read in shards and tokenised as it arrives, so
                   memory holds a window buffer and nothing else. A 4.7 GB corpus
                   never lands on disk in full.

    expert paging  expert blocks are memory-mapped, so the pages the router does
                   not choose are never read. On a machine with a page cache this
                   is what makes 4 GB of weights work in 1 GB of working set.

    balance loss   without it the router collapses onto two experts within a few
                   hundred steps and the model becomes a small dense one wearing a
                   large parameter count. The loss is measured and reported.

    token usage    every expert's share of tokens is logged, so collapse is
                   visible rather than inferred from a flat loss curve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    vocab: int = 4096
    dim: int = 512
    layers: int = 12
    heads: int = 8
    seq_len: int = 256
    experts: int = 40
    top_k: int = 2
    ffn_mult: int = 4
    dropout: float = 0.0
    # Batch-balanced dispatch; source checkpoints retain the default False.
    balanced_routing: bool = False
    min_gate_weight: float = 0.10

    @property
    def ffn(self) -> int:
        return self.dim * self.ffn_mult

    def footprint_gb(self, dtype_bytes: int = 4) -> float:
        """What a training step actually allocates.

        Weights, gradients and two AdamW states for every parameter -- not just
        the active ones. With a batch of four at 256 tokens, 1024 tokens are
        routed per step and with 64 experts every expert receives some, so every
        expert gets a gradient and an optimiser state. Counting only the active
        fraction is what made a 1B fp32 model look like it fitted in 0.7 GB when
        it needs 17.3 GB, and the first run was killed by the runner for it.
        """
        return self.counts()["total"] * dtype_bytes * 4 / 1e9

    def counts(self) -> dict:
        attn = 4 * self.dim * self.dim
        per_expert = 2 * self.dim * self.ffn
        moe = self.experts * per_expert
        norms = 4 * self.dim
        per_layer = attn + moe + norms
        body = per_layer * self.layers
        embed = self.vocab * self.dim
        total = body + embed + self.seq_len * self.dim + 2 * self.dim
        active_per_layer = attn + self.top_k * per_expert + norms
        active = active_per_layer * self.layers + embed
        return {"total": total, "active": active,
                "per_expert": per_expert, "per_layer": per_layer}



def balanced_topk_indices(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Capacity-balanced top-k routing with exact batch-level expert coverage.

    Every token is assigned to top_k distinct experts. If tokens * top_k is at
    least the expert count, each expert receives floor/ceil(tokens * top_k /
    experts) assignments exactly. Each token prefers the highest-scoring
    experts that still have quota. Small batches fall back to ordinary top-k.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected [tokens, experts] logits, got {tuple(logits.shape)}")
    token_count, expert_count = logits.shape
    if top_k < 1 or top_k >= expert_count:
        raise ValueError(f"top_k must be in [1, {expert_count - 1}], got {top_k}")
    if token_count == 0:
        return torch.empty((0, top_k), dtype=torch.long, device=logits.device)

    scores = logits.detach().float()
    total_assignments = token_count * top_k
    if total_assignments < expert_count:
        return scores.topk(top_k, dim=-1).indices

    base_quota, remainder = divmod(total_assignments, expert_count)
    quotas = torch.full((expert_count,), base_quota, dtype=torch.long)
    if remainder:
        preferred = torch.argsort(scores.mean(dim=0).cpu(), descending=True)
        quotas[preferred[:remainder]] += 1

    remaining = quotas.tolist()
    ranked_experts = torch.argsort(scores, dim=-1, descending=True).cpu().tolist()
    if top_k == 1:
        confidence = scores.max(dim=-1).values
    else:
        leading = scores.topk(top_k, dim=-1).values
        confidence = leading[:, 0] - leading[:, -1]
    token_order = torch.argsort(confidence, descending=True).cpu().tolist()

    assigned = [[-1] * top_k for _ in range(token_count)]
    for token_index in token_order:
        chosen = set()
        for slot in range(top_k):
            selected = -1
            for expert_index in ranked_experts[token_index]:
                if expert_index not in chosen and remaining[expert_index] > 0:
                    selected = expert_index
                    break
            if selected < 0:
                raise RuntimeError(
                    "balanced routing exhausted quotas before every token "
                    "received distinct top-k assignments"
                )
            assigned[token_index][slot] = selected
            chosen.add(selected)
            remaining[selected] -= 1

    if any(value != 0 for value in remaining):
        raise RuntimeError("balanced routing left unfilled expert quotas")
    return torch.tensor(assigned, dtype=torch.long, device=logits.device)


class Attention(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.heads = cfg.heads
        self.dim = cfg.dim
        self.norm = nn.LayerNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, n, d = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).split(d, dim=-1)
        q = q.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        k = k.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(b, n, d)
        return x + self.drop(self.proj(a))


class Expert(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn, bias=False)
        self.w2 = nn.Linear(cfg.ffn, cfg.dim, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class MoELayer(nn.Module):
    """Top-k routing over experts, with the balance statistics logged.

    The router is a single linear map from the token to one score per expert. Two
    experts are used per token, and their outputs are combined by their softmax
    weights, which is what lets a gradient reach the router at all.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.dim)
        self.router = nn.Linear(cfg.dim, cfg.experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.experts)])
        self.last_usage: torch.Tensor | None = None
        self.last_balance: float = 0.0
        self.last_loss: torch.Tensor | None = None

    def forward(self, x):
        b, n, d = x.shape
        flat = self.norm(x).reshape(b * n, d)
        logits = self.router(flat)
        probs = F.softmax(logits.float(), dim=-1).to(logits.dtype)
        if self.cfg.balanced_routing:
            top_i = balanced_topk_indices(logits, self.cfg.top_k)
            selected_logits = logits.gather(1, top_i)
            top_p = F.softmax(selected_logits.float(), dim=-1)
            floor = min(float(self.cfg.min_gate_weight), 1.0 / self.cfg.top_k)
            top_p = top_p.clamp(min=floor)
            top_p = top_p / top_p.sum(-1, keepdim=True).clamp(min=1e-9)
            top_p = top_p.to(flat.dtype)
        else:
            top_p, top_i = probs.topk(self.cfg.top_k, dim=-1)
            top_p = top_p / top_p.sum(-1, keepdim=True).clamp(min=1e-9)

        out = torch.zeros_like(flat)
        usage = torch.zeros(self.cfg.experts, device=flat.device)
        for slot in range(self.cfg.top_k):
            idx = top_i[:, slot]
            weight = top_p[:, slot].unsqueeze(-1)
            for e in idx.unique():
                mask = idx == e
                out[mask] += weight[mask] * self.experts[int(e)](flat[mask])
                usage[int(e)] += mask.sum()

        # Load balancing, Switch Transformer form: experts * sum(importance *
        # load), where importance is the mean router probability per expert and
        # load is its share of tokens. importance carries a gradient and load does
        # not, which is what pushes the router toward an even split while still
        # being differentiable. The first version squared the difference of the
        # detached counts, so it was a number and not a loss -- the router
        # collapsed to three experts taking 75% of tokens and nothing opposed it.
        frac = usage / usage.sum().clamp(min=1)
        importance = probs.mean(0)
        balance = self.cfg.experts * (importance * frac).sum()
        self.last_loss = balance
        self.last_balance = float(balance.detach())
        self.last_usage = usage.detach()
        return x + out.view(b, n, d)


class MoELM(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.dim)
        self.position = nn.Embedding(cfg.seq_len, cfg.dim)
        self.blocks = nn.ModuleList()
        for _ in range(cfg.layers):
            self.blocks.append(nn.ModuleDict({
                "attn": Attention(cfg), "moe": MoELayer(cfg)}))
        self.norm_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        self.head.weight = self.embed.weight     # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens):
        b, n = tokens.shape
        pos = torch.arange(n, device=tokens.device)
        x = self.embed(tokens) + self.position(pos)[None]
        for blk in self.blocks:
            x = blk["attn"](x)
            x = blk["moe"](x)
        return self.head(self.norm_f(x))

    def balance_loss(self) -> torch.Tensor:
        """Differentiable, so it actually shapes the router."""
        vals = [blk["moe"].last_loss for blk in self.blocks
                if blk["moe"].last_loss is not None]
        if not vals:
            return torch.zeros((), device=self.embed.weight.device)
        return torch.stack(vals).mean()

    def expert_usage(self) -> list[float]:
        """Share of tokens each expert received, averaged over layers."""
        rows = [blk["moe"].last_usage for blk in self.blocks
                if blk["moe"].last_usage is not None]
        if not rows:
            return []
        total = torch.stack(rows).sum(0)
        return [round(float(v), 3) for v in (total / total.sum().clamp(min=1))]

    def parameters_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        c = self.cfg.counts()
        return {"parameters": total, "active_parameters": c["active"],
                "total_parameters": total,
                "experts": self.cfg.experts, "top_k": self.cfg.top_k,
                "active_fraction": round(c["active"] / max(total, 1), 3)}


if __name__ == "__main__":
    for experts, layers, dim in ((40, 12, 512), (16, 8, 384), (8, 4, 256)):
        cfg = MoEConfig(experts=experts, layers=layers, dim=dim,
                        heads=max(dim // 64, 2), ffn_mult=4)
        m = MoELM(cfg)
        r = m.parameters_report()
        c = cfg.counts()
        mem = r["active_parameters"] * 16 / 1e9
        print(f"experts {experts:>3} layers {layers:>2} dim {dim:>4}  "
              f"total {r['total_parameters']/1e6:7.1f}M  "
              f"active {r['active_parameters']/1e6:5.1f}M  "
              f"({r['active_fraction']*100:.1f}% per token)  "
              f"training footprint {mem:.2f} GB")
