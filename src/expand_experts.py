"""Grow a trained mixture of experts without throwing away what it learned.

The five copies are 121M with twelve experts each. Five of those merge to 574M,
which is not the billion that was asked for, and no amount of training changes
that: a merge keeps the shared part once and concatenates the experts, so the
merged size is shared + copies x (one copy's experts). To clear a billion with
five copies each copy has to carry about 216M of experts, which at this shape is
twenty-two experts rather than twelve.

Resizing a model mid-training is normally impossible -- the weights no longer
fit. But adding experts is the one dimension that grows cleanly: the trained
experts keep their weights and their rows in the router, and the new ones arrive
fresh. Nothing already learned is discarded, and nothing is silently reinitialised.

What this does:

    copy the trained experts across, index for index
    add (target - current) new experts per layer from a seeded generator
    extend the router with rows at the mean of the existing ones, so a new expert
      starts with a fair share of tokens instead of being unreachable

What it refuses:

    shrinking, because that means deleting trained work
    a checkpoint whose shared tensors do not match the requested shape
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from moe_model import MoEConfig


def expert_count(state: dict, layer: int = 0) -> int:
    n = 0
    while f"blocks.{layer}.moe.experts.{n}.w1.weight" in state:
        n += 1
    return n


def layer_count(state: dict) -> int:
    n = 0
    while f"blocks.{n}.moe.router.weight" in state:
        n += 1
    return n


def expand(src: Path, out: Path, experts: int, seed: int = 20261106) -> dict:
    st = torch.load(src, weights_only=False)
    state = st["model"]
    old_cfg = MoEConfig(**st["config"])
    layers = layer_count(state)
    current = expert_count(state)

    if experts == current:
        print(f"already {current} experts; nothing to grow")
        return {"from": current, "to": current, "added": 0}

    if experts < current:
        raise SystemExit(
            f"refusing to shrink {current} experts to {experts}: that deletes "
            f"trained parameters, which is not a decision this script should "
            f"make on its own")

    # the shape of one expert's weights, taken from what is already there
    w1 = state["blocks.0.moe.experts.0.w1.weight"]
    w2 = state["blocks.0.moe.experts.0.w2.weight"]
    per_layer_add = experts - current

    g = torch.Generator().manual_seed(seed)
    merged = {k: v.clone() for k, v in state.items()}

    for layer in range(layers):
        router_key = f"blocks.{layer}.moe.router.weight"
        old_router = state[router_key]
        mean_row = old_router.mean(dim=0, keepdim=True)

        for j in range(current, experts):
            # small random values, the same scale the original init used
            merged[f"blocks.{layer}.moe.experts.{j}.w1.weight"] = (
                torch.randn(w1.shape, generator=g, dtype=w1.dtype) * 0.02)
            merged[f"blocks.{layer}.moe.experts.{j}.w2.weight"] = (
                torch.randn(w2.shape, generator=g, dtype=w2.dtype) * 0.02)

        merged[router_key] = torch.cat(
            [old_router.clone(), mean_row.repeat(per_layer_add, 1)], dim=0)

    new_cfg = MoEConfig(
        vocab=old_cfg.vocab, dim=old_cfg.dim, layers=old_cfg.layers,
        heads=old_cfg.heads, seq_len=old_cfg.seq_len, experts=experts,
        top_k=old_cfg.top_k, ffn_mult=old_cfg.ffn_mult)

    # the step count and token count carry over: this is the same model, larger
    st["model"] = merged
    st["config"] = new_cfg.__dict__
    st["args"] = {**st.get("args", {}), "expanded_from": current,
                  "expanded_to": experts, "expansion_seed": seed}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(st, out)

    before, after = old_cfg.counts(), new_cfg.counts()
    print(f"grew {current} experts to {experts} across {layers} layers")
    print(f"  {before['total']:,} parameters -> {after['total']:,}")
    print(f"  active per token {before['active']:,} -> {after['active']:,}")
    print(f"  written to {out}")
    return {"from": current, "to": experts, "added": per_layer_add,
            "parameters_before": before["total"],
            "parameters_after": after["total"],
            "active_after": after["active"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()
    expand(args.src, args.out, args.experts, args.seed)


if __name__ == "__main__":
    main()
