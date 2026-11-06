"""Measure whether experts have actually diverged, before trusting a merge.

The five-copy merge produced 60 experts and no duplicates dropped, which sounds
like success and is not evidence of anything. The copies all started from seed
20261106 and trained briefly, so their experts were 95% alike; at threshold 0.95
the merge discarded 47 of 60 and at 0.999 it discarded none. Both numbers came
from the same checkpoints. The threshold did not change the models, only the
question that was asked about them.

So the question a merge needs answered first is not "are these duplicates" but
"have these diverged at all". This measures that directly:

    within-copy    how unlike each other the experts of one copy are
    across-copy    how unlike the same expert is in two different copies
    ratio          across-copy similarity divided by within-copy similarity

A ratio near 1.0 means the copies are interchangeable and merging them adds
parameters without adding knowledge. A ratio well below 1.0 means the copies
learned different things and the merge is worth doing.

The distinction matters because cosine similarity between two randomly
initialised experts is already around 0.9 for the shapes we use. An absolute
threshold cannot tell "trained to be similar" from "never trained apart". The
ratio can, because the within-copy figure is the baseline that training has to
beat.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load(path: Path) -> dict:
    state = torch.load(path, weights_only=False)
    return state["model"] if "model" in state else state


def layer_experts(state: dict, layer: int) -> torch.Tensor:
    """Every expert in one layer, flattened to a row each."""
    rows = []
    i = 0
    while f"blocks.{layer}.moe.experts.{i}.w1.weight" in state:
        parts = [state[f"blocks.{layer}.moe.experts.{i}.{w}"].reshape(-1).float()
                 for w in ("w1.weight", "w2.weight")]
        rows.append(torch.cat(parts))
        i += 1
    return torch.stack(rows) if rows else torch.empty(0)


def layers_of(state: dict) -> int:
    n = 0
    while f"blocks.{n}.moe.router.weight" in state:
        n += 1
    return n


def mean_offdiag_similarity(rows: torch.Tensor) -> float:
    """How alike a set of vectors is, ignoring each vector's match with itself."""
    if rows.size(0) < 2:
        return 1.0
    n = F.normalize(rows, dim=1)
    sim = n @ n.T
    k = sim.size(0)
    off = sim.sum() - sim.diagonal().sum()
    return float(off / (k * (k - 1)))


def movement_from(a: torch.Tensor, init: torch.Tensor) -> float:
    """How far a set of experts has travelled from its starting point."""
    if a.size(0) != init.size(0):
        return float("nan")
    an, iN = F.normalize(a, dim=1), F.normalize(init, dim=1)
    return float(1.0 - (an * iN).sum(dim=1).mean())


def compare(a_path: Path, b_path: Path, init_path: Path | None = None) -> dict:
    a, b = load(a_path), load(b_path)
    init = load(init_path) if init_path else None
    layers = min(layers_of(a), layers_of(b))
    if layers == 0:
        raise SystemExit("no mixture-of-experts layers found in these checkpoints")

    per_layer = []
    for layer in range(layers):
        ea, eb = layer_experts(a, layer), layer_experts(b, layer)
        if ea.size(0) == 0 or eb.size(0) == 0:
            continue
        if ea.size(0) != eb.size(0):
            raise SystemExit(
                f"layer {layer}: {ea.size(0)} experts against {eb.size(0)}. "
                f"Compare copies that have not been folded yet.")

        an, bn = F.normalize(ea, dim=1), F.normalize(eb, dim=1)
        across = float((an * bn).sum(dim=1).mean())
        row = {"layer": layer, "across": round(across, 4)}
        if init is not None:
            ei = layer_experts(init, layer)
            if ei.size(0) == ea.size(0):
                row["moved_a"] = round(movement_from(ea, ei), 4)
                row["moved_b"] = round(movement_from(eb, ei), 4)
        per_layer.append(row)

    across = [p["across"] for p in per_layer]
    mean_across = sum(across) / len(across)
    moved = [p.get("moved_a", 0) for p in per_layer] + \
            [p.get("moved_b", 0) for p in per_layer]
    mean_moved = sum(moved) / len(moved) if moved else None

    # The actionable test is movement, not similarity. Two copies that both
    # barely moved from their shared starting point have not diverged, however
    # their mutual similarity reads.
    if mean_moved is not None and mean_moved < 0.02:
        verdict = ("barely trained: the experts have moved "
                   f"{mean_moved*100:.2f}% from where they started, so there is "
                   "nothing distinct to transfer yet")
    elif mean_moved is not None and mean_moved < 0.10:
        verdict = ("lightly trained: experts have moved "
                   f"{mean_moved*100:.1f}% from init. A merge would carry some "
                   "difference, but more training would carry more")
    elif mean_across > 0.99:
        verdict = ("trained but converged: both copies moved, and landed in the "
                   "same place, so merging adds parameters rather than knowledge")
    else:
        verdict = ("diverged: both copies moved from init and landed apart; the "
                   "merge is worth doing")

    return {
        "a": str(a_path), "b": str(b_path),
        "init": str(init_path) if init_path else None,
        "layers": layers,
        "mean_across_similarity": round(mean_across, 4),
        "mean_movement_from_init": (round(mean_moved, 4)
                                    if mean_moved is not None else None),
        "verdict": verdict,
        "per_layer": per_layer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None,
                    help="the checkpoint both copies started from. Without it "
                         "there is no way to tell 'trained to be similar' from "
                         "'never trained apart'")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    r = compare(args.a, args.b, args.init)
    print(f"  across-copy similarity   {r['mean_across_similarity']:.4f}")
    if r["mean_movement_from_init"] is not None:
        print(f"  movement from init       {r['mean_movement_from_init']:.4f}")
    print(f"  verdict                  {r['verdict']}")
    print(f"\n  {'layer':>5} {'across':>8} {'moved A':>9} {'moved B':>9}")
    for p in r["per_layer"]:
        print(f"  {p['layer']:>5} {p['across']:>8.4f} "
              f"{p.get('moved_a', float('nan')):>9.4f} "
              f"{p.get('moved_b', float('nan')):>9.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, indent=2))
        print(f"\nwritten to {args.out}")

    if r["mean_movement_from_init"] is not None and \
            r["mean_movement_from_init"] < 0.02:
        print("\n  Do not merge these yet. Train longer first: the experts are")
        print("  still where they started, so the extra experts would carry no")
        print("  information the model did not already have.")


if __name__ == "__main__":
    main()
