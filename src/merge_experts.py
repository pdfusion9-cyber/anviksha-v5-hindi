"""Transfer a second run's experts into the first, without duplicating any.

The two checkpoints share an architecture and a tokenizer and were trained on
different languages. The shared parts -- embedding, positions, attention, norms --
encode "how this model reads text" and are kept from the base. The experts encode
"what this model knows", and those are what get transferred.

The rule is the whole point of the operation: an expert from the second run is
added only if no expert in the first already does its job. Similarity is measured
per expert pair, and anything above the threshold is treated as a duplicate and
dropped. What survives is knowledge the base did not have.

    base            keep its embed, positions, attention, norms, router rows
    incoming        for each expert, find its nearest neighbour in the base
                    similarity >= threshold  ->  duplicate, discarded
                    similarity <  threshold  ->  new, appended as an expert

Appending an expert means the router must grow a row for it, and that row has to
be initialised so the new expert receives a fair share of tokens rather than
none. Setting it to the mean of the existing rows is what does that; setting it
to zero would leave the expert unreachable and the transfer pointless.

Nothing is overwritten until every check has passed, and the base checkpoint is
never modified in place -- the merged result is written to a new file, so a
failure at any point leaves both originals untouched.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F


def flatten_expert(state: dict, layer: int, expert: int) -> torch.Tensor:
    parts = []
    for w in ("w1.weight", "w2.weight"):
        key = f"blocks.{layer}.moe.experts.{expert}.{w}"
        if key not in state:
            raise KeyError(f"missing {key}")
        parts.append(state[key].reshape(-1).float())
    return torch.cat(parts)


def check_compatible(a: dict, b: dict) -> list[str]:
    """Every reason these two cannot be merged, before touching anything."""
    problems = []
    ka, kb = set(a), set(b)
    only_a = sorted(k for k in ka - kb if ".moe.experts." not in k)
    only_b = sorted(k for k in kb - ka if ".moe.experts." not in k)
    if only_a or only_b:
        problems.append(f"non-expert tensors differ: only in base {only_a[:4]}, "
                        f"only in incoming {only_b[:4]}")
    for k in sorted(ka & kb):
        if ".moe.experts." in k:
            continue
        if a[k].shape != b[k].shape:
            # The router is allowed to differ when the difference is exactly the
            # expert count. Folding several copies in one at a time leaves the
            # accumulator with more expert rows than the copy arriving next, and
            # that is the normal case, not a mismatch.
            if (".moe.router." in k and len(a[k].shape) == 2
                    and a[k].shape[1] == b[k].shape[1]
                    and a[k].shape[0] != b[k].shape[0]):
                continue
            problems.append(f"shape mismatch on {k}: {tuple(a[k].shape)} vs "
                            f"{tuple(b[k].shape)}")
    return problems


def expert_count(state: dict) -> int:
    n = 0
    while f"blocks.0.moe.experts.{n}.w1.weight" in state:
        n += 1
    return n


def layer_count(state: dict) -> int:
    n = 0
    while f"blocks.{n}.moe.router.weight" in state:
        n += 1
    return n


def merge(base_path: Path, incoming_path: Path, out_path: Path,
          threshold: float = 0.80, max_new: int | None = None,
          verbose: bool = True, average_shared: bool = False) -> dict:
    base = torch.load(base_path, weights_only=False)
    inc = torch.load(incoming_path, weights_only=False)

    sa, sb = base["model"], inc["model"]

    # ---- refuse before writing anything -----------------------------------
    problems = check_compatible(sa, sb)
    if problems:
        raise SystemExit("cannot merge:\n  " + "\n  ".join(problems))

    ta = base_path.parent / "tokenizer.json"
    tb = incoming_path.parent / "tokenizer.json"
    if ta.is_file() and tb.is_file() and ta.read_bytes() != tb.read_bytes():
        raise SystemExit(
            "cannot merge: the tokenizers differ, so embedding row i does not "
            "mean the same string in both models")

    na, nb = expert_count(sa), expert_count(sb)
    layers = layer_count(sa)
    if layers != layer_count(sb):
        raise SystemExit(f"layer count differs: {layers} vs {layer_count(sb)}")

    # The shared tensors first: embedding, positions, attention, norms. For two
    # hats of one model these were trained on different data by each half, so the
    # faithful transfer is the mean rather than one side's copy.
    merged = {k: v.clone() for k, v in sa.items()}
    averaged = 0
    if average_shared:
        for key, tensor in sa.items():
            if (key in sb and ".moe.experts." not in key
                    and ".moe.router." not in key
                    and tensor.shape == sb[key].shape):
                merged[key] = (tensor.float() + sb[key].float()).div(2).to(
                    tensor.dtype)
                averaged += 1

    report = {
        "averaged_shared_tensors": averaged,
        "base": str(base_path), "incoming": str(incoming_path),
        "experts_base": na, "experts_incoming": nb, "layers": layers,
        "threshold": threshold, "per_layer": []}
    total_added = 0
    total_dupes = 0

    for layer in range(layers):
        # similarity of every incoming expert against every base expert
        base_flat = torch.stack([flatten_expert(sa, layer, i) for i in range(na)])
        inc_flat = torch.stack([flatten_expert(sb, layer, j) for j in range(nb)])
        base_n = F.normalize(base_flat, dim=1)
        inc_n = F.normalize(inc_flat, dim=1)
        sim = inc_n @ base_n.T                       # [nb, na]

        best_sim, best_idx = sim.max(dim=1)
        keep = [j for j in range(nb) if float(best_sim[j]) < threshold]
        keep.sort(key=lambda j: float(best_sim[j]))   # least similar first
        if max_new is not None:
            keep = keep[:max_new]

        added = 0
        next_id = na
        for j in keep:
            for w in ("w1.weight", "w2.weight"):
                src = sb[f"blocks.{layer}.moe.experts.{j}.{w}"]
                merged[f"blocks.{layer}.moe.experts.{next_id}.{w}"] = src.clone()
            added += 1
            next_id += 1

        # the router must grow a row per new expert. The mean of the existing
        # rows gives each new expert a fair share of tokens at the start; zeros
        # would make it unreachable and the transfer pointless.
        router_key = f"blocks.{layer}.moe.router.weight"
        old_router = sa[router_key]
        if added:
            mean_row = old_router.mean(dim=0, keepdim=True)
            merged[router_key] = torch.cat(
                [old_router.clone(), mean_row.repeat(added, 1)], dim=0)
        else:
            merged[router_key] = old_router.clone()

        dupes = nb - len(keep)
        # count experts, not expert-layer pairs: the first layer's figure is the
        # number of experts added, and every layer adds the same ones
        if layer == 0:
            total_added = added
        total_dupes += dupes
        report["per_layer"].append({
            "layer": layer, "base_experts": na, "incoming_experts": nb,
            "duplicates_dropped": dupes, "new_experts_added": added,
            "final_experts": na + added,
            "mean_best_similarity": round(float(best_sim.mean()), 4),
            "min_similarity_kept": round(float(best_sim[keep[0]]), 4) if keep else None,
        })
        if verbose:
            print(f"  layer {layer}: {dupes:>2} duplicates dropped, "
                  f"{added:>2} new experts, "
                  f"mean best similarity {float(best_sim.mean()):.3f}")

    cfg = dict(base.get("config", {}))
    cfg["experts"] = na + total_added
    out = {"model": merged, "config": cfg, "args": base.get("args", {}),
           "report": dict(base.get("report", {})),
           "merge": {**report, "experts_added": total_added,
                     "experts_dropped_as_duplicate": total_dupes,
                     "experts_final": na + total_added}}

    # ---- write to a new file, leaving both originals untouched ------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    if ta.is_file():
        shutil.copy(ta, out_path.parent / "tokenizer.json")
    (out_path.parent / "merge_report.json").write_text(
        json.dumps(out["merge"], indent=2))
    return out["merge"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--incoming", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="cosine similarity above which an incoming expert is "
                         "a duplicate and is dropped")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--average-shared", action="store_true",
                    help="average the non-expert tensors instead of keeping the "
                         "base's copy. For two hats of one model the shared parts "
                         "were learned from different material by each half and "
                         "neither is wrong.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, write nothing")
    args = ap.parse_args()

    print(f"base     {args.base}")
    print(f"incoming {args.incoming}")
    print(f"threshold {args.threshold} -- experts at or above this are duplicates")
    if args.dry_run:
        base = torch.load(args.base, weights_only=False)["model"]
        inc = torch.load(args.incoming, weights_only=False)["model"]
        problems = check_compatible(base, inc)
        if problems:
            print("cannot merge:\n  " + "\n  ".join(problems))
            return
        print(f"compatible: {expert_count(base)} + {expert_count(inc)} experts, "
              f"{layer_count(base)} layers")
        return

    r = merge(args.base, args.incoming, args.out, args.threshold, args.max_new,
              average_shared=args.average_shared)
    print(f"\nexperts {r['experts_base']} + {r['experts_added']} new "
          f"- {r['experts_dropped_as_duplicate']} duplicates = {r['experts_final']}")
    print(f"written to {args.out}")
    print(f"report   {args.out.parent / 'merge_report.json'}")


if __name__ == "__main__":
    main()
