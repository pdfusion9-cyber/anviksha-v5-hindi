from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from merge_experts import (
    check_compatible, expert_count, expert_counts, flatten_expert, layer_count,
)
from moe_model import MoEConfig


SOURCE_RUNS = {
    0: 42275510662,
    1: 42273920152,
    2: 42271805121,
    3: 42267619625,
    4: 42272958418,
}
THRESHOLD = 0.999


def only_file(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"{root}: expected one {name}, found {len(matches)}")
    return matches[0]


def load_checkpoint(path: Path):
    # mmap avoids eagerly copying unused optimizer/storage pages into RAM.
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def fold(base: dict, incoming: dict, merged: dict, prior_copies: int,
         source_copy: int, report: dict) -> None:
    sa, sb = base["model"], incoming["model"]
    problems = check_compatible(sa, sb)
    if problems:
        raise RuntimeError("incompatible checkpoint: " + "; ".join(problems))


    la, lb = layer_count(sa), layer_count(sb)
    if la != lb:
        raise RuntimeError(f"layer count mismatch: {la} vs {lb}")

    per_layer = []
    total_added = 0
    for layer in range(la):
        na = expert_count(merged, layer)
        nb = expert_count(sb, layer)
        if nb != 22:
            raise RuntimeError(f"copy {source_copy}, layer {layer}: expected 22 source experts, got {nb}")

        base_flat = torch.stack([flatten_expert(merged, layer, i) for i in range(na)])
        inc_flat = torch.stack([flatten_expert(sb, layer, j) for j in range(nb)])
        similarities = F.normalize(inc_flat, dim=1) @ F.normalize(base_flat, dim=1).T
        best_sim, _ = similarities.max(dim=1)
        keep = [j for j in range(nb) if float(best_sim[j]) < THRESHOLD]
        keep.sort(key=lambda j: float(best_sim[j]))

        # A source-specific router is only meaningful when every corresponding
        # source expert is kept. Refuse rather than silently misaligning rows.
        if len(keep) != nb:
            raise RuntimeError(
                f"copy {source_copy}, layer {layer}: {nb-len(keep)} experts "
                f"would be deduplicated; router row alignment requires review"
            )

        for new_index, source_index in enumerate(keep, start=na):
            for weight in ("w1.weight", "w2.weight"):
                source_key = f"blocks.{layer}.moe.experts.{source_index}.{weight}"
                target_key = f"blocks.{layer}.moe.experts.{new_index}.{weight}"
                merged[target_key] = sb[source_key]

        router_key = f"blocks.{layer}.moe.router.weight"
        # Critical repair: preserve the router row trained with each incoming
        # expert; do not synthesize repeated copies of the old router mean.
        incoming_router = sb[router_key]
        merged[router_key] = torch.cat(
            [merged[router_key], incoming_router[keep]], dim=0
        )

        per_layer.append({
            "layer": layer,
            "base_experts_before_copy": na,
            "incoming_experts": nb,
            "experts_added": len(keep),
            "experts_dropped": nb - len(keep),
            "mean_best_cosine_similarity": round(float(best_sim.mean()), 6),
            "min_best_cosine_similarity": round(float(best_sim.min()), 6),
            "max_best_cosine_similarity": round(float(best_sim.max()), 6),
            "router_rows_preserved_from_source": len(keep),
        })
        total_added += len(keep)

    # Correct online arithmetic mean across all source checkpoints.
    # prior_copies is the number already represented in merged shared tensors.
    averaged = 0
    for key, old_tensor in list(merged.items()):
        if ".moe.experts." in key or ".moe.router." in key:
            continue
        if key not in sb:
            continue
        if tuple(old_tensor.shape) != tuple(sb[key].shape):
            raise RuntimeError(f"shared tensor shape mismatch: {key}")
        if old_tensor.is_floating_point():
            merged[key] = (
                old_tensor.float().mul(prior_copies).add(sb[key].float())
                .div(prior_copies + 1).to(dtype=old_tensor.dtype)
            )
        else:
            if not torch.equal(old_tensor, sb[key]):
                raise RuntimeError(f"non-floating shared tensor differs: {key}")
        averaged += 1

    report["copies"].append({
        "copy": source_copy,
        "source_run_id": SOURCE_RUNS[source_copy],
        "prior_copies_in_shared_mean": prior_copies,
        "shared_tensors_averaged": averaged,
        "experts_added_per_layer": total_added // la,
        "per_layer": per_layer,
    })


def main() -> None:
    out = Path("repaired")
    out.mkdir(exist_ok=True)
    source_files = {
        i: only_file(Path(f"copies/{i}"), "chat.pt") for i in range(5)
    }
    tokenizer_files = {
        i: only_file(Path(f"copies/{i}"), "tokenizer.json") for i in range(5)
    }
    tok_hashes = {p.read_bytes() for p in tokenizer_files.values()}
    if len(tok_hashes) != 1:
        raise RuntimeError("The five source checkpoints do not share byte-identical tokenizers")

    base = load_checkpoint(source_files[0])
    merged = base["model"]
    config = dict(base["config"])
    report = {
        "repair": "preserve each source checkpoint's router rows with its expert weights",
        "shared_weight_average": "unbiased five-way arithmetic mean via online mean",
        "deduplication_threshold": THRESHOLD,
        "source_run_ids": SOURCE_RUNS,
        "copies": [],
        "router_row_policy": "concatenate source router rows in expert-index order",
    }

    # Each fold mutates only this isolated experiment's in-memory accumulator.
    # The five original artifacts and checkpoints are read-only inputs.
    for source_copy in range(1, 5):
        incoming = load_checkpoint(source_files[source_copy])
        fold(base, incoming, merged, source_copy, source_copy, report)
        # Keep a single accumulator reference; release incoming checkpoint pages.
        base["model"] = merged
        del incoming
        if source_copy > 0:
            import shutil
            shutil.rmtree(Path(f"copies/{source_copy}"))
        print(
            f"copy {source_copy}: added "
            f"{report['copies'][-1]['experts_added_per_layer']} experts/layer; "
            f"router rows preserved"
        )

    config["experts"] = expert_count(merged, 0)
    if len(set(expert_counts(merged))) != 1:
        raise RuntimeError(f"layers ended with inconsistent expert counts: {expert_counts(merged)}")
    cfg = MoEConfig(**config)
    counts = cfg.counts()
    total_parameters = sum(t.numel() for t in merged.values())
    if total_parameters != counts["total"]:
        raise RuntimeError(
            f"parameter count mismatch: state={total_parameters}, config={counts['total']}"
        )

    checkpoint = {
        "model": merged,
        "config": config,
        "args": base.get("args", {}),
        "report": {
            "parameters": counts["total"],
            "total_parameters": counts["total"],
            "active_parameters": counts["active"],
            "experts": cfg.experts,
            "top_k": cfg.top_k,
            "active_fraction": round(counts["active"] / counts["total"], 6),
        },
        "merge": report,
    }
    torch.save(checkpoint, out / "merged.pt")
    (out / "tokenizer.json").write_bytes(tokenizer_files[0].read_bytes())
    summary = {
        "config": config,
        "parameters": counts["total"],
        "total_parameters": counts["total"],
        "active_parameters": counts["active"],
        "merge": report,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "total_parameters": counts["total"],
        "active_parameters": counts["active"],
        "experts_per_layer": cfg.experts,
        "layers": cfg.layers,
        "router_row_policy": report["router_row_policy"],
        "shared_weight_average": report["shared_weight_average"],
    }, indent=2))


if __name__ == "__main__":
    main()
