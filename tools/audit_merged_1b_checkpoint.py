from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, "src")
from bpe_tokenizer import BPETokenizer
from moe_model import MoEConfig


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck or "config" not in ck:
        raise SystemExit("checkpoint missing model/config")
    cfg = MoEConfig(**ck["config"])
    tok = BPETokenizer.load(args.tokenizer)
    if tok.size != cfg.vocab:
        raise SystemExit(f"tokenizer size {tok.size} != model vocab {cfg.vocab}")
    if tok.eos < tok.base or tok.eos >= tok.size:
        raise SystemExit("EOS token ID outside tokenizer special-token range")

    state = ck["model"]
    if not isinstance(state, dict):
        raise SystemExit("checkpoint model state is not a mapping")
    total = 0
    nonfinite = []
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise SystemExit(f"invalid tensor state entry: {name}")
        total += tensor.numel()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            nonfinite.append(name)
    if nonfinite:
        raise SystemExit(f"non-finite tensors: {nonfinite[:10]}")

    embed = state.get("embed.weight")
    head = state.get("head.weight")
    tied = bool(embed is not None and head is not None and
                embed.shape == head.shape and torch.equal(embed, head))
    if not tied:
        raise SystemExit("embedding/output weights are not tied-identical")
    extra = ck.get("extra") or {}
    usage = extra.get("expert_usage", [])
    usage_sum = float(sum(usage)) if usage else 0.0
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(args.checkpoint),
        "tokenizer_sha256": sha256(args.tokenizer),
        "global_step": int(ck.get("step", 0)),
        "parameters": total,
        "config": cfg.__dict__,
        "tokenizer": {"size": tok.size, "base": tok.base, "eos_id": tok.eos},
        "tied_embeddings": tied,
        "all_float_tensors_finite": not nonfinite,
        "expert_usage_available": bool(usage),
        "expert_usage": usage,
        "expert_usage_sum": usage_sum,
        "optimizer_state_present": isinstance(ck.get("optimiser"), dict),
        "resume_policy": extra.get("resume_policy"),
        "corpus_sha256": extra.get("corpus_sha256"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
