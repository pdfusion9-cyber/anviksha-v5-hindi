"""Merge a fine-tuned candidate into the model without ever risking the base.

The last RFT attempt replaced a good checkpoint with a worse one, because
fine-tuning wrote straight over `runs/chat.pt`. This script exists so that
cannot happen again:

    the base checkpoint is only ever read
    the candidate is written to its own file
    a merged file is produced as a third file
    all three are scored on the same held-out set
    the best one is recorded as the decision, and nothing is overwritten

The merge itself is a weight average ("model soup") controlled by alpha:
alpha=0 is the base, alpha=1 is the candidate. Averaging often keeps the
candidate's gains while pulling back its damage, but that is decided by the
measurement, not assumed -- if the base still wins, the base is what gets used.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint import build_from_checkpoint  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


@torch.no_grad()
def perplexity(model, tok, pairs, seq_len: int, batch: int = 4) -> float:
    total, tokens = 0.0, 0
    for start in range(0, len(pairs), batch):
        rows = []
        for pair in pairs[start:start + batch]:
            row = ([tok.stoi["<bos>"], tok.stoi["<user>"]] + tok.encode(pair["user"])
                   + [tok.stoi["<sep>"], tok.stoi["<assistant>"]]
                   + tok.encode(pair["assistant"]) + [tok.eos])[:seq_len]
            rows.append(row + [tok.pad] * (seq_len - len(row)))
        tensor = torch.tensor(rows, dtype=torch.long)
        logits = model(tensor[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, tok.size),
                               tensor[:, 1:].reshape(-1),
                               ignore_index=tok.pad, reduction="sum")
        total += float(loss)
        tokens += int((tensor[:, 1:] != tok.pad).sum())
    mean = total / max(tokens, 1)
    try:
        return math.exp(mean)
    except OverflowError:
        return float("inf")


def merge_state(base: dict, candidate: dict, alpha: float) -> dict:
    """Weight average. Every tensor in the candidate must match the base."""
    out = {}
    for key, base_tensor in base.items():
        if key not in candidate:
            raise KeyError(f"candidate is missing {key}")
        cand = candidate[key]
        if cand.shape != base_tensor.shape:
            raise ValueError(f"{key}: shape {cand.shape} != {base_tensor.shape}")
        if base_tensor.is_floating_point():
            out[key] = (1 - alpha) * base_tensor.float() + alpha * cand.float()
        else:
            out[key] = base_tensor
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="runs/chat.pt", type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--heldout", default="data/multi_heldout.jsonl", type=Path)
    parser.add_argument("--out", default="runs/merged.pt", type=Path)
    parser.add_argument("--decision", default="runs/merge_decision.json", type=Path)
    parser.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--limit", default=150, type=int)
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    if args.out.resolve() == args.base.resolve():
        raise SystemExit("refusing to overwrite the base checkpoint")
    if args.out.resolve() == args.candidate.resolve():
        raise SystemExit("refusing to overwrite the candidate checkpoint")

    tok = CharTokenizer.load(args.base.parent / "tokenizer.json")
    pairs = [json.loads(line) for line in
             args.heldout.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = pairs[: args.limit]

    base_state = torch.load(args.base, weights_only=False)
    cand_state = torch.load(args.candidate, weights_only=False)
    cfg = base_state["args"]

    base_model, _ = build_from_checkpoint(args.base, tok)
    base_ppl = perplexity(base_model, tok, pairs, cfg["seq_len"])

    results = [{"alpha": 0.0, "label": "base", "perplexity": round(base_ppl, 4)}]
    best = (base_ppl, 0.0, base_state["model"])

    for alpha in [float(a) for a in args.alphas.split(",") if a.strip()]:
        if alpha == 0.0:
            continue
        merged = merge_state(base_state["model"], cand_state["model"], alpha)
        model, _ = build_from_checkpoint(args.base, tok)
        model.load_state_dict(merged)
        model.eval()
        ppl = perplexity(model, tok, pairs, cfg["seq_len"])
        label = "candidate" if alpha == 1.0 else f"merge@{alpha}"
        results.append({"alpha": alpha, "label": label,
                        "perplexity": round(ppl, 4) if math.isfinite(ppl) else None})
        if math.isfinite(ppl) and ppl < best[0]:
            best = (ppl, alpha, merged)

    # whatever won, it is written to --out, a file that did not exist before
    torch.save({"model": best[2], "args": cfg}, args.out)

    decision = {
        "base": str(args.base),
        "candidate": str(args.candidate),
        "written": str(args.out),
        "scored": results,
        "winner_alpha": best[1],
        "winner_perplexity": round(best[0], 4),
        "base_perplexity": round(base_ppl, 4),
        "base_was_kept": best[1] == 0.0,
        "note": ("the base checkpoint was read only; it is unchanged"
                 if best[1] != 1.0 else
                 "the candidate won outright at alpha=1.0; base file still unchanged"),
    }
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    args.decision.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
