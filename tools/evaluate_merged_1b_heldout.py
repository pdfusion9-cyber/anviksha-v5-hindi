from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from bpe_tokenizer import BPETokenizer
from moe_model import MoEConfig, MoELM


def load(checkpoint: Path):
    ck = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg_data = dict(ck["config"])
    cfg_data["balanced_routing"] = True
    cfg = MoEConfig(**cfg_data)
    with torch.device("meta"):
        model = MoELM(cfg)
    result = model.load_state_dict(ck["model"], strict=True, assign=True)
    model.head.weight = model.embed.weight
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {result}")
    model.eval()
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-docs", type=int, default=36)
    args = ap.parse_args()
    tok = BPETokenizer.load(args.tokenizer)
    model, ck = load(args.checkpoint)
    losses = {"en": [], "hi": []}
    rows = []
    with torch.inference_mode():
        for line in args.heldout.open(encoding="utf-8"):
            if len(rows) >= args.max_docs:
                break
            obj = json.loads(line)
            lang = obj["language"]
            ids = tok.encode(obj["text"])
            if len(ids) < 4:
                continue
            # Non-overlapping chunks with next-token loss; never train on this set.
            losses_for_doc = []
            for start in range(0, min(len(ids) - 1, model.cfg.seq_len * 4), model.cfg.seq_len):
                chunk = ids[start:start + model.cfg.seq_len + 1]
                if len(chunk) < 2:
                    continue
                x = torch.tensor([chunk[:-1]], dtype=torch.long)
                y = torch.tensor([chunk[1:]], dtype=torch.long)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                losses_for_doc.append(float(loss))
            if losses_for_doc:
                value = sum(losses_for_doc) / len(losses_for_doc)
                losses.setdefault(lang, []).append(value)
                rows.append({"language": lang, "topic": obj.get("topic"),
                             "tokens": len(ids), "loss": value})
    report = {
        "checkpoint": str(args.checkpoint),
        "global_step": int(ck.get("step", 0)),
        "heldout_sha256": hashlib.sha256(args.heldout.read_bytes()).hexdigest(),
        "documents": len(rows),
        "by_language": {
            lang: {"documents": len(values),
                   "mean_loss": sum(values) / len(values) if values else None}
            for lang, values in losses.items()
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["by_language"], ensure_ascii=False, indent=2))
    if len(rows) < 12 or not losses.get("en") or not losses.get("hi"):
        raise SystemExit("held-out evaluation requires at least 12 docs and both languages")


if __name__ == "__main__":
    main()
