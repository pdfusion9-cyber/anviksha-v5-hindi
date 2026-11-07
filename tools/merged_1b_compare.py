from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from bpe_tokenizer import BPETokenizer
from moe_model import MoEConfig, MoELM

PROMPTS = [
    "Cryptography protects data by converting plaintext into",
    "A network firewall examines packets to decide",
    "SQL injection happens when untrusted input is inserted into",
    "An incident response plan should begin with",
    "Access control ensures that each user can",
    "साइबर सुरक्षा में नेटवर्क ट्रैफिक का विश्लेषण",
]


def load_model(checkpoint: Path):
    ck = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg_data = dict(ck["config"])
    cfg_data["balanced_routing"] = True
    cfg = MoEConfig(**cfg_data)
    with torch.device("meta"):
        model = MoELM(cfg)
    result = model.load_state_dict(ck["model"], strict=True, assign=True)
    model.head.weight = model.embed.weight
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict load failed: {result}")
    model.eval()
    return model, ck


def evaluate(checkpoint: Path, tokenizer: BPETokenizer, label: str, max_new: int):
    model, ck = load_model(checkpoint)
    cfg = model.cfg
    blocked = list(range(tokenizer.base, tokenizer.size))
    rows = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = tokenizer.encode(prompt)
            if not ids:
                raise RuntimeError(f"tokenizer returned no tokens: {prompt}")
            context = ids[-cfg.seq_len:]
            x = torch.tensor([context], dtype=torch.long)
            logits = model(x)[0, -1].float()
            if not torch.isfinite(logits).all().item():
                raise RuntimeError(f"non-finite logits for prompt: {prompt}")
            eos_rank = 1 + int((logits > logits[tokenizer.eos]).sum().item())
            top_ids = torch.topk(logits, 5).indices.tolist()
            generated = []
            for _ in range(max_new):
                x = torch.tensor([context[-cfg.seq_len:]], dtype=torch.long)
                next_logits = model(x)[0, -1].float().clone()
                next_logits[blocked] = -torch.inf
                token_id = int(torch.argmax(next_logits).item())
                generated.append(token_id)
                context.append(token_id)
            text = tokenizer.decode(generated)
            rows.append({
                "prompt": prompt,
                "first_raw_top_ids": top_ids,
                "eos_rank": eos_rank,
                "generated_tokens": len(generated),
                "unique_token_ratio": round(len(set(generated)) / max(len(generated), 1), 4),
                "continuation": text,
            })
    del model
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(ck.get("step", 0)),
        "expert_usage": (ck.get("extra") or {}).get("expert_usage", []),
        "prompts": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()
    torch.set_num_threads(2)
    tok = BPETokenizer.load(args.tokenizer)
    before = evaluate(args.baseline, tok, "baseline_repaired_1b", args.max_new)
    after = evaluate(args.candidate, tok, "eos_aware_continuation", args.max_new)
    rows = []
    for b, a in zip(before["prompts"], after["prompts"]):
        rows.append({
            "prompt": b["prompt"],
            "baseline_eos_rank": b["eos_rank"],
            "candidate_eos_rank": a["eos_rank"],
            "baseline_unique_token_ratio": b["unique_token_ratio"],
            "candidate_unique_token_ratio": a["unique_token_ratio"],
            "baseline_continuation": b["continuation"],
            "candidate_continuation": a["continuation"],
        })
    report = {
        "method": "same six prompts, same tokenizer, greedy decode, all special tokens suppressed",
        "baseline": before,
        "candidate": after,
        "comparison": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline_step": before["checkpoint_step"],
                      "candidate_step": after["checkpoint_step"],
                      "comparison": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
