from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import sys
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


def find_one(root: Path, name: str) -> Path:
    found = list(root.rglob(name))
    if len(found) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(found)}")
    return found[0]


def render_token(tokenizer, token_id: int) -> str:
    if token_id >= tokenizer.base:
        index = token_id - tokenizer.base
        return tokenizer.specials[index] if index < len(tokenizer.specials) else "<special?>"
    return tokenizer.decode([token_id])


def main() -> None:
    root = Path("artifact")
    ckpt_path = find_one(root, "merged.pt")
    tok_path = find_one(root, "tokenizer.json")
    ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    tokenizer = BPETokenizer.load(tok_path)
    cfg_data = dict(ck["config"])
    cfg_data["balanced_routing"] = True
    cfg = MoEConfig(**cfg_data)

    torch.set_num_threads(2)
    with torch.device("meta"):
        model = MoELM(cfg)
    loaded = model.load_state_dict(ck["model"], strict=True, assign=True)
    model.head.weight = model.embed.weight
    model.eval()
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"strict load failed: missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}")

    blocked_ids = list(range(tokenizer.base, tokenizer.size))
    if tokenizer.eos not in blocked_ids:
        raise RuntimeError("EOS id is not in tokenizer special-token range")

    report = {
        "source_artifact": "repaired-merged-1b-model from workflow run 42352023153",
        "checkpoint": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "experts_per_layer": cfg.experts,
            "layers": cfg.layers,
            "top_k": cfg.top_k,
            "vocab": cfg.vocab,
        },
        "generation_policy": {
            "mode": "diagnostic-only greedy continuation",
            "max_new_tokens": 48,
            "all_special_tokens_suppressed": True,
            "eos_suppressed": True,
            "reason": "training stream currently tokenizes document text without appending EOS; EOS rank is reported separately",
        },
        "special_tokens": {
            "base": tokenizer.base,
            "size": tokenizer.size,
            "eos_id": tokenizer.eos,
            "eos_symbol": tokenizer.specials[tokenizer.eos - tokenizer.base],
            "blocked_ids": blocked_ids,
        },
        "prompts": [],
    }

    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = tokenizer.encode(prompt)
            if not ids:
                report["prompts"].append({"prompt": prompt, "error": "empty tokenizer output"})
                continue
            context = list(ids[-cfg.seq_len:])
            x = torch.tensor([context], dtype=torch.long)
            logits = model(x)[0, -1].float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"non-finite first-step logits for prompt: {prompt}")

            eos_logit = float(logits[tokenizer.eos])
            eos_rank = 1 + int((logits > logits[tokenizer.eos]).sum().item())
            top_values, top_ids = torch.topk(logits, k=12)
            raw_top = [
                {
                    "id": int(tid),
                    "token": render_token(tokenizer, int(tid)),
                    "logit": round(float(val), 5),
                    "is_special": int(tid) >= tokenizer.base,
                }
                for val, tid in zip(top_values.tolist(), top_ids.tolist())
            ]

            generated = []
            for _ in range(48):
                x = torch.tensor([context[-cfg.seq_len:]], dtype=torch.long)
                next_logits = model(x)[0, -1].float().clone()
                next_logits[blocked_ids] = -torch.inf
                nxt = int(torch.argmax(next_logits).item())
                generated.append(nxt)
                context.append(nxt)

            continuation = tokenizer.decode(generated)
            report["prompts"].append({
                "prompt": prompt,
                "input_tokens": len(ids),
                "raw_first_prediction": raw_top[0],
                "eos_rank": eos_rank,
                "eos_logit": round(eos_logit, 5),
                "raw_top_12": raw_top,
                "diagnostic_continuation": continuation,
                "generated_tokens": len(generated),
                "unique_token_ratio": round(len(set(generated)) / max(len(generated), 1), 4),
            })

    out = Path("diagnostic")
    out.mkdir(exist_ok=True)
    (out / "generation_diagnostic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "eos_id": tokenizer.eos,
        "prompt_results": [
            {
                "prompt": row["prompt"],
                "raw_first_prediction": row.get("raw_first_prediction"),
                "eos_rank": row.get("eos_rank"),
                "diagnostic_continuation": row.get("diagnostic_continuation"),
            }
            for row in report["prompts"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
