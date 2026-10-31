"""Evaluate the trained checkpoint on a dataset it has never seen.

The training data came from smangrul/hindi_instruct_v1. This script scores the
same checkpoint on ai4bharat/indic-instruct-data-v0.1 (Dolly, Hindi), which is a
different corpus with a different translation pipeline — so the number it
produces is a generalisation measurement, not a memory test.

Only pairs that fit the model's 128-character context are scored, and the script
reports how many of the source rows qualified, because quietly dropping most of
the data would flatter the result.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import HindiLM  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def load_unseen(path: Path, max_prompt: int, max_reply: int) -> tuple[list[dict], int]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept = [r for r in rows
                if 4 <= len(r["user"]) <= max_prompt and 4 <= len(r["assistant"]) <= max_reply]
        return kept, len(rows)
    handle = pq.ParquetFile(path, memory_map=True)
    total = 0
    pairs: list[dict] = []
    for batch in handle.iter_batches(batch_size=512, columns=["instruction", "response"]):
        for row in batch.to_pylist():
            total += 1
            instruction = (row["instruction"] or "").strip()
            response = (row["response"] or "").strip()
            if not instruction or not response:
                continue
            if len(instruction) > max_prompt or len(response) > max_reply:
                continue
            pairs.append({"user": instruction, "assistant": response})
    return pairs, total


@torch.no_grad()
def perplexity(model, tok, pairs, seq_len, batch=2) -> tuple[float, int, int]:
    total_loss, tokens, used = 0.0, 0, 0
    for start in range(0, len(pairs), batch):
        rows = []
        for pair in pairs[start : start + batch]:
            row = ([tok.stoi["<bos>"], tok.stoi["<user>"]] + tok.encode(pair["user"])
                   + [tok.stoi["<sep>"], tok.stoi["<assistant>"]]
                   + tok.encode(pair["assistant"]) + [tok.eos])[:seq_len]
            rows.append(row + [tok.pad] * (seq_len - len(row)))
        tensor = torch.tensor(rows, dtype=torch.long)
        logits = model(tensor[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, tok.size), tensor[:, 1:].reshape(-1),
                               ignore_index=tok.pad, reduction="sum")
        total_loss += float(loss)
        tokens += int((tensor[:, 1:] != tok.pad).sum())
        used += len(rows)
    return math.exp(total_loss / max(tokens, 1)), tokens, used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="cache/dolly.parquet", type=Path)
    parser.add_argument("--checkpoint", default="runs/chat.pt", type=Path)
    parser.add_argument("--out", default="runs/unseen_eval.json", type=Path)
    parser.add_argument("--limit", default=400, type=int)
    parser.add_argument("--max-prompt", default=60, type=int)
    parser.add_argument("--max-reply", default=60, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)

    tok = CharTokenizer.load(args.checkpoint.parent / "tokenizer.json")
    state = torch.load(args.checkpoint, weights_only=False)
    cfg = state["args"]
    model = HindiLM(tok.size, cfg["dim"], cfg["layers"], cfg["heads"], cfg["ffn"],
                    cfg["seq_len"])
    model.load_state_dict(state["model"])
    model.eval()

    pairs, total = load_unseen(args.parquet, args.max_prompt, args.max_reply)
    print(f"{total} rows in the unseen corpus, {len(pairs)} fit the "
          f"{args.max_prompt}/{args.max_reply} character window")
    pairs = pairs[: args.limit]

    ppl, tokens, used = perplexity(model, tok, pairs, cfg["seq_len"])
    report = {
        "unseen_dataset": "ai4bharat/indic-instruct-data-v0.1 (dolly, hi)",
        "training_dataset": "smangrul/hindi_instruct_v1",
        "rows_in_corpus": total,
        "rows_that_fit_window": len(pairs),
        "pairs_scored": used,
        "tokens_scored": tokens,
        "unseen_perplexity": round(ppl, 3),
        "window_chars": [args.max_prompt, args.max_reply],
    }
    print(json.dumps(report, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
