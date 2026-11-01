"""Rejection-sampling fine-tuning: turn the rewards into weights.

The rewards so far only score text after it exists, and the self-correction loop
only filters at answer time. Neither changes the model. This script does:

    1. ask the model for several replies to each prompt
    2. score every reply on path and form (and on agreement with a reference
       when the prompt has one)
    3. keep only the best reply, and only if it clears the threshold
    4. write the survivors as a chat corpus

Training on that corpus is what moves the reward into the weights: the model
learns from its own good outputs rather than from a rule written by hand.

If too few prompts survive, the honest output is an empty file and a clear
number saying so -- not a quietly degraded training set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint import build_from_checkpoint  # noqa: E402
from grammar_reward import score_hindi  # noqa: E402
from meaning_reward import score_against_reference  # noqa: E402
from path_reward import score_path  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def sample(model, tokenizer, cfg, prompt: str, temperature: float, top_k: int,
           max_new: int, seed: int) -> str:
    torch.manual_seed(seed)
    ids = ([tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
           + tokenizer.encode(prompt)
           + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"],
              tokenizer.stoi.get("<hi>", tokenizer.eos)])
    tokens = torch.tensor([ids])
    with torch.no_grad():
        for _ in range(max_new):
            if tokens.shape[1] >= cfg["seq_len"]:
                break
            logits = model(tokens)[:, -1, :] / temperature
            values, _ = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
            logits[logits < values[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, nxt], dim=1)
            if int(nxt) == tokenizer.eos:
                break
    return tokenizer.decode(tokens[0, len(ids):].tolist())


def score(prompt: str, reply: str, reference: str | None) -> tuple[float, dict]:
    if len(reply.strip()) < 2:
        return 0.0, {"path": 0.0, "form": 0.0, "answer": None, "notes": ["empty reply"]}
    path = score_path(prompt, reply)
    form = score_hindi(reply)
    answer = score_against_reference(reply, reference) if reference else None
    if answer is None:
        combined = 0.6 * path.score + 0.4 * form.score
    else:
        combined = 0.35 * path.score + 0.25 * form.score + 0.40 * answer.total
    return combined, {
        "path": round(path.score, 1),
        "form": round(form.score, 1),
        "answer": None if answer is None else answer.total,
        "notes": path.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/chat.pt", type=Path)
    parser.add_argument("--prompts", default="data/multi_heldout.jsonl", type=Path)
    parser.add_argument("--out", default="data/rft_train.jsonl", type=Path)
    parser.add_argument("--report", default="runs/rft_report.json", type=Path)
    parser.add_argument("--limit", default=800, type=int)
    parser.add_argument("--candidates", default=4, type=int)
    parser.add_argument("--threshold", default=60.0, type=float)
    parser.add_argument("--temperature", default=0.85, type=float)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--max-new", default=60, type=int)
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    tokenizer = CharTokenizer.load(args.checkpoint.parent / "tokenizer.json")
    model, cfg = build_from_checkpoint(args.checkpoint, tokenizer)

    rows = [json.loads(line) for line in
            args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.limit]

    kept, tried, scores = [], 0, []
    for i, row in enumerate(rows):
        prompt = row["user"]
        reference = row.get("assistant")
        best = None
        for c in range(args.candidates):
            reply = sample(model, tokenizer, cfg, prompt, args.temperature,
                           args.top_k, args.max_new, seed=i * 100 + c)
            tried += 1
            combined, detail = score(prompt, reply, reference)
            if best is None or combined > best[0]:
                best = (combined, reply, detail)
        scores.append(best[0])
        if best[0] >= args.threshold:
            kept.append({"user": prompt, "assistant": best[1],
                         "score": round(best[0], 1), "source": "rft"})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} prompts, kept {len(kept)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n",
                        encoding="utf-8")

    report = {
        "prompts": len(rows),
        "candidates_per_prompt": args.candidates,
        "generations": tried,
        "kept": len(kept),
        "kept_fraction": round(len(kept) / max(len(rows), 1), 3),
        "threshold": args.threshold,
        "mean_best_score": round(sum(scores) / max(len(scores), 1), 1),
        "max_best_score": round(max(scores), 1) if scores else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not kept:
        print("\nNo reply cleared the threshold. Training on this file would teach "
              "nothing, so the run should stop here rather than continue.")


if __name__ == "__main__":
    main()
