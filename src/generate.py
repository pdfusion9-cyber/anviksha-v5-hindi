"""Sample replies from the chat model, optionally with answer-time self-correction.

Two modes:

    default          one sample per prompt, screened and printed
    --self-correct   several candidates per prompt, scored on path and form,
                     and only returned if one clears the threshold; otherwise
                     the loop goes back and retries, and finally refuses

The second mode is the one that keeps a wrong reply from reaching a user. It
costs more compute per answer, so it is opt-in rather than the default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint import build_from_checkpoint  # noqa: E402
from safety import screen  # noqa: E402
from self_correct import answer_with_self_correction  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

PROMPTS = [
    "नमस्ते",
    "आपका नाम क्या है?",
    "आप कैसे हैं?",
    "भारत की राजधानी क्या है?",
    "मुझे उदासी है।",
    "धन्यवाद",
]


def make_generator(model, tokenizer, cfg, temperature: float, top_k: int,
                   max_new: int):
    """Wrap the sampler as (prompt, seed) -> text, which is what the loop wants."""
    def generate(prompt: str, seed: int) -> str:
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
    return generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/chat.pt", type=Path)
    parser.add_argument("--out", default="runs/generations.json")
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--max-new", default=40, type=int)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--self-correct", action="store_true",
                        help="generate several candidates and only answer if one "
                             "clears --threshold")
    parser.add_argument("--threshold", default=65.0, type=float)
    parser.add_argument("--candidates", default=4, type=int)
    parser.add_argument("--rounds", default=3, type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    tokenizer = CharTokenizer.load(args.checkpoint.parent / "tokenizer.json")
    model, cfg = build_from_checkpoint(args.checkpoint, tokenizer)
    generate = make_generator(model, tokenizer, cfg, args.temperature,
                              args.top_k, args.max_new)

    results = []
    for prompt in PROMPTS:
        if args.self_correct:
            outcome = answer_with_self_correction(
                prompt, generate, reference=None, threshold=args.threshold,
                candidates_per_round=args.candidates, max_rounds=args.rounds)
            reply = outcome.answer
            results.append({
                "prompt": prompt,
                "reply": reply,
                "accepted": outcome.accepted,
                "rounds": outcome.rounds,
                "candidates_tried": outcome.candidates_tried,
                "best_combined": None if outcome.best is None else round(outcome.best.combined, 1),
                "best_path": None if outcome.best is None else round(outcome.best.path, 1),
                "refusal_reason": outcome.refusal_reason,
                "safety": screen(reply) if reply else {"clean": True, "issues": []},
            })
        else:
            reply = generate(prompt, 0)
            results.append({"prompt": prompt, "reply": reply, "safety": screen(reply)})

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    for row in results:
        if args.self_correct and not row["accepted"]:
            print(f"U: {row['prompt']}\nA: [मना किया] {row['refusal_reason']}\n")
            continue
        flag = "" if row["safety"]["clean"] else f"  [FLAG: {row['safety']['issues']}]"
        print(f"U: {row['prompt']}\nA: {row['reply']}{flag}\n")


if __name__ == "__main__":
    main()
