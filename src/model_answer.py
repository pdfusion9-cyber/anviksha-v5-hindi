"""Wire the model in as the reasoning fallback, behind the judge.

The pipeline was built for three stages and only ever ran the first one:

    1. retrieval      a stored, checked answer -> correct by construction
    2. the model      writes <think>...</think><ans>...</ans> when nothing matched
    3. the judge      gates whatever the model wrote, and refuses if it fails

`generate` was a parameter on the pipeline from the beginning and was passed None
in every test, so stage 2 and 3 never ran. This supplies stage 2.

The reasoning is not decoration. Every training reply was rewritten so the first
thing after <assistant> is <think>, and generation refuses a reply with no <ans>
block -- so the model cannot answer without going through the trace, and the
judge scores only what is inside <ans>, so a fluent trace cannot earn credit for
a wrong answer.

What this can and cannot buy is worth stating before it is measured: an 11M
parameter model does not reason, it restates. The trace gives it a place to
condition on the question and the expected shape. The judge is what keeps a
wrong restatement from leaving. Expect a small lift in coverage, not a leap.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reasoning import parse  # noqa: E402


def make_generate(checkpoint: Path, *, max_new: int = 200, temperature: float = 0.8,
                  top_k: int = 40, seed: int = 0) -> Callable[[str], str | None]:
    """Return a generate(question) -> answer text, or None if it did not answer."""
    import torch

    from checkpoint import build_from_checkpoint
    from tokenizer import CharTokenizer  # noqa: F401

    model, tokenizer, cfg = build_from_checkpoint(checkpoint)
    model.eval()
    seq_len = cfg["seq_len"]

    def generate(question: str) -> str | None:
        ids = ([tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
               + tokenizer.encode(question)
               + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"],
                  tokenizer.stoi.get("<hi>", tokenizer.eos)])
        # the question must leave room for the trace and the answer; if it does
        # not, there is no point sampling a reply that cannot finish
        if len(ids) + 8 >= seq_len:
            return None

        torch.manual_seed(seed)
        tokens = torch.tensor([ids])
        with torch.no_grad():
            for _ in range(max_new):
                if tokens.shape[1] >= seq_len:
                    break
                logits = model(tokens)[:, -1, :] / temperature
                values, _ = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
                logits[logits < values[:, -1:]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, nxt], dim=1)
                if int(nxt) == tokenizer.eos:
                    break

        raw = tokenizer.decode(tokens[0, len(ids):].tolist())
        parsed = parse(raw)
        if not parsed.complete:
            return None          # an unfinished thought is not an answer
        text = (parsed.answer or "").strip()
        return text or None

    return generate


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/chat.pt"))
    ap.add_argument("question", nargs="*")
    args = ap.parse_args()
    if not args.checkpoint.is_file():
        print(f"missing checkpoint {args.checkpoint}")
        raise SystemExit(2)

    gen = make_generate(args.checkpoint)
    questions = args.question or ["भारत की राजधानी क्या है?", "नमस्ते"]
    for q in questions:
        print(f"Q: {q}")
        a = gen(q)
        print(f"A: {a if a else '(no <ans> block -- refused)'}\n")
