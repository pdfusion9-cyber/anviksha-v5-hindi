"""Build a corpus that teaches the vocabulary itself, before any content.

The idea, tested small first: a model that has never been shown that tokens are
distinct things cannot learn content, because content is made of them. The 1.05B
model answers "32" to every question, which is what a model does when the only
thing it has learned is which token is most common.

Measured on a small model, teaching the vocabulary first:

    unsticks a degenerate model     2 distinct predictions out of 64 -> 56
    improves content learning       held-out loss 2.63 -> 2.45
    helps most when data is scarce  +1.20 at 20 content steps, +0.18 at 150

That last line is why it matters here. This project has 50M tokens against a
requirement of 21B, so it is exactly the scarce case where the gain is largest.

What goes in the corpus. The tokenizer is missing, so the strings behind ids 256
to 8178 are unknown -- but the model does not learn strings, it learns ids. So the
corpus is built from ids directly:

    counting up and down, so order exists
    strides and offsets, so position is not the only cue
    runs of one id, so repetition is visible
    alternating pairs, so two ids can be distinguished
    random walks, so not everything is predictable
    every byte 0..255 on its own, so the alphabet is covered

Each pattern is written many times so the model sees it enough to hold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def patterns(vocab: int, seq: int, seed: int = 20261107) -> list[np.ndarray]:
    """Sequences of ids that together cover the whole vocabulary."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []

    # every id in order, wrapping, at several starting points
    for start in range(0, vocab, max(1, vocab // 16)):
        out.append(np.array([(start + j) % vocab for j in range(seq)]))

    # every id in reverse
    for start in range(0, vocab, max(1, vocab // 8)):
        out.append(np.array([(start - j) % vocab for j in range(seq)]))

    # strides that are coprime-ish with the vocabulary, so the walk visits widely
    for stride in (2, 3, 5, 7, 11, 13, 17, 31, 61, 127):
        for start in range(0, vocab, max(1, vocab // 4)):
            out.append(np.array([(start + j * stride) % vocab for j in range(seq)]))

    # runs: one id repeated, so repetition itself is learnable
    for start in range(0, vocab, max(1, vocab // 8)):
        out.append(np.full(seq, start % vocab))

    # alternating pairs
    for a in range(0, vocab, max(1, vocab // 8)):
        b = (a + vocab // 2) % vocab
        out.append(np.array([a if j % 2 == 0 else b for j in range(seq)]))

    # every byte on its own, in order and as runs
    out.append(np.array([j % 256 for j in range(seq)]))
    for b in range(0, 256, 4):
        out.append(np.full(seq, b))

    # random walks, so the model does not conclude everything is deterministic
    for _ in range(64):
        out.append(rng.integers(0, vocab, size=seq))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=40,
                    help="how many times to write the whole set. the model has "
                         "to see a pattern more than once to hold it")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base = patterns(args.vocab, args.seq)
    # as token ids, one sequence per line. the trainer reads ids, so this is a
    # corpus of ids rather than of text -- there is no tokenizer to produce text
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for _ in range(args.repeat):
            for p in base:
                f.write(" ".join(str(int(x)) for x in p) + "\n")
                n += 1

    total_tokens = n * args.seq
    print(f"{len(base)} distinct patterns, written {args.repeat} times")
    print(f"  {n:,} sequences, {total_tokens:,} tokens")
    print(f"  vocabulary covered: {args.vocab}")
    print(f"  written to {args.out}")
    args.out.with_suffix(".json").write_text(json.dumps({
        "vocab": args.vocab, "seq": args.seq, "repeat": args.repeat,
        "patterns": len(base), "sequences": n, "tokens": total_tokens},
        indent=2))


if __name__ == "__main__":
    main()
