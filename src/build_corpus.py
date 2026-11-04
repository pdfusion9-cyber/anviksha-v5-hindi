"""Build a plain-text corpus for the pretrain stage from the chat pairs.

The pretrain stage needs raw text and refuses to run on a stub. Rather than
embedding a heredoc inside the workflow -- which broke the YAML -- the job is
done here, where it can be read and tested.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", type=Path, default=Path("data/multi_train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("hindi_corpus.txt"))
    ap.add_argument("--chars", type=int, default=3_000_000)
    args = ap.parse_args()

    if args.out.is_file() and args.out.stat().st_size > 5000:
        print(f"{args.out} already has {args.out.stat().st_size:,} bytes")
        return

    rows = [json.loads(l) for l in
            args.chat.read_text(encoding="utf-8").splitlines() if l.strip()]
    text = "\n".join(r["user"] + "\n" + r["assistant"] for r in rows)[: args.chars]
    args.out.write_text(text, encoding="utf-8")
    deva = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    print(f"wrote {args.out}: {len(text):,} chars, {deva/len(text):.1%} Devanagari")


if __name__ == "__main__":
    main()
