"""Fetch a Hindi instruction dataset and split it into train and held-out.

Source: smangrul/hindi_instruct_v1 (MIT licence, 20,215 rows).
Only the 13 MB parquet is downloaded, and it is read in batches so the whole
table never sits in memory at once.

The held-out split is carved out by index before training and written to a
separate file, so it cannot leak into training.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

URL = ("https://huggingface.co/datasets/smangrul/hindi_instruct_v1/"
       "resolve/main/data/train-00000-of-00001.parquet")


def download(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", URL, "-o", str(path)], check=True)
    return path


def extract_pairs(path: Path, max_reply: int, max_prompt: int) -> list[dict]:
    pairs: list[dict] = []
    seen: set[str] = set()
    handle = pq.ParquetFile(path, memory_map=True)
    for batch in handle.iter_batches(batch_size=512, columns=["messages"]):
        for row in batch.to_pylist():
            messages = row["messages"]
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant = next(
                (m["content"] for m in messages if m["role"] == "assistant"), ""
            )
            user, assistant = user.strip(), assistant.strip()
            if not (4 <= len(user) <= max_prompt and 4 <= len(assistant) <= max_reply):
                continue
            key = user[:120]
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"user": user, "assistant": assistant})
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=8000, type=int)
    parser.add_argument("--held-out", default=400, type=int)
    parser.add_argument("--max-reply", default=220, type=int)
    parser.add_argument("--max-prompt", default=160, type=int)
    parser.add_argument("--out", default="data", type=Path)
    args = parser.parse_args()

    parquet = download(Path("/tmp/hindi_instruct.parquet"))
    pairs = extract_pairs(parquet, args.max_reply, args.max_prompt)
    print(f"{len(pairs)} usable unique pairs")

    random.Random(0).shuffle(pairs)
    pairs = pairs[: args.target]
    held = pairs[: args.held_out]
    train = pairs[args.held_out :]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, split in (("chat_train.jsonl", train), ("chat_heldout.jsonl", held)):
        path = args.out / name
        with path.open("w", encoding="utf-8") as handle:
            for row in split:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(split)} pairs to {path}")


if __name__ == "__main__":
    main()
