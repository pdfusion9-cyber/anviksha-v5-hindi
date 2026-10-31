"""Build a multi-source Hindi instruction corpus with an unseen-source test set.

Training sources (mixed):
    smangrul/hindi_instruct_v1        (MIT)
    ai4bharat dolly/hi                (translated Dolly)
    ai4bharat anudesh/hi              (human-written Hindi)

Unseen test source (never trained on):
    ai4bharat hh-rlhf/hi              (translated Anthropic dialogue)

Mixing sources is the point: the previous run memorised one corpus and scored
13.79 perplexity on anything else. Holding out a whole source means the unseen
number stays honest no matter how much training data is added.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1/resolve/main"
CACHE = Path("cache")
SMANGRUL = ("https://huggingface.co/datasets/smangrul/hindi_instruct_v1/"
            "resolve/main/data/train-00000-of-00001.parquet")


def download(url: str, name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        subprocess.run(["curl", "-sL", url, "-o", str(path)], check=True)
    return path


def from_messages(path: Path, source: str, max_prompt: int, max_reply: int, cap: int = 12000) -> list[dict]:
    """Extract pairs; kept in memory only for one source at a time."""
    pairs, seen = [], set()
    handle = pq.ParquetFile(path)
    for batch in handle.iter_batches(batch_size=128, columns=["messages"]):
        for row in batch.to_pylist():
            messages = row["messages"]
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            user, assistant = user.strip(), assistant.strip()
            if not (4 <= len(user) <= max_prompt and 4 <= len(assistant) <= max_reply):
                continue
            key = user[:100]
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"user": user, "assistant": assistant, "source": source})
            if len(pairs) >= cap:
                return pairs
    return pairs


def from_columns(path: Path, source: str, max_prompt: int, max_reply: int, cap: int = 12000) -> list[dict]:
    """Handles both column namings in use: instruction/response and inputs/targets."""
    pairs, seen = [], set()
    handle = pq.ParquetFile(path)
    names = set(handle.schema_arrow.names)
    if {"instruction", "response"} <= names:
        prompt_col, reply_col = "instruction", "response"
    elif {"inputs", "targets"} <= names:
        prompt_col, reply_col = "inputs", "targets"
    else:
        raise KeyError(f"no known prompt/reply columns in {sorted(names)}")
    for batch in handle.iter_batches(batch_size=128, columns=[prompt_col, reply_col]):
        for row in batch.to_pylist():
            user = (row[prompt_col] or "").strip()
            assistant = (row[reply_col] or "").strip()
            if not (4 <= len(user) <= max_prompt and 4 <= len(assistant) <= max_reply):
                continue
            key = user[:100]
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"user": user, "assistant": assistant, "source": source})
            if len(pairs) >= cap:
                return pairs
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", type=Path)
    parser.add_argument("--max-prompt", default=100, type=int)
    parser.add_argument("--max-reply", default=140, type=int)
    parser.add_argument("--heldout", default=500, type=int)
    parser.add_argument("--cap", default=12000, type=int, help="max pairs per source")
    args = parser.parse_args()

    extractors = {
        "smangrul": (from_messages, download(SMANGRUL, "smangrul.parquet")),
        "dolly": (from_columns, download(f"{BASE}/dolly/hi-00000-of-00001.parquet", "dolly.parquet")),
        "anudesh": (from_messages, download(f"{BASE}/anudesh/hi-00000-of-00001.parquet", "anudesh.parquet")),
        "flan_v2": (from_columns, download(f"{BASE}/flan_v2/hi-00000-of-00001.parquet", "flan.parquet")),
    }

    # one source at a time, appended to disk, so peak memory stays flat
    cache_dir = CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_source: dict[str, int] = {}
    for name, (extractor, path) in extractors.items():
        try:
            pairs = extractor(path, name, args.max_prompt, args.max_reply, args.cap)
        except Exception as exc:  # noqa: BLE001 — one bad source must not kill the run
            print(f"{name}: skipped ({type(exc).__name__}: {exc})")
            per_source[name] = 0
            (cache_dir / f"{name}.jsonl").write_text("", encoding="utf-8")
            continue
        per_source[name] = len(pairs)
        print(f"{name}: {len(pairs)} usable pairs")
        with (cache_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        del pairs

    pool: list[dict] = []
    seen: set[str] = set()
    for name in extractors:
        for line in (cache_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pair = json.loads(line)
            key = pair["user"][:100]
            if key in seen:
                continue
            seen.add(key)
            pool.append(pair)
    print(f"combined and deduplicated: {len(pool)} pairs")

    random.Random(0).shuffle(pool)
    held = pool[: args.heldout]
    train = pool[args.heldout :]

    unseen = from_messages(download(f"{BASE}/hh-rlhf/hi-00000-of-00001.parquet", "hh.parquet"),
                           "hh-rlhf", args.max_prompt, args.max_reply)
    print(f"unseen source hh-rlhf: {len(unseen)} pairs fit the window")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("multi_train.jsonl", train), ("multi_heldout.jsonl", held),
                       ("unseen_hh.jsonl", unseen)):
        path = args.out / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} -> {path}")

    manifest = {
        "per_source": per_source,
        "train": len(train), "heldout": len(held), "unseen_hh_rlhf": len(unseen),
        "window": [args.max_prompt, args.max_reply],
    }
    (args.out / "multi_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
