"""Fetch Hindi pretraining text at real scale.

The rows API is convenient but flaky from CI: it returned 404 and 429 on
different runs, and a silent fallback then produced a zero-length corpus, which
skipped pretraining entirely. That is how every earlier run ended up training a
40M-parameter model from scratch on 12k chat pairs.

So this script prefers the parquet files directly (238 MB of Hindi Wikipedia),
falls back to the rows API, and **fails loudly** if it ends up with too little
text. Pretending to pretrain is worse than not starting.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

PARQUETS = [
    ("https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
     "20231101.hi/train-00000-of-00002.parquet", "wiki_hi_0.parquet"),
    ("https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
     "20231101.hi/train-00001-of-00002.parquet", "wiki_hi_1.parquet"),
]
ROWS_API = ("https://datasets-server.huggingface.co/rows"
            "&config=20231101.hi&split=train&offset={offset}&length=100")
CACHE = Path("cache")


def download(url: str, name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists() or path.stat().st_size < 1_000_000:
        subprocess.run(["curl", "-sL", "--retry", "3", url, "-o", str(path)], check=True)
    return path


def from_parquet(target_chars: int) -> str:
    chunks: list[str] = []
    total = 0
    for url, name in PARQUETS:
        path = download(url, name)
        handle = pq.ParquetFile(str(path))
        for batch in handle.iter_batches(batch_size=256, columns=["text"]):
            for row in batch.to_pylist():
                text = row["text"]
                chunks.append(text)
                total += len(text)
            if total >= target_chars:
                return "\n\n".join(chunks)
    return "\n\n".join(chunks)


def from_rows_api(target_chars: int) -> str:
    chunks, total, offset = [], 0, 0
    while total < target_chars and offset < 20000:
        try:
            with urllib.request.urlopen(ROWS_API.format(offset=offset), timeout=60) as response:
                payload = json.load(response)
        except Exception as exc:  # noqa: BLE001
            print(f"rows API stopped at offset {offset}: {exc}")
            break
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            text = item["row"]["text"]
            chunks.append(text)
            total += len(text)
        offset += 100
    return "\n\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="hindi_corpus.txt", type=Path)
    parser.add_argument("--chars", default=40_000_000, type=int)
    parser.add_argument("--min-chars", default=5_000_000, type=int,
                        help="below this the script exits non-zero instead of pretending")
    args = parser.parse_args()

    text = ""
    try:
        text = from_parquet(args.chars)
        print(f"parquet source: {len(text):,} characters")
    except Exception as exc:  # noqa: BLE001
        print(f"parquet failed ({type(exc).__name__}: {exc}); trying rows API")
        text = from_rows_api(args.chars)
        print(f"rows API source: {len(text):,} characters")

    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {len(text):,} characters to {args.out}")

    if len(text) < args.min_chars:
        raise SystemExit(
            f"only {len(text):,} characters fetched, below the {args.min_chars:,} floor. "
            "Refusing to continue, because pretraining on a stub is what silently "
            "produced untrained models before.")


if __name__ == "__main__":
    main()
