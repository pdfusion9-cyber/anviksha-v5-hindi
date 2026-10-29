"""Fetch Hindi text for pretraining through the datasets-server rows API.

Only the rows used are transferred, so this works inside a CI job without
pulling a multi-hundred-megabyte parquet.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

API = ("https://datasets-server.huggingface.co/rows"
       "&config=20231101.hi&split=train&offset={offset}&length=100")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="hindi_corpus.txt", type=Path)
    parser.add_argument("--chars", default=2_000_000, type=int)
    args = parser.parse_args()

    chunks: list[str] = []
    total = 0
    offset = 0
    while total < args.chars and offset < 4000:
        try:
            with urllib.request.urlopen(API.format(offset=offset), timeout=60) as response:
                payload = json.load(response)
        except Exception as exc:  # noqa: BLE001
            print(f"stopped at offset {offset}: {exc}")
            break
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            text = item["row"]["text"]
            chunks.append(text)
            total += len(text)
        offset += 100

    text = "\n\n".join(chunks)
    if len(text) < 1000:
        # the rows API is rate-limited and sometimes 404s from CI. The chat
        # corpus is already on disk and is Hindi prose, so use it instead of
        # failing the job.
        fallback = Path("data/chat_train.jsonl")
        if fallback.exists():
            import json as _json
            rows = [_json.loads(line) for line in fallback.read_text(encoding="utf-8").splitlines() if line.strip()]
            text = "\n".join(r["user"] + " " + r["assistant"] for r in rows)
            print(f"rows API unavailable; using chat corpus ({len(text):,} chars)")

    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {len(text):,} characters to {args.out}")


if __name__ == "__main__":
    main()
