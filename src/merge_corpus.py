"""Merge every cached source into one corpus, with the held-out split.

Reads whatever `<source>.jsonl` files exist in the cache, deduplicates across
them, and writes train/held-out. Kept separate from the fetchers so a source can
be added or dropped without re-downloading the rest.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache", type=Path)
    parser.add_argument("--out", default="data", type=Path)
    parser.add_argument("--heldout", default=1000, type=int)
    parser.add_argument("--exclude", default="hh,unseen", type=str,
                        help="comma separated source names to keep out of training")
    args = parser.parse_args()

    skip = {s.strip() for s in args.exclude.split(",") if s.strip()}
    files = sorted(args.cache.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no source files in {args.cache}")

    pool: list[dict] = []
    seen: set[str] = set()
    per_source: dict[str, int] = {}
    for path in files:
        name = path.stem
        if any(s in name for s in skip):
            continue
        kept = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["user"][:100]
            if key in seen:
                continue
            seen.add(key)
            row["source"] = row.get("source", name)
            pool.append(row)
            kept += 1
        per_source[name] = kept

    print("per source:")
    for name, n in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"  {name:16} {n:>7}")

    random.Random(0).shuffle(pool)
    held = pool[: args.heldout]
    train = pool[args.heldout:]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "multi_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
        encoding="utf-8")
    (args.out / "multi_heldout.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in held) + "\n",
        encoding="utf-8")

    stats = {"per_source": per_source, "total_unique": len(pool),
             "train": len(train), "heldout": len(held)}
    (args.out / "corpus_stats.json").write_text(json.dumps(stats, indent=2),
                                                encoding="utf-8")
    print(json.dumps({"total_unique": len(pool), "train": len(train),
                      "heldout": len(held)}, indent=2))
    if len(train) < 20000:
        print(f"\nNOTE: only {len(train)} training pairs; the target was 100,000+")


if __name__ == "__main__":
    main()
