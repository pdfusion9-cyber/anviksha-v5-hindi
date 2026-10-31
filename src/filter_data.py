"""Filter the training corpus with the reward functions, before training.

Two things are checked per pair:

    form     grammar score of the reply (Devanagari and Hinglish both accepted)
    shape    the reply must be a plausible answer, not a stub or a duplicate

Nothing here judges whether the answer is *true* — that needs a judge model this
project does not have. What it removes is the material that teaches the model
bad habits: English-only replies in a Hindi corpus, one-word stubs, repeated
text, and near-duplicate prompts.

    python src/filter_data.py --input data/multi_train.jsonl --out data/filtered
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from grammar_reward import score_hindi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/multi_train.jsonl", type=Path)
    parser.add_argument("--out", default="data/filtered", type=Path)
    parser.add_argument("--min-score", default=55, type=int)
    parser.add_argument("--min-chars", default=8, type=int)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    print(f"input pairs: {len(rows)}")

    kept, dropped = [], []
    reasons: Counter[str] = Counter()
    scores: list[int] = []
    scripts: Counter[str] = Counter()
    seen_prompts: set[str] = set()

    for row in rows:
        reply = row["assistant"]
        prompt = row["user"]
        result = score_hindi(reply)
        scores.append(result.score)
        scripts[result.script] += 1

        reason = None
        if len(reply.strip()) < args.min_chars:
            reason = "too_short"
        elif result.score < args.min_score:
            reason = f"low_grammar:{result.script}"
        elif prompt.strip()[:80] in seen_prompts:
            reason = "duplicate_prompt"

        if reason:
            reasons[reason] += 1
            dropped.append({**row, "filter_reason": reason, "grammar": result.score})
        else:
            seen_prompts.add(prompt.strip()[:80])
            kept.append({**row, "grammar": result.score, "script": result.script})

    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train_filtered.jsonl", kept), ("dropped.jsonl", dropped)):
        with (args.out / name).open("w", encoding="utf-8") as handle:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    scores.sort()
    stats = {
        "input": len(rows),
        "kept": len(kept),
        "dropped": len(dropped),
        "kept_pct": round(100 * len(kept) / max(len(rows), 1), 1),
        "drop_reasons": dict(reasons),
        "scripts": dict(scripts),
        "grammar_score": {
            "min": scores[0] if scores else 0,
            "median": scores[len(scores) // 2] if scores else 0,
            "max": scores[-1] if scores else 0,
        },
        "min_score_threshold": args.min_score,
    }
    (args.out / "filter_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
