"""Rewrite the chat corpus so every reply carries a reasoning trace.

Every pair becomes

    user: <question>
    assistant: <think> ...plan... </think> <ans> ...answer... </ans>

so the model is never shown an example of answering without thinking. The trace
is derived from the question's own shape (reasoning.build_trace), which keeps it
honest: it states what kind of answer the question needs, and when a reference
exists it states the target too.

The output also records the bare answer, so evaluation can score the answer and
not the trace.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reasoning import build_trace  # noqa: E402

TRACE_PREFIX = "<think>"
TRACE_SUFFIX = "</think>"
ANS_PREFIX = "<ans>"
ANS_SUFFIX = "</ans>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/multi_train.jsonl", type=Path)
    parser.add_argument("--heldout", default="data/multi_heldout.jsonl", type=Path)
    parser.add_argument("--out-train", default="data/reason_train.jsonl", type=Path)
    parser.add_argument("--out-heldout", default="data/reason_heldout.jsonl", type=Path)
    parser.add_argument("--keep-answer", action="store_true", default=True,
                        help="also write the bare answer under 'answer' for scoring")
    args = parser.parse_args()

    def convert(path: Path, out: Path) -> int:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row["user"].strip()
            answer = row["assistant"].strip()
            if not prompt or not answer:
                continue
            trace = build_trace(prompt, answer)
            rows.append({
                "user": prompt,
                "assistant": f"{TRACE_PREFIX}{trace}{TRACE_SUFFIX}"
                             f"{ANS_PREFIX}{answer}{ANS_SUFFIX}",
                "answer": answer,
                "source": row.get("source", "unknown"),
            })
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                       encoding="utf-8")
        return len(rows)

    n_train = convert(args.train, args.out_train)
    n_held = convert(args.heldout, args.out_heldout)
    print(f"train pairs with a trace: {n_train}")
    print(f"held-out pairs with a trace: {n_held}")
    if n_train == 0:
        raise SystemExit("no training pairs written; the corpus is empty")

    sample = json.loads(args.out_train.read_text(encoding="utf-8").splitlines()[0])
    print("\nsample:")
    print(f"  user      : {sample['user'][:80]}")
    print(f"  assistant : {sample['assistant'][:150]}")
    print(f"  answer    : {sample['answer']}")


if __name__ == "__main__":
    main()
