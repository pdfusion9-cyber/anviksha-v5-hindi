"""Show what the model actually answers, by kind of question.

A single accuracy number hides everything that matters. The same pipeline can be
reliable on factual questions and hopeless on translations, and an average of the
two says nothing useful. This groups held-out questions by kind, runs the best
configuration, and prints real question/answer/gold triples so the behaviour can
be read rather than inferred from a score.

Kinds are detected from the question text with simple rules. The rules are crude
on purpose: they only need to be good enough to separate groups that behave
differently, and every group's count is printed so a misclassification is
visible rather than hidden.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_pipeline import KnowledgeBase, answer  # noqa: E402
from meaning_reward import score_against_reference  # noqa: E402

DEVANAGARI = re.compile(r"[\u0900-\u097f]")
NUMBER = re.compile(r"\d+")

TRANSLATE = re.compile(
    r"translate|anuvad|अनुवाद|हिंदी में|अंग्रेज़ी में|english mein|hindi mein", re.I)
ARITHMETIC = re.compile(
    r"[\d]+\s*[\+\-\*/x×]\s*[\d]+|calculate|गणना|जोड़|घटा|गुणा|भाग|sum of|"
    r"how much is", re.I)
CODE = re.compile(r"\bdef\b|\bclass\b|\bimport\b|function|python|code|कोड|"
                  r"program|प्रोग्राम|sql|regex", re.I)
LIST = re.compile(r"\blist\b|सूची|list out|नाम बताएं|enumerate|give me \d+", re.I)
DEFINE = re.compile(r"what is|what are|क्या है|क्या हैं|define|meaning of|"
                    r"explain|समझाएं|बताइए", re.I)
HOWTO = re.compile(r"how (do|to|can)|कैसे|steps|तरीका|प्रक्रिया", re.I)


def kind_of(question: str) -> str:
    if TRANSLATE.search(question):
        return "translation"
    if CODE.search(question):
        return "code"
    if ARITHMETIC.search(question):
        return "arithmetic"
    if LIST.search(question):
        return "list"
    if HOWTO.search(question):
        return "how-to"
    if DEFINE.search(question):
        return "factual"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, default=Path("data/multi_train.jsonl"))
    ap.add_argument("--test", type=Path, default=Path("data/multi_heldout.jsonl"))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--per-kind", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.92)
    ap.add_argument("--out", type=Path, default=Path("runs/answers_by_kind.json"))
    args = ap.parse_args()

    kb = KnowledgeBase.load(args.kb)
    rows = [json.loads(l) for l in
            args.test.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[: args.limit]
    print(f"knowledge base {len(kb.rows):,} rows, held-out {len(rows)}")
    print(f"threshold {args.threshold}\n")

    stats: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list] = defaultdict(list)

    for row in rows:
        k = kind_of(row["user"])
        gold = row.get("assistant", "")
        result = answer(row["user"], kb, generate=None,
                        match_threshold=args.threshold)

        if result.via == "knowledge_base":
            ok = bool(gold) and score_against_reference(result.text, gold).total >= 60
            stats[k]["answered"] += 1
            stats[k]["correct" if ok else "wrong"] += 1
            if len(examples[k]) < args.per_kind:
                examples[k].append({"verdict": "correct" if ok else "WRONG",
                                    "question": row["user"], "answer": result.text,
                                    "gold": gold, "confidence": result.confidence})
        else:
            stats[k]["refused"] += 1
            if len(examples[k]) < args.per_kind:
                examples[k].append({"verdict": "refused", "question": row["user"],
                                    "answer": None, "gold": gold,
                                    "why": (result.violations or ["no match"])[0]})

    print("=" * 78)
    print(f"{'kind':<13} {'asked':>6} {'answered':>9} {'correct':>8} {'wrong':>7} "
          f"{'refused':>8} {'accuracy':>9}")
    print("=" * 78)
    order = sorted(stats, key=lambda k: -sum(stats[k].values()))
    for k in order:
        c = stats[k]
        asked = sum(c.values())
        answered = c["answered"]
        acc = c["correct"] / answered if answered else 0.0
        print(f"{k:<13} {asked:>6} {answered:>9} {c['correct']:>8} {c['wrong']:>7} "
              f"{c['refused']:>8} {acc:>8.1%}")

    for k in order:
        print(f"\n{'=' * 78}\n{k.upper()}\n{'=' * 78}")
        for e in examples[k]:
            print(f"[{e['verdict'].upper()}] {e['question'][:88]}")
            if e["answer"]:
                print(f"   answer: {e['answer'][:88]}")
                print(f"   gold  : {e['gold'][:88]}")
                print(f"   confidence {e['confidence']:.2f}")
            else:
                print(f"   refused: {e['why'][:80]}")
            print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {k: dict(v) for k, v in stats.items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
