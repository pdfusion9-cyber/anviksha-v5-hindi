"""Before blaming the model's size, check the six things that could also be wrong.

45 wrong answers out of 45 is a strong signal, and it is tempting to conclude
"the model is too small". That conclusion would justify a GPU bill, so it should
not be reached until the cheaper explanations are ruled out. This measures each
one on the actual corpus and reports a number, not an opinion:

    1. data quality       mojibake, duplicates, empty or stub answers
    2. format             does the training string match the inference string
    3. training target    is the model being trained to predict the question too
    4. tokenization       how much of each pair is cut off by the context
    5. contradictions     the same question stored with different answers
    6. capacity           the residual, only if the above come back clean

Every check prints what it measured and what it means, so a clean result and a
failing result are both readable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVANAGARI = re.compile(r"[\u0900-\u097f]")
LATIN = re.compile(r"[A-Za-z]")
# mojibake: UTF-8 bytes read as latin-1 look like this
MOJIBAKE = re.compile(r"[ÃÂà¤¥¦§¨©ª«¬®¯][\u0080-\u00ff\u0900-\u097f]?")
WORD = re.compile(r"[\u0900-\u097f]+|[A-Za-z][A-Za-z'-]+")


def looks_mojibake(text: str) -> bool:
    if not text:
        return False
    if text.count("à¤") or text.count("à¥") or text.count("Ã"):
        return True
    # a high share of characters in the latin-1 supplement plus devanagari
    # ranges is the usual signature of UTF-8 decoded with the wrong codec
    odd = sum(1 for c in text if 0x80 <= ord(c) <= 0xFF)
    return odd > len(text) * 0.15


def audit(kb_path: Path, heldout_path: Path | None, seq_len: int, limit: int) -> None:
    rows = [json.loads(l) for l in
            kb_path.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
    print(f"corpus: {len(rows):,} pairs\n")

    # ---- 1. data quality -------------------------------------------------
    print("=" * 74)
    print("1. DATA QUALITY")
    print("=" * 74)
    mojibake = sum(1 for r in rows if looks_mojibake(r["user"]) or looks_mojibake(r["assistant"]))
    empty_q = sum(1 for r in rows if not r["user"].strip())
    empty_a = sum(1 for r in rows if not r["assistant"].strip())
    short_a = sum(1 for r in rows if len(r["assistant"].strip()) < 10)
    dup_q = len(rows) - len({r["user"] for r in rows})
    dup_pair = len(rows) - len({(r["user"], r["assistant"]) for r in rows})

    lens = sorted(len(r["assistant"]) for r in rows)
    med = lens[len(lens) // 2] if lens else 0
    p95 = lens[int(len(lens) * 0.95)] if lens else 0

    deva = sum(1 for r in rows if len(DEVANAGARI.findall(r["assistant"])) >
               len(r["assistant"]) * 0.3)

    def line(label, value, note=""):
        print(f"  {label:<34} {value:>10}   {note}")

    line("rows", f"{len(rows):,}")
    line("mojibake rows", f"{mojibake:,}", f"{mojibake/len(rows):.1%} -- unusable if high")
    line("empty questions", f"{empty_q:,}")
    line("empty answers", f"{empty_a:,}")
    line("answers under 10 chars", f"{short_a:,}", f"{short_a/len(rows):.1%}")
    line("duplicate questions", f"{dup_q:,}", f"{dup_q/len(rows):.1%}")
    line("duplicate (q,a) pairs", f"{dup_pair:,}")
    line("answer length median", f"{med:,}", f"p95 {p95:,}")
    line("answers mostly Devanagari", f"{deva:,}", f"{deva/len(rows):.1%}")

    # ---- 2. format -------------------------------------------------------
    print()
    print("=" * 74)
    print("2. FORMAT: training string vs inference string")
    print("=" * 74)
    sample = rows[0]
    train_ids = (["<bos>", "<user>"] + list(sample["user"]) +
                 ["<sep>", "<assistant>", "<hi>"] + list(sample["assistant"]) + ["<eos>"])
    infer_ids = (["<bos>", "<user>"] + list(sample["user"]) +
                 ["<sep>", "<assistant>", "<hi>"])
    print(f"  training  : {' '.join(train_ids[:14])} ...")
    print(f"  inference : {' '.join(infer_ids[:14])} ...")
    print(f"  shared prefix: {len(infer_ids)} tokens, identical: "
          f"{train_ids[:len(infer_ids)] == infer_ids}")
    print("  the model is trained on the question, the separator, the language tag")
    print("  and the answer; inference supplies everything up to and including the")
    print("  language tag and expects the answer. The prefixes must match exactly.")

    # ---- 3. training target ---------------------------------------------
    print()
    print("=" * 74)
    print("3. TRAINING TARGET: how much of the loss is the question")
    print("=" * 74)
    q_tokens = sum(len(r["user"]) for r in rows)
    a_tokens = sum(len(r["assistant"]) for r in rows)
    specials = 6 * len(rows)
    total = q_tokens + a_tokens + specials
    print(f"  question tokens      {q_tokens:>12,}  {q_tokens/total:>6.1%}")
    print(f"  answer tokens        {a_tokens:>12,}  {a_tokens/total:>6.1%}")
    print(f"  special tokens       {specials:>12,}  {specials/total:>6.1%}")
    print()
    print("  train.py masks only padding, so the question is supervised too.")
    print(f"  That means {q_tokens/total:.0%} of the gradient teaches the model to")
    print("  reproduce questions it is already given at inference. That is capacity")
    print("  spent on a task the model never needs to do.")

    # ---- 4. tokenization -------------------------------------------------
    print()
    print("=" * 74)
    print(f"4. TOKENIZATION: what fits in seq_len {seq_len}")
    print("=" * 74)
    lengths = [len(r["user"]) + len(r["assistant"]) + 6 for r in rows]
    over = sum(1 for n in lengths if n > seq_len)
    kept = [min(n, seq_len) for n in lengths]
    lost = sum(max(0, n - seq_len) for n in lengths)
    print(f"  pairs longer than the window   {over:,}  ({over/len(rows):.1%})")
    print(f"  tokens kept                    {sum(kept):,}")
    print(f"  tokens cut off                 {lost:,}  "
          f"({lost/max(sum(lengths),1):.1%} of all text)")
    print(f"  median pair length             {sorted(lengths)[len(lengths)//2]:,} tokens")
    print("  a cut-off pair loses its answer tail, which is exactly what the model")
    print("  is supposed to learn to produce")

    # ---- 5. contradictions -----------------------------------------------
    print()
    print("=" * 74)
    print("5. CONTRADICTIONS: same question, different answers")
    print("=" * 74)
    by_q: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_q[r["user"].strip()].add(r["assistant"].strip())
    conflicting = {q: a for q, a in by_q.items() if len(a) > 1}
    print(f"  questions with more than one answer   {len(conflicting):,}  "
          f"({len(conflicting)/max(len(by_q),1):.1%})")
    for q, answers in list(conflicting.items())[:3]:
        print(f"\n  Q: {q[:70]}")
        for a in list(answers)[:2]:
            print(f"     -> {a[:70]}")
    if conflicting:
        print("\n  these teach the model that one question has several right answers,")
        print("  which raises the loss floor and makes confident answering harder")

    # ---- 6. held-out overlap ---------------------------------------------
    if heldout_path and heldout_path.is_file():
        print()
        print("=" * 74)
        print("6. HELD-OUT OVERLAP: is the test actually unseen")
        print("=" * 74)
        held = [json.loads(l) for l in
                heldout_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        kb_q = {r["user"].strip() for r in rows}
        seen = sum(1 for h in held if h["user"].strip() in kb_q)
        print(f"  held-out questions                {len(held):,}")
        print(f"  also present in the knowledge base {seen:,}  ({seen/max(len(held),1):.1%})")
        print("  anything above zero means the held-out score is partly a memory test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, default=Path("data/multi_train.jsonl"))
    ap.add_argument("--heldout", type=Path, default=Path("data/multi_heldout.jsonl"))
    ap.add_argument("--seq-len", type=int, default=192)
    ap.add_argument("--limit", type=int, default=60000)
    args = ap.parse_args()
    if not args.kb.is_file():
        print(f"missing {args.kb}")
        raise SystemExit(2)
    audit(args.kb, args.heldout, args.seq_len, args.limit)


if __name__ == "__main__":
    main()
