"""End-to-end: how many questions come back 100% correct.

The pipeline answers from a checked source when it can, lets the model write when
it cannot, and refuses when a check fails. This measures which of those actually
happened, and -- the number that matters -- how often a retrieved answer was
wrong. A retrieval hit that is wrong is worse than a refusal, because it arrives
with the confidence of a stored fact.

Reported per question:

    kb_correct    matched a stored answer, and that answer is the right one
    kb_wrong      matched a stored answer, and it is the wrong one   <- the danger
    model_correct wrote an answer, and the rules and judge accepted it, and it matches
    model_wrong   wrote an answer, checks passed, but it does not match
    refused       nothing was returned, which is safe but not an answer

"Correct" means the returned answer matches the held-out gold answer by the same
meaning score the judge uses, at or above the bar the judge accepts. It is not
graded by the model's own opinion of itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_pipeline import KnowledgeBase, answer  # noqa: E402
from meaning_reward import score_against_reference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="data/multi_train.jsonl", type=Path)
    parser.add_argument("--test", default="data/multi_heldout.jsonl", type=Path)
    parser.add_argument("--limit", default=300, type=int)
    parser.add_argument("--threshold", default=0.62, type=float,
                        help="similarity needed before a stored answer is trusted")
    parser.add_argument("--out", default="runs/e2e.json", type=Path)
    parser.add_argument("--samples", default=12, type=int)
    parser.add_argument("--sweep", default="",
                        help="comma separated thresholds to compare, e.g. 0.62,0.75,0.85")
    args = parser.parse_args()

    kb = KnowledgeBase.load(args.kb)
    test = [json.loads(line) for line in
            args.test.read_text(encoding="utf-8").splitlines() if line.strip()]
    test = test[: args.limit]
    print(f"knowledge base: {len(kb.rows)} checked answers")
    print(f"held-out questions: {len(test)}")
    print(f"trust threshold: {args.threshold}\n")

    # Two questions have to be answered separately, because they mean different
    # things:
    #   known   -- a question the knowledge base actually contains. Here the
    #              pipeline should return the stored answer, and it should be
    #              right. This is the "100% correct" claim and it is testable.
    #   unseen  -- a question the base does not contain. The pipeline should
    #              refuse rather than return a similar-looking wrong answer.
    import random
    rng = random.Random(0)
    known = rng.sample(kb.rows, min(args.limit, len(kb.rows)))

    counts = {"kb_correct": 0, "kb_wrong": 0, "model_correct": 0,
              "model_wrong": 0, "refused": 0}
    worst = []
    shown = 0

    known_counts = {"correct": 0, "wrong": 0, "refused": 0}
    for row in known:
        result = answer(row["user"], kb, generate=None,
                        match_threshold=args.threshold)
        if result.via == "knowledge_base" and result.text == row["assistant"]:
            known_counts["correct"] += 1
        elif result.via == "knowledge_base":
            known_counts["wrong"] += 1
            worst.append({"question": row["user"][:100],
                          "returned": (result.text or "")[:100],
                          "gold": row["assistant"][:100],
                          "similarity": result.confidence})
        else:
            known_counts["refused"] += 1

    for row in test:
        gold = row.get("assistant", "")
        result = answer(row["user"], kb, generate=None,
                        match_threshold=args.threshold)

        if result.via == "knowledge_base":
            ok = score_against_reference(result.text, gold).total >= 60 if gold else False
            counts["kb_correct" if ok else "kb_wrong"] += 1
            if not ok:
                worst.append({"question": row["user"][:100],
                              "returned": result.text[:100],
                              "gold": gold[:100],
                              "similarity": result.confidence})
        elif result.via == "model":
            ok = score_against_reference(result.text, gold).total >= 60 if gold else False
            counts["model_correct" if ok else "model_wrong"] += 1
        else:
            counts["refused"] += 1

        if shown < args.samples:
            shown += 1
            print(f"Q: {row['user'][:66]}")
            print(f"   via={result.via:14} confidence={result.confidence:.2f}")
            if result.text:
                print(f"   A: {result.text[:76]}")
                print(f"   gold: {gold[:70]}")
            else:
                print(f"   refused: {'; '.join(result.violations[:1])[:70]}")
            print()

    if args.sweep:
        # The threshold decides the trade: too low and a lookalike question gets
        # someone else's answer, too high and real questions go unanswered. Which
        # point is right is a measurement, not an opinion, so measure all of them
        # on the same sample and report coverage against wrong answers.
        thresholds = [float(t) for t in args.sweep.split(",") if t.strip()]
        frontier = []
        print("\n" + "=" * 62)
        print("threshold sweep")
        print(f"{'thresh':>7} {'known ok':>9} {'known %':>8} "
              f"{'unseen ok':>10} {'unseen WRONG':>13}")
        for th in thresholds:
            kc = sum(1 for row in known
                     if answer(row["user"], kb, generate=None,
                               match_threshold=th).text == row["assistant"])
            uw = 0
            uc = 0
            for row in test:
                res = answer(row["user"], kb, generate=None, match_threshold=th)
                if res.via == "knowledge_base":
                    ok = score_against_reference(res.text, row.get("assistant", "")).total >= 60
                    if ok:
                        uc += 1
                    else:
                        uw += 1
            frontier.append({"threshold": th,
                             "known_correct": kc,
                             "known_rate": round(kc / max(len(known), 1), 3),
                             "unseen_correct": uc,
                             "unseen_wrong": uw})
            print(f"{th:>7.2f} {kc:>9} {kc / max(len(known), 1):>7.1%} "
                  f"{uc:>10} {uw:>13}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        (args.out.parent / "e2e_sweep.json").write_text(
            json.dumps(frontier, indent=2), encoding="utf-8")
        safe = [f for f in frontier if f["unseen_wrong"] == 0]
        best = max(safe, key=lambda f: f["known_rate"]) if safe else None
        print("\nsafest threshold with zero wrong answers:",
              best["threshold"] if best else "none of those tried")
        if best:
            print(f"  at {best['threshold']}: {best['known_rate']:.1%} of known "
                  f"questions answered correctly, {best['unseen_wrong']} wrong")

    total = sum(counts.values())
    answered = counts["kb_correct"] + counts["kb_wrong"] + \
        counts["model_correct"] + counts["model_wrong"]
    correct = counts["kb_correct"] + counts["model_correct"]
    known_total = sum(known_counts.values())
    report = {
        "known_in_kb": {
            "asked": known_total,
            **known_counts,
            "correct_rate": round(known_counts["correct"] / max(known_total, 1), 3),
        },
        "prompts": total,
        **counts,
        "answered": answered,
        "correct": correct,
        "refused_rate": round(counts["refused"] / max(total, 1), 3),
        # of everything that was actually returned, how much was right
        "precision_when_answered": round(correct / answered, 3) if answered else None,
        "wrong_returned": counts["kb_wrong"] + counts["model_wrong"],
        "threshold": args.threshold,
        "kb_size": len(kb.rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 62)
    print(json.dumps(report, indent=2))
    print("=" * 62)
    if worst:
        print(f"\n{len(worst)} retrieved answers were wrong -- these are the dangerous ones:")
        for w in worst[:5]:
            print(f"  Q: {w['question'][:70]}")
            print(f"     returned: {w['returned'][:70]}")
            print(f"     gold    : {w['gold'][:70]}")
    else:
        print("\nno retrieved answer was wrong")

    if report["wrong_returned"] > 0:
        print("\nA wrong answer was returned. The threshold is too low or the rules "
              "are too weak; raising the threshold trades coverage for safety.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
