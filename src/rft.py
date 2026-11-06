"""Rejection-sampling fine-tuning: turn the rewards into weights.

The rewards so far only score text after it exists, and the self-correction loop
only filters at answer time. Neither changes the model. This script does:

    1. ask the model for several replies to each prompt
    2. score every reply on path and form (and on agreement with a reference
       when the prompt has one)
    3. keep only the best reply, and only if it clears the threshold
    4. write the survivors as a chat corpus

Training on that corpus is what moves the reward into the weights: the model
learns from its own good outputs rather than from a rule written by hand.

If too few prompts survive, the honest output is an empty file and a clear
number saying so -- not a quietly degraded training set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint import build_from_checkpoint  # noqa: E402
from grammar_reward import score_hindi  # noqa: E402
from meaning_reward import score_against_reference  # noqa: E402
from judge import judge  # noqa: E402
from reasoning import parse  # noqa: E402
from lessons import LessonStore, best_fragment
from rule_engine import check as check_rules  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def sample(model, tokenizer, cfg, prompt: str, temperature: float, top_k: int,
           max_new: int, seed: int) -> str:
    torch.manual_seed(seed)
    ids = ([tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
           + tokenizer.encode(prompt)
           + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"],
              tokenizer.stoi.get("<hi>", tokenizer.eos)])
    tokens = torch.tensor([ids])
    with torch.no_grad():
        for _ in range(max_new):
            if tokens.shape[1] >= cfg["seq_len"]:
                break
            logits = model(tokens)[:, -1, :] / temperature
            values, _ = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
            logits[logits < values[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, nxt], dim=1)
            if int(nxt) == tokenizer.eos:
                break
    return tokenizer.decode(tokens[0, len(ids):].tolist())


def _norm(text: str) -> str:
    """Coarse normalisation so near-identical replies collapse to one key."""
    import re
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def score(prompt: str, reply: str, reference: str | None) -> tuple[float, dict]:
    """Gate first, score second. A reply that fails any gate scores at most 30,
    so a wrong answer cannot average its way past the threshold on good form."""
    verdict = judge(prompt, reply, reference)
    if verdict.passed and not check_rules(reply).passed:
        verdict.passed = False
        verdict.reasons.append('rules: grammar violation')
    return verdict.score, {
        "path": verdict.path,
        "form": verdict.form,
        "answer": verdict.answer,
        "passed": verdict.passed,
        "notes": verdict.reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/chat.pt", type=Path)
    parser.add_argument("--prompts", default="data/multi_heldout.jsonl", type=Path)
    parser.add_argument("--out", default="data/rft_train.jsonl", type=Path)
    parser.add_argument("--report", default="runs/rft_report.json", type=Path)
    parser.add_argument("--limit", default=800, type=int)
    parser.add_argument("--candidates", default=4, type=int)
    parser.add_argument("--threshold", default=60.0, type=float)
    parser.add_argument("--temperature", default=0.85, type=float)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--max-new", default=60, type=int)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--lessons", default="data/lessons.jsonl", type=Path,
                        help="persistent record of flaws, deduplicated and counted")
    parser.add_argument("--fragments", default="data/analysis/fragments.jsonl", type=Path,
                        help="best span from each rejected reply, for deeper analysis")
    parser.add_argument("--fragment-mode", action="store_true",
                        help="keep only the best span of the winning reply, not the whole reply")
    parser.add_argument("--dedupe-against", default="data/multi_train.jsonl", type=Path,
                        help="skip a fragment that already exists in this corpus")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    tokenizer = CharTokenizer.load(args.checkpoint.parent / "tokenizer.json")
    model, cfg = build_from_checkpoint(args.checkpoint, tokenizer)

    rows = [json.loads(line) for line in
            args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.limit]

    lessons = LessonStore(args.lessons)
    fragments = []
    # existing knowledge, so a fragment is only added if it is genuinely new
    existing = set()
    if args.dedupe_against.exists():
        for line in args.dedupe_against.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing.add(_norm(row.get("assistant", "")))
    print(f"dedupe against {len(existing)} existing replies")

    kept, tried, scores = [], 0, []
    for i, row in enumerate(rows):
        prompt = row["user"]
        reference = row.get("assistant")
        best = None
        for c in range(args.candidates):
            raw = sample(model, tokenizer, cfg, prompt, args.temperature,
                         args.top_k, args.max_new, seed=i * 100 + c)
            tried += 1
            # a reply with no <ans> block never answered, so it cannot be kept
            parsed = parse(raw)
            reply = parsed.answer if parsed.complete else ""
            combined, detail = score(prompt, reply, reference)
            if not parsed.complete:
                detail["notes"] = list(detail["notes"]) + ["no <ans> block"]
            if best is None or combined > best[0]:
                best = (combined, reply, detail)
            else:
                # a rejected reply is not uniformly bad: keep the span that still
                # scores, and write the flaw down once with a count
                frag = best_fragment(prompt, reply)
                for note in detail["notes"]:
                    lessons.record(note, prompt, reply, fragment=frag)
                fragments.append({"prompt": prompt, "reply": reply,
                                  "score": round(combined, 1),
                                  "gates_passed": detail.get("passed", False),
                                  "best_fragment": frag, "flaws": detail["notes"]})
        scores.append(best[0])
        if best[0] >= args.threshold and best[2].get("passed", False):
            text = best[1]
            if args.fragment_mode:
                # the best span of the best reply: higher precision, less noise
                frag = best_fragment(prompt, text)
                if frag:
                    text = frag
            key = _norm(text)
            if key in existing:
                continue  # already known, adding it again would only reinforce
            existing.add(key)
            kept.append({"user": prompt, "assistant": text,
                         "reference": reference,
                         "score": round(best[0], 1), "source": "rft"})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} prompts, kept {len(kept)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n",
                        encoding="utf-8")

    report = {
        "prompts": len(rows),
        "candidates_per_prompt": args.candidates,
        "generations": tried,
        "kept": len(kept),
        "kept_fraction": round(len(kept) / max(len(rows), 1), 3),
        "threshold": args.threshold,
        "mean_best_score": round(sum(scores) / max(len(scores), 1), 1),
        "max_best_score": round(max(scores), 1) if scores else None,
        "fragment_mode": bool(args.fragment_mode),
        "existing_replies_compared": len(existing),
        "rejected_analysed": len(fragments),
        "rejected_by_gate": sum(1 for f in fragments if not f.get("gates_passed", True)),
        "lessons": lessons.summary(),
    }
    lessons.save()
    if fragments:
        args.fragments.parent.mkdir(parents=True, exist_ok=True)
        args.fragments.write_text(
            "\n".join(json.dumps(f, ensure_ascii=False) for f in fragments) + "\n",
            encoding="utf-8")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not kept:
        print("\nNo reply cleared the threshold. Training on this file would teach "
              "nothing, so the run should stop here rather than continue.")


if __name__ == "__main__":
    main()
