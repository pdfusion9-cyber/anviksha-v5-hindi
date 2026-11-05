"""Measure the model on the data it was trained on.

Every evaluation so far has used held-out questions, and the answer has been
zero. That tells us the model cannot generalise, but it does not tell us why:
a model that never learned its own training answers has a capacity problem, and
a model that reproduces them perfectly but fails on anything new has a
memorisation problem. Those need opposite responses, and the held-out score
cannot distinguish them.

Two measurements, because they fail differently:

  teacher-forced   feed the exact training sequence and ask whether the model's
                   top-1 prediction is the right next token at each position.
                   This has no sampling noise -- it is the model's own
                   probability, read directly -- so it isolates "did it learn
                   this pattern at all".

  free generation  prompt with the question and let it write. This is what the
                   user sees, so it is the honest end-to-end number, but it
                   carries temperature and length noise.

Reported separately for the question span and the answer span, because the
answer-only loss mask means the model was never trained to predict questions.
A high question score and a low answer score would mean the mask is not working.

If teacher-forced accuracy on training data is low, the model has not fitted
even what it was shown, and no amount of held-out evaluation will be
informative until that is fixed.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from checkpoint import build_from_checkpoint  # noqa: E402
from bpe_tokenizer import load_any  # noqa: E402
from tokenizer import language_tag  # noqa: E402


def find_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    found = sorted(Path("runs").rglob(path.name)) if Path("runs").is_dir() else []
    if found:
        return found[0]
    raise SystemExit(f"no checkpoint at {path} and none under runs/")


def load(path: Path):
    ckpt = find_checkpoint(path)
    tok_path = ckpt.parent / "tokenizer.json"
    if not tok_path.is_file():
        raise SystemExit(f"no tokenizer.json beside {ckpt}")
    tokenizer = load_any(tok_path)
    model, cfg = build_from_checkpoint(ckpt, tokenizer)
    model.eval()
    return model, tokenizer, cfg, ckpt


def build_sequence(tokenizer, question: str, answer: str) -> tuple[list[int], int]:
    """The exact format train.py uses, and the index where the answer starts."""
    head = ([tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
            + tokenizer.encode(question)
            + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"]]
            + [tokenizer.stoi[language_tag(question)]])
    body = tokenizer.encode(answer) + [tokenizer.eos]
    return head + body, len(head)


def teacher_forced(model, tokenizer, cfg, question, answer, seq_len):
    """Top-1 accuracy per position, split into question and answer spans."""
    ids, start = build_sequence(tokenizer, question, answer)
    ids = ids[:seq_len]
    if len(ids) < 8 or start >= len(ids) - 1:
        return None
    x = torch.tensor([ids[:-1]])
    y = torch.tensor(ids[1:])
    with torch.no_grad():
        logits = model(x)[0]
    pred = logits.argmax(dim=-1)
    correct = (pred == y).tolist()
    # position i predicts token i+1, so the answer span in target space starts
    # at start-1
    q_end = max(0, min(start - 1, len(correct)))
    q_hits = correct[:q_end]
    a_hits = correct[q_end:]
    return {
        "question_tokens": len(q_hits),
        "question_acc": sum(q_hits) / len(q_hits) if q_hits else None,
        "answer_tokens": len(a_hits),
        "answer_acc": sum(a_hits) / len(a_hits) if a_hits else None,
    }


def free_generate(model, tokenizer, cfg, question, max_new, temperature, top_k,
                  seed):
    seq_len = cfg.get("seq_len", 192)
    head = ([tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
            + tokenizer.encode(question)
            + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"],
               tokenizer.stoi[language_tag(question)]])
    if len(head) + 8 >= seq_len:
        return ""
    torch.manual_seed(seed)
    tokens = torch.tensor([head])
    with torch.no_grad():
        for _ in range(max_new):
            if tokens.shape[1] >= seq_len:
                break
            logits = model(tokens)[:, -1, :] / temperature
            values, _ = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
            logits[logits < values[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, nxt], dim=1)
            tid = int(nxt)
            if tid == tokenizer.eos:
                break
            # a special token other than <eos> in the answer position means the
            # model has collapsed. Continuing would spend the whole budget
            # repeating it and then decode to "", which reads as "said nothing".
            if tid >= tokenizer.base if hasattr(tokenizer, "base") else tid >= 256:
                break
    ids = tokens[0, len(head):].tolist()
    if ids and ids[-1] != tokenizer.eos and ids[-1] >= 256:
        ids = ids[:-1]
    return tokenizer.decode(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/big/chat.pt"))
    ap.add_argument("--pretrain", type=Path, default=Path("runs/big/pretrain.pt"),
                    help="optional, to show what the chat stage added")
    ap.add_argument("--chat", type=Path, default=Path("data/multi_train.jsonl"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("runs/distill.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            args.chat.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[: args.limit]
    print(f"measuring on {len(rows)} training pairs\n")

    results = {}
    for label, path in (("chat", args.checkpoint), ("pretrain", args.pretrain)):
        if not path.is_file() and not list(Path("runs").rglob(path.name)):
            print(f"{label}: no checkpoint at {path}, skipped")
            continue
        model, tokenizer, cfg, ckpt = load(path)
        seq_len = cfg.get("seq_len", 192)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"=== {label}: {ckpt} ({n_params:,} parameters) ===")

        tf_q, tf_a, gen_sim, exact = [], [], [], 0
        samples = []
        for i, row in enumerate(rows):
            q = row.get("user") or row.get("question") or ""
            a = row.get("assistant") or row.get("answer") or ""
            if not q or not a:
                continue
            tf = teacher_forced(model, tokenizer, cfg, q, a, seq_len)
            if tf:
                if tf["question_acc"] is not None:
                    tf_q.append(tf["question_acc"])
                if tf["answer_acc"] is not None:
                    tf_a.append(tf["answer_acc"])
            gen = free_generate(model, tokenizer, cfg, q, args.max_new,
                                args.temperature, args.top_k, seed=i)
            sim = difflib.SequenceMatcher(None, gen.strip(), a.strip()).ratio()
            gen_sim.append(sim)
            if gen.strip() == a.strip():
                exact += 1
            if len(samples) < args.show:
                samples.append((q, a, gen, tf))

        def mean(xs):
            return round(sum(xs) / len(xs), 3) if xs else None

        report = {
            "checkpoint": str(ckpt),
            "parameters": n_params,
            "pairs": len(gen_sim),
            "teacher_forced_question_acc": mean(tf_q),
            "teacher_forced_answer_acc": mean(tf_a),
            "free_generation_similarity": mean(gen_sim),
            "exact_match": exact,
            "exact_match_rate": round(exact / max(len(gen_sim), 1), 3),
        }
        results[label] = report
        print(json.dumps(report, indent=2))

        for q, a, gen, tf in samples:
            print(f"\n  Q: {q[:110]}")
            print(f"  trained answer : {a[:150]}")
            print(f"  model wrote    : {gen[:150]!r}")
            if tf and tf["answer_acc"] is not None:
                print(f"  answer top-1 accuracy: {tf['answer_acc']:.1%}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    print("\nreading: teacher_forced_answer_acc near 0 means the model never fitted")
    print("its own training answers -- a capacity limit, and held-out evaluation")
    print("cannot be informative until that changes. High teacher-forced with low")
    print("free generation means it learned the distribution but cannot sample from")
    print("it reliably, which is a decoding and calibration problem, not capacity.")


if __name__ == "__main__":
    main()
