"""Measure what the rewards actually score, not just perplexity.

Perplexity says how well the model predicts the next character. It says nothing
about whether a reply is grounded, structured, or on the same path the answer
needs -- which is exactly what the path reward is for, and therefore what has to
be measured before and after fine-tuning on rejection-sampled data.

Generates one reply per prompt and reports the mean of each component, so a
change in path score cannot be hidden behind a change in perplexity.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint import build_from_checkpoint  # noqa: E402
from grammar_reward import score_hindi  # noqa: E402
from meaning_reward import score_against_reference  # noqa: E402
from judge import judge  # noqa: E402
from reasoning import parse  # noqa: E402
from path_reward import score_path  # noqa: E402
from tokenizer import CharTokenizer, language_tag  # noqa: E402


def sample(model, tokenizer, cfg, prompt, temperature, top_k, max_new, seed):
    torch.manual_seed(seed)
    ids = ([tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
           + tokenizer.encode(prompt)
           + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"],
              tokenizer.stoi[language_tag(prompt)]])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/chat.pt", type=Path)
    parser.add_argument("--prompts", default="data/multi_heldout.jsonl", type=Path)
    parser.add_argument("--out", default="runs/path_eval.json", type=Path)
    parser.add_argument("--limit", default=200, type=int)
    parser.add_argument("--max-new", default=80, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    tokenizer = CharTokenizer.load(args.checkpoint.parent / "tokenizer.json")
    model, cfg = build_from_checkpoint(args.checkpoint, tokenizer)

    rows = [json.loads(line) for line in
            args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = rows[: args.limit]

    paths, forms, answers, empties, verdicts = [], [], [], 0, []
    incomplete = 0
    for i, row in enumerate(rows):
        raw = sample(model, tokenizer, cfg, row["user"], args.temperature,
                     args.top_k, args.max_new, seed=i)
        # every reply must think first and answer inside <ans>; anything else
        # is an unfinished thought, not an answer
        parsed = parse(raw)
        reply = parsed.answer
        if not parsed.complete:
            incomplete += 1
        if len(reply.strip()) < 2:
            empties += 1
        paths.append(score_path(row["user"], reply).score)
        forms.append(score_hindi(reply).score)
        if row.get("assistant"):
            answers.append(score_against_reference(reply, row["assistant"]).total)
            verdicts.append(judge(row["user"], reply, row["assistant"]).passed)

    def mean(xs):
        return round(statistics.mean(xs), 2) if xs else None

    report = {
        "prompts": len(rows),
        "empty_replies": empties,
        "no_answer_block": incomplete,
        "answered": len(rows) - incomplete,
        "passed": sum(verdicts) if verdicts else 0,
        # only replies that actually produced an answer can fail; counting the
        # no-answer ones as failures hid the difference in run 23 (failed said
        # 150 when 93 of those never answered at all)
        "failed": (len(rows) - incomplete - sum(verdicts)) if verdicts else 0,
        "path_mean": mean(paths),
        "form_mean": mean(forms),
        "answer_mean": mean(answers),
        "path_ge_60": round(sum(1 for p in paths if p >= 60) / max(len(paths), 1), 3),
        "form_ge_60": round(sum(1 for f in forms if f >= 60) / max(len(forms), 1), 3),
        "strict_pass_rate": (round(sum(verdicts) / len(verdicts), 3)
                             if verdicts else None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
