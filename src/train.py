"""Pretrain on Hindi text, then fine-tune on the chat seed set.

    python src/train.py --stage pretrain --steps 600
    python src/train.py --stage chat --steps 600
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clock import make_budget  # noqa: E402
from grammar_reward import score_hindi  # noqa: E402
from model import HindiLM  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pretrain", "chat"], required=True)
    p.add_argument("--text", default="hindi_corpus.txt", type=Path)
    p.add_argument("--chat", default="data/chat_train.jsonl", type=Path)
    p.add_argument("--out", default="runs", type=Path)
    p.add_argument("--dim", default=192, type=int)
    p.add_argument("--layers", default=4, type=int)
    p.add_argument("--heads", default=4, type=int)
    p.add_argument("--ffn", default=768, type=int)
    p.add_argument("--seq-len", default=192, type=int)
    p.add_argument("--batch", default=8, type=int)
    p.add_argument("--steps", default=600, type=int)
    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument("--threads", default=2, type=int)
    p.add_argument("--time-budget", default=0.0, type=float)
    p.add_argument("--target-steps", default=0, type=int,
                   help="step ceiling; the clock budget may lower it")
    p.add_argument("--clock-budget-minutes", default=0.0, type=float,
                   help="bind the run to wall-clock time at the measured CPU clock")
    return p.parse_args()


def load_model(path: Path, tokenizer: CharTokenizer, args) -> HindiLM:
    model = HindiLM(tokenizer.size, args.dim, args.layers, args.heads, args.ffn, args.seq_len)
    if path.exists():
        state = torch.load(path, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"  resumed from {path}")
    return model


def save(model: HindiLM, tokenizer: CharTokenizer, path: Path, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, path)
    tokenizer.save(path.parent / "tokenizer.json")


def bind_to_clock(args, measure=None):
    """Measure the machine, then decide how many steps actually fit.

    `measure` is a callable that runs a handful of steps and returns the elapsed
    seconds; it is supplied by each training stage so the measurement uses the
    real model, real batch size and real data path.
    """
    if args.clock_budget_minutes <= 0:
        return None, args.target_steps or None
    budget = make_budget(args.clock_budget_minutes * 60)
    print(f"clock {budget.clock_ghz:.3f} GHz ({budget.clock_source}), "
          f"budget {args.clock_budget_minutes:.0f} min = "
          f"{budget.cycles / 1e9:.0f} Gcycles")
    if measure is not None:
        elapsed, steps_done = measure()
        rate = steps_done / max(elapsed, 1e-6)
        planned = budget.plan(args.target_steps or 10**9, rate)
        print(f"measured {rate:.2f} steps/s over {steps_done} steps "
              f"-> {planned:,} steps fit the budget")
        return budget, planned
    return budget, args.target_steps or None


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage == "pretrain":
        if not Path(args.text).exists():
            raise SystemExit(f"no corpus at {args.text}")
        text = Path(args.text).read_text(encoding="utf-8")[:2_000_000]
        if len(text) < 5000:
            print(f"corpus too small ({len(text)} chars); skipping pretrain")
            return
        tokenizer = CharTokenizer.train(text)
        print(f"vocab {tokenizer.size}")
        ids = np.array(tokenizer.encode(text), dtype=np.int64)
        model = load_model(args.out / "pretrain.pt", tokenizer, args)
        print(f"model parameters {sum(p.numel() for p in model.parameters()):,}")

        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = np.random.default_rng(0)
        started = time.time()
        for step in range(1, args.steps + 1):
            model.train()
            starts = rng.integers(0, len(ids) - args.seq_len - 1, size=args.batch)
            batch = np.stack([ids[s : s + args.seq_len] for s in starts])
            targets = np.stack([ids[s + 1 : s + args.seq_len + 1] for s in starts])
            x = torch.from_numpy(batch)
            y = torch.from_numpy(targets)
            optimiser.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, tokenizer.size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            if step % 100 == 0:
                print(f"  step {step}  loss {loss.item():.4f}  "
                      f"ppl {math.exp(loss.item()):.1f}  {time.time()-started:.0f}s", flush=True)
                save(model, tokenizer, args.out / "pretrain.pt", args)
            if args.time_budget and time.time() - started > args.time_budget:
                break
        save(model, tokenizer, args.out / "pretrain.pt", args)
        return

    # chat fine-tune — self-sufficient: if no pretrain artefacts exist (a fresh
    # CI checkout has none), build the tokenizer from the chat data and start
    # from random weights instead of failing
    tokenizer_path = args.out / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = CharTokenizer.load(tokenizer_path)
        print("  loaded tokenizer from pretrain")
    else:
        corpus = "\n".join(
            json.loads(line)["user"] + "\n" + json.loads(line)["assistant"]
            for line in Path(args.chat).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        tokenizer = CharTokenizer.train(corpus)
        tokenizer.save(tokenizer_path)
        print(f"  built tokenizer from chat data: vocab {tokenizer.size}")
    model = load_model(args.out / "pretrain.pt", tokenizer, args)
    pairs = [json.loads(line) for line in Path(args.chat).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{len(pairs)} chat pairs, vocab {tokenizer.size}")

    def language_tag(text: str) -> str:
        dev = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
        lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        if dev and dev >= lat:
            return "<hi>"
        if lat and dev == 0:
            words = {w.lower() for w in text.replace("?", " ").split()}
            markers = words & {"hai", "hain", "ka", "ki", "ke", "ko", "se", "me",
                               "kya", "mujhe", "aap", "nahi", "kare", "karo"}
            return "<mix>" if markers else "<en>"
        return "<mix>"

    def encode_pair(pair: dict) -> list[int]:
        return (
            [tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
            + tokenizer.encode(pair["user"])
            + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"]]
            + [tokenizer.stoi[language_tag(pair["assistant"])]]
            + tokenizer.encode(pair["assistant"])
            + [tokenizer.eos]
        )

    sequences = [encode_pair(p) for p in pairs]

    # reward weighting: a pair whose reply scores badly on the grammar reward
    # contributes less to the gradient, without being removed entirely
    if getattr(args, "reward_weighting", False):
        weights = []
        for pair in pairs:
            score = score_hindi(pair["assistant"]).score
            weights.append(0.5 + 0.5 * score / 100.0)
        print(f"  reward weighting on: mean weight {sum(weights)/len(weights):.3f}")
    else:
        weights = [1.0] * len(pairs)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5)
    rng = np.random.default_rng(1)

    def run_steps(count: int, step_offset: int = 0) -> tuple[float, int]:
        """Run `count` optimisation steps; return (elapsed, steps actually run)."""
        started = time.time()
        last_loss = 0.0
        done = 0
        for _ in range(count):
            model.train()
            picks = rng.integers(0, len(sequences), size=args.batch)
            batch_weights = torch.tensor([weights[int(i)] for i in picks],
                                         dtype=torch.float32)
            rows = []
            for index in picks:
                row = sequences[index][: args.seq_len]
                rows.append(row + [tokenizer.pad] * (args.seq_len - len(row)))
            batch = np.array(rows, dtype=np.int64)
            x = torch.from_numpy(batch[:, :-1])
            y = torch.from_numpy(batch[:, 1:])
            optimiser.zero_grad()
            logits = model(x)
            per_token = F.cross_entropy(
                logits.reshape(-1, tokenizer.size), y.reshape(-1),
                ignore_index=tokenizer.pad, reduction="none")
            per_token = per_token.view(len(picks), -1)
            mask = (y != tokenizer.pad).float()
            per_pair = (per_token * mask).sum(1) / mask.sum(1).clamp(min=1)
            loss = (per_pair * batch_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            last_loss = float(loss.item())
            done += 1
            global_step = step_offset + done
            if global_step % 25 == 0:
                print(f"  step {global_step}  loss {last_loss:.4f}  "
                      f"{time.time() - started:.0f}s", flush=True)
                save(model, tokenizer, args.out / "chat.pt", args)
        return time.time() - started, done

    # measure the machine on the real model, then plan the run
    budget, planned = bind_to_clock(
        args, measure=lambda: run_steps(15)
    )
    target = planned or args.target_steps or args.steps
    print(f"running {target:,} steps")
    remaining = max(target - 15, 0) if planned else target
    elapsed, done = run_steps(remaining, step_offset=15 if planned else 0)
    print(f"completed {15 + done:,} steps in {elapsed:.0f}s")

    if budget is not None:
        report = budget.as_dict()
        report["steps_completed"] = 15 + done
        Path(args.out / "clock_report.json").write_text(
            json.dumps(report, indent=2)
        )
        print("clock report:", report)

    save(model, tokenizer, args.out / "chat.pt", args)


if __name__ == "__main__":
    main()
