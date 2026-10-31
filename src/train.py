"""Pretrain on Hindi text, then fine-tune on the chat corpus.

    python src/train.py --stage pretrain --steps 600
    python src/train.py --stage chat --steps 600

Five brain-inspired mechanisms can be switched on independently, so each one can
be attributed when the numbers move:

    --share-blocks     one block applied N times (ALBERT), fewer parameters
    --multi-level      every block predicts the next token (predictive coding)
    --sparsity L       L1 penalty on hidden activations (sparse coding)
    --criticality      keep the branching ratio near 1.0 (Beggs & Plenz)
    --hippocampus      replay buffer mixed into every batch (CLS theory)
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
    p.add_argument("--text-chars", default=2_000_000, type=int,
                   help="how much of the corpus to use; kept equal to what the "
                        "budget can cover so one full pass is possible")
    # brain-inspired switches
    p.add_argument("--share-blocks", action="store_true")
    p.add_argument("--multi-level", action="store_true")
    p.add_argument("--sparsity", default=0.0, type=float,
                   help="weight of the L1 penalty on hidden activations")
    p.add_argument("--aux-weight", default=0.3, type=float)
    p.add_argument("--criticality", action="store_true")
    p.add_argument("--hippocampus", action="store_true")
    p.add_argument("--hippocampus-fraction", default=0.25, type=float)
    p.add_argument("--reward-weighting", action="store_true")
    return p.parse_args()


def build_model(tokenizer: CharTokenizer, args) -> HindiLM:
    return HindiLM(tokenizer.size, args.dim, args.layers, args.heads, args.ffn,
                   args.seq_len, share_blocks=args.share_blocks,
                   multi_level=args.multi_level, criticality=args.criticality)


def load_model(path: Path, tokenizer: CharTokenizer, args) -> HindiLM:
    model = build_model(tokenizer, args)
    if path.exists():
        state = torch.load(path, weights_only=False)
        try:
            model.load_state_dict(state["model"])
            print(f"  resumed from {path}")
        except RuntimeError as exc:
            print(f"  checkpoint incompatible with this configuration, starting fresh "
                  f"({str(exc).splitlines()[0][:80]})")
    return model


def save(model: HindiLM, tokenizer: CharTokenizer, path: Path, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, path)
    tokenizer.save(path.parent / "tokenizer.json")


def combined_loss(logits, aux_logits, hidden_scale, x_targets, tokenizer, args,
                  weights=None):
    """Main next-token loss plus the two brain-inspired penalties.

    multi-level: each block's own prediction is scored, so the gradient reaches
    every depth without travelling the whole network (predictive coding).
    sparsity:    an L1 term on hidden activation magnitude, which is what makes a
    representation sparse rather than merely small (sparse coding).
    """
    per_token = F.cross_entropy(
        logits.reshape(-1, tokenizer.size), x_targets.reshape(-1),
        ignore_index=tokenizer.pad, reduction="none")
    mask = (x_targets != tokenizer.pad).float().reshape(-1)
    if weights is None:
        loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
    else:
        per_token = per_token.view(len(weights), -1)
        per_pair = (per_token * mask.view(len(weights), -1)).sum(1) / \
                   mask.view(len(weights), -1).sum(1).clamp(min=1)
        loss = (per_pair * weights).mean()

    parts = {"main": float(loss.item())}

    if aux_logits:
        aux = 0.0
        for layer_logits in aux_logits:
            aux = aux + F.cross_entropy(
                layer_logits.reshape(-1, tokenizer.size), x_targets.reshape(-1),
                ignore_index=tokenizer.pad)
        aux = aux / len(aux_logits)
        loss = loss + args.aux_weight * aux
        parts["aux"] = float(aux.item())

    if args.sparsity > 0 and hidden_scale:
        penalty = sum(hidden_scale) / len(hidden_scale)
        loss = loss + args.sparsity * penalty
        parts["sparsity"] = float(penalty)

    parts["total"] = float(loss.item())
    return loss, parts


def bind_to_clock(args, measure=None):
    if args.clock_budget_minutes <= 0:
        return None, args.target_steps or None
    budget = make_budget(args.clock_budget_minutes * 60)
    print(f"clock {budget.clock_ghz:.3f} GHz ({budget.clock_source}), "
          f"budget {args.clock_budget_minutes:.0f} min")
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
        text = Path(args.text).read_text(encoding="utf-8")[: args.text_chars]
        if len(text) < 5000:
            raise SystemExit(f"corpus too small ({len(text)} chars); refusing to "
                             f"pretrain on a stub")
        tokenizer = CharTokenizer.train(text)
        ids = np.array(tokenizer.encode(text), dtype=np.int64)
        model = load_model(args.out / "pretrain.pt", tokenizer, args)
        report = model.parameters_report()
        print(f"vocab {tokenizer.size}  characters {len(text):,}  "
              f"parameters {report['parameters']:,}")
        print(f"  tokens/parameter {len(text)/max(report['parameters'],1):.2f}")

        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = np.random.default_rng(0)
        started = time.time()
        step = 0
        while step < args.steps:
            step += 1
            model.train()
            starts = rng.integers(0, len(ids) - args.seq_len - 1, size=args.batch)
            batch = np.stack([ids[s : s + args.seq_len] for s in starts])
            targets = np.stack([ids[s + 1 : s + args.seq_len + 1] for s in starts])
            x = torch.from_numpy(batch)
            y = torch.from_numpy(targets)
            optimiser.zero_grad()
            logits, aux_logits, hidden = model(x, return_aux=args.multi_level)
            loss, _ = combined_loss(logits, aux_logits, hidden, y, tokenizer, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            if model.controller is not None:
                for index in range(len(model.blocks)):
                    model.controller.update(index)
            if step % 100 == 0:
                print(f"  step {step}  loss {loss.item():.4f}  "
                      f"ppl {math.exp(loss.item()):.1f}  {time.time()-started:.0f}s",
                      flush=True)
                save(model, tokenizer, args.out / "pretrain.pt", args)
            if args.time_budget and time.time() - started > args.time_budget:
                break
        save(model, tokenizer, args.out / "pretrain.pt", args)
        if model.controller is not None:
            Path(args.out / "criticality.json").write_text(
                json.dumps(model.controller.report(), indent=2))
        return

    # ---------- chat fine-tune ----------
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
    pairs = [json.loads(line) for line in
             Path(args.chat).read_text(encoding="utf-8").splitlines() if line.strip()]
    report = model.parameters_report()
    print(f"{len(pairs)} chat pairs, vocab {tokenizer.size}, "
          f"parameters {report['parameters']:,}")

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

    if args.reward_weighting:
        weights = [0.5 + 0.5 * score_hindi(p["assistant"]).score / 100.0 for p in pairs]
        print(f"  reward weighting on: mean weight {sum(weights)/len(weights):.3f}")
    else:
        weights = [1.0] * len(pairs)

    # hippocampal buffer: a small, fast store that keeps recent experiences and
    # replays them alongside fresh ones, so new material does not overwrite old
    hippocampus: list[int] = []

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5)
    rng = np.random.default_rng(1)
    loss_log: list[float] = []

    def run_steps(count: int, step_offset: int = 0) -> tuple[float, int]:
        started = time.time()
        last = 0.0
        done = 0
        for _ in range(count):
            model.train()
            picks = rng.integers(0, len(sequences), size=args.batch)
            if args.hippocampus and hippocampus:
                take = int(args.batch * args.hippocampus_fraction)
                replay = rng.choice(len(hippocampus), size=take, replace=False)
                picks = np.concatenate([picks[: args.batch - take],
                                        np.array([hippocampus[i] for i in replay])])
            rows = []
            for index in picks:
                row = sequences[index][: args.seq_len]
                rows.append(row + [tokenizer.pad] * (args.seq_len - len(row)))
            batch = np.array(rows, dtype=np.int64)
            x = torch.from_numpy(batch[:, :-1])
            y = torch.from_numpy(batch[:, 1:])
            optimiser.zero_grad()
            logits, aux_logits, hidden = model(x, return_aux=args.multi_level)
            batch_weights = torch.tensor([weights[int(i)] for i in picks])
            loss, parts = combined_loss(logits, aux_logits, hidden, y, tokenizer,
                                        args, weights=batch_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            if model.controller is not None:
                for index in range(len(model.blocks)):
                    model.controller.update(index)
            if args.hippocampus:
                for index in picks:
                    if index not in hippocampus:
                        hippocampus.append(int(index))
                if len(hippocampus) > 4000:
                    hippocampus.pop(0)
            last = float(loss.item())
            loss_log.append(last)
            done += 1
            global_step = step_offset + done
            if global_step % 25 == 0:
                print(f"  step {global_step}  loss {last:.4f}  "
                      f"{time.time() - started:.0f}s", flush=True)
                save(model, tokenizer, args.out / "chat.pt", args)
        return time.time() - started, done

    budget, planned = bind_to_clock(args, measure=lambda: run_steps(15))
    target = planned or args.target_steps or args.steps
    print(f"running {target:,} steps")
    remaining = max(target - 15, 0) if planned else target
    elapsed, done = run_steps(remaining, step_offset=15 if planned else 0)
    print(f"completed {15 + done:,} steps in {elapsed:.0f}s")

    total_steps = 15 + done
    per_step = elapsed / max(done, 1)
    eta = {
        "steps_completed": total_steps,
        "seconds": round(elapsed, 1),
        "seconds_per_step": round(per_step, 3),
        "pairs": len(sequences),
        "batch": args.batch,
        "epochs_completed": round(total_steps * args.batch / max(len(sequences), 1), 2),
        "projection": {
            f"{n}_epochs": {
                "steps": int(n * len(sequences) / args.batch),
                "minutes": round(n * len(sequences) / args.batch * per_step / 60, 1),
            } for n in (1, 3, 10)
        },
    }
    Path(args.out / "eta.json").write_text(json.dumps(eta, indent=2))
    print("ETA:", json.dumps(eta["projection"]))

    if budget is not None:
        clock = budget.as_dict()
        clock["steps_completed"] = total_steps
        Path(args.out / "clock_report.json").write_text(json.dumps(clock, indent=2))

    brain = {
        "mechanisms": {
            "share_blocks": args.share_blocks,
            "multi_level_prediction": args.multi_level,
            "sparsity_weight": args.sparsity,
            "criticality": args.criticality,
            "hippocampus": args.hippocampus,
            "reward_weighting": args.reward_weighting,
        },
        "parameters": report["parameters"],
        "loss_first_50_mean": round(float(np.mean(loss_log[:50])), 4) if loss_log else None,
        "loss_last_50_mean": round(float(np.mean(loss_log[-50:])), 4) if loss_log else None,
        "hippocampus_size": len(hippocampus),
    }
    if model.controller is not None:
        brain["criticality"] = model.controller.report()
    Path(args.out / "brain_report.json").write_text(json.dumps(brain, indent=2))
    print("brain report:", json.dumps(brain["mechanisms"]))

    save(model, tokenizer, args.out / "chat.pt", args)


if __name__ == "__main__":
    main()
