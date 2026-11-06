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
from tokenizer import CharTokenizer, language_tag  # noqa: E402
from bpe_tokenizer import BPETokenizer, load_any  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pretrain", "chat"], required=True)
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char",
                   help="bpe gives roughly 6x more Hindi text per window at "
                        "the same parameter count; char is the old default")
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
    p.add_argument("--chat-lr-scale", default=0.5, type=float,
                   help="chat lr as a fraction of --lr. Was hardcoded at 0.5.")
    p.add_argument("--replay-fraction", default=0.0, type=float,
                   help="fraction of each chat batch drawn from --replay-text. "
                        "Measured need: the chat stage cost +0.68 nats of "
                        "general Hindi, because nothing of the pretraining "
                        "distribution was replayed.")
    p.add_argument("--min-answer-chars", default=8, type=int,
                   help="drop chat pairs whose answer is shorter than this; a "
                        "one-word answer teaches nothing and was 0.7%% of the "
                        "corpus at <10 chars")
    p.add_argument("--content-filter", action="store_true",
                   help="drop text matching the blocklist in "
                        "rules/content_blocklist.json before training")
    p.add_argument("--blocklist", default=Path("rules/content_blocklist.json"),
                   type=Path)
    p.add_argument("--replay-text", default=None, type=Path,
                   help="pretraining corpus to replay from during chat")
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
    p.add_argument("--cross-links", action="store_true",
                   help="each block also sees a running average of all earlier layer "
                        "outputs; no new parameters")
    p.add_argument("--recurrent", default=1, type=int,
                   help="apply the block stack N times with a step embedding; buys "
                        "effective depth for recurrent x dim parameters")
    p.add_argument("--distinct-blocks", action="store_true",
                   help="one block per layer; combines with --drift")
    p.add_argument("--drift", default=0, type=int,
                   help="shared waves giving every layer its own gain")
    p.add_argument("--init-from", default=None, type=Path,
                   help="checkpoint to start from. Without this the chat stage looks "
                        "for <out>/pretrain.pt and silently starts from random weights "
                        "if it is absent, which is how two 'fine-tunes' trained from "
                        "scratch and lost to the model they were meant to improve")
    p.add_argument("--checkpoint-every", default=200, type=int,
                   help="how often to write a checkpoint; 25 was wasteful for large models")
    return p.parse_args()


def build_model(tokenizer: CharTokenizer, args) -> HindiLM:
    return HindiLM(tokenizer.size, args.dim, args.layers, args.heads, args.ffn,
                   args.seq_len, share_blocks=args.share_blocks,
                   multi_level=args.multi_level, criticality=args.criticality,
                   cross_links=getattr(args, "cross_links", False),
                   recurrent=getattr(args, "recurrent", 1),
                   drift=getattr(args, "drift", 0),
                   distinct=getattr(args, "distinct_blocks", False))


def load_model(path: Path, tokenizer: CharTokenizer, args,
               required: bool = False) -> HindiLM:
    model = build_model(tokenizer, args)
    if not path.exists():
        if required:
            raise SystemExit(
                f"--init-from {path} does not exist. Refusing to train from random "
                f"weights when a starting checkpoint was requested.")
        print(f"  no checkpoint at {path}: training from random weights")
        return model
    state = torch.load(path, weights_only=False)
    try:
        model.load_state_dict(state["model"])
        print(f"  resumed from {path}")
    except RuntimeError as exc:
        if required:
            raise SystemExit(
                f"--init-from {path} is incompatible with this configuration: "
                f"{str(exc).splitlines()[0][:120]}")
        print(f"  checkpoint incompatible with this configuration, starting fresh "
              f"({str(exc).splitlines()[0][:80]})")
    return model


def save(model: HindiLM, tokenizer: CharTokenizer, path: Path, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, path)
    tokenizer.save(path.parent / "tokenizer.json")


def load_blocklist(path: Path) -> list[str]:
    import json as _json
    if not path.is_file():
        return []
    d = _json.loads(path.read_text(encoding="utf-8"))
    terms = list(d.get("terms", []))
    return [t.lower() for t in terms if t]


def filter_text(text: str, terms: list[str]) -> tuple[str, bool]:
    """Drop a document containing a blocked term. Returns (text, dropped)."""
    low = text.lower()
    for t in terms:
        if t in low:
            return text, True
    return text, False


def combined_loss(logits, aux_logits, hidden_scale, x_targets, tokenizer, args,
                  weights=None, token_mask=None):
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
    if token_mask is not None:
        # supervise the answer only. The question is handed to the model at
        # inference, so predicting it teaches a task it never has to do -- and
        # the audit showed that was 50% of all tokens, i.e. half the gradient.
        mask = mask * token_mask.float().reshape(-1)
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
        if args.content_filter:
            terms = load_blocklist(args.blocklist)
            kept_lines, dropped_lines = [], 0
            for line in text.splitlines():
                _, bad = filter_text(line, terms)
                if bad:
                    dropped_lines += 1
                else:
                    kept_lines.append(line)
            text = "\n".join(kept_lines)
            print(f"  content filter: dropped {dropped_lines:,} lines "
                  f"matching {len(terms)} blocked terms")

        if args.tokenizer == "bpe":
            # 4096 total: 256 bytes + merges + the 13 specials
            tokenizer = BPETokenizer.train(text, vocab_size=4096)
            print(f"  bpe tokenizer: vocab {tokenizer.size}")
        else:
            tokenizer = CharTokenizer.train(text)
        ids = np.array(tokenizer.encode(text), dtype=np.int64)
        model = load_model(args.out / "pretrain.pt", tokenizer, args)
        report = model.parameters_report()
        print(f"vocab {tokenizer.size}  characters {len(text):,}  "
              f"parameters {report['parameters']:,}")
        print(f"  tokens/parameter {len(text)/max(report['parameters'],1):.2f}")

        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rng = np.random.default_rng(0)
        stats_p: dict[str, int] = {}
        started = time.time()
        step = 0
        # The clock budget must bound this loop. The chat stage already did that,
        # but pretrain used args.steps directly, so a 60-minute budget stopped
        # after 600 steps and 141 seconds and 58 minutes went unused. The
        # deadline is checked every step; the step count is only a ceiling.
        deadline = None
        step_ceiling = args.steps
        if args.clock_budget_minutes > 0:
            deadline = started + args.clock_budget_minutes * 60
            step_ceiling = 10**9
        while step < step_ceiling:
            if deadline is not None and time.time() > deadline:
                print(f"  clock budget reached at step {step}", flush=True)
                break
            step += 1
            model.train()
            starts = rng.integers(0, len(ids) - args.seq_len - 1, size=args.batch)
            batch = np.stack([ids[s : s + args.seq_len] for s in starts])
            targets = np.stack([ids[s + 1 : s + args.seq_len + 1] for s in starts])
            x = torch.from_numpy(batch)
            y = torch.from_numpy(targets)
            optimiser.zero_grad()
            logits, aux_logits, hidden = model(x, return_aux=True)
            loss, _ = combined_loss(logits, aux_logits, hidden, y, tokenizer, args)
            if not torch.isfinite(loss):
                # Skip the step and say so. Continuing would write NaN into every
                # weight, and the run would finish and save a checkpoint that
                # cannot produce anything.
                stats_p["nan_skipped"] = stats_p.get("nan_skipped", 0) + 1
                if stats_p["nan_skipped"] in (1, 10, 100):
                    print(f"  non-finite loss at step {step}; step skipped "
                          f"({stats_p['nan_skipped']} so far)", flush=True)
                optimiser.zero_grad()
                continue
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
        tokenizer = load_any(tokenizer_path)
        print("  loaded tokenizer from pretrain")
    else:
        corpus = "\n".join(
            json.loads(line)["user"] + "\n" + json.loads(line)["assistant"]
            for line in Path(args.chat).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        # honour the requested tokenizer here too: falling back to char silently
        # is how a --tokenizer bpe run trained a character model and nobody noticed
        if args.content_filter:
            terms = load_blocklist(args.blocklist)
            kept_lines, dropped_lines = [], 0
            for line in text.splitlines():
                _, bad = filter_text(line, terms)
                if bad:
                    dropped_lines += 1
                else:
                    kept_lines.append(line)
            text = "\n".join(kept_lines)
            print(f"  content filter: dropped {dropped_lines:,} lines "
                  f"matching {len(terms)} blocked terms")

        if args.tokenizer == "bpe":
            tokenizer = BPETokenizer.train(corpus, vocab_size=4096)
            print(f"  built bpe tokenizer from chat data: vocab {tokenizer.size}")
        else:
            tokenizer = CharTokenizer.train(corpus)
            print(f"  built tokenizer from chat data: vocab {tokenizer.size}")
        tokenizer.save(tokenizer_path)

    if args.init_from is not None:
        model = load_model(args.init_from, tokenizer, args, required=True)
        print(f"  chat stage started from {args.init_from}")
    else:
        model = load_model(args.out / "pretrain.pt", tokenizer, args)
    pairs = [json.loads(line) for line in
             Path(args.chat).read_text(encoding="utf-8").splitlines() if line.strip()]
    before = len(pairs)
    if args.content_filter:
        terms = load_blocklist(args.blocklist)
        pairs = [p for p in pairs
                 if not filter_text(p["user"] + " " + p["assistant"], terms)[1]]
        print(f"  content filter: {before - len(pairs):,} pairs dropped")
    short = [p for p in pairs if len(p["assistant"].strip()) < args.min_answer_chars]
    if short:
        pairs = [p for p in pairs
                 if len(p["assistant"].strip()) >= args.min_answer_chars]
        print(f"  dropped {len(short):,} pairs with answers under "
              f"{args.min_answer_chars} characters")

    report = model.parameters_report()
    print(f"{len(pairs)} chat pairs, vocab {tokenizer.size}, "
          f"parameters {report['parameters']:,}")

    def encode_pair(pair: dict) -> tuple[list[int], int]:
        """Return the ids and the index where the answer starts.

        Everything before that index is context: the model reads it and is never
        scored on reproducing it.
        """
        head = (
            [tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
            + tokenizer.encode(pair["user"])
            + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"]]
            + [tokenizer.stoi[language_tag(pair["assistant"])]]
        )
        body = tokenizer.encode(pair["assistant"]) + [tokenizer.eos]
        return head + body, len(head)

    encoded = [encode_pair(p) for p in pairs]
    sequences = [ids for ids, _ in encoded]
    answer_starts = [start for _, start in encoded]
    supervised = sum(len(ids) - start for ids, start in encoded)
    total_tokens = sum(len(ids) for ids in sequences)
    print(f"  supervision: {supervised:,} of {total_tokens:,} tokens "
          f"({supervised/max(total_tokens,1):.1%}) -- the answer only")

    if args.reward_weighting:
        weights = [0.5 + 0.5 * score_hindi(p["assistant"]).score / 100.0 for p in pairs]
        print(f"  reward weighting on: mean weight {sum(weights)/len(weights):.3f}")
    else:
        weights = [1.0] * len(pairs)

    stats = {"dropped_long": 0, "replay": 0}

    # Replay pool. The chat stage sees only instruction pairs, so the model's
    # distribution narrows to them and general Hindi degrades -- measured at
    # +0.68 nats on held-out sentences. Mixing a fraction of pretraining text
    # back in is the standard remedy and was simply missing.
    replay_windows: list[list[int]] = []
    if args.replay_fraction > 0:
        if args.replay_text is None or not Path(args.replay_text).is_file():
            raise SystemExit(
                "--replay-fraction was set but --replay-text is missing or does "
                "not exist; refusing to train without the data it asks for")
        raw = Path(args.replay_text).read_text(encoding="utf-8", errors="replace")
        raw = raw[: args.text_chars] if args.text_chars else raw
        ids = tokenizer.encode(raw)
        step = max(args.seq_len // 2, 1)
        replay_windows = [ids[i:i + args.seq_len + 1]
                          for i in range(0, max(len(ids) - args.seq_len - 1, 1), step)]
        print(f"  replay pool: {len(replay_windows):,} windows from "
              f"{args.replay_text} ({len(ids):,} tokens)")

    # hippocampal buffer: a small, fast store that keeps recent experiences and
    # replays them alongside fresh ones, so new material does not overwrite old
    hippocampus: list[int] = []

    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr * args.chat_lr_scale)
    print(f"  chat lr {args.lr * args.chat_lr_scale:.2e} "
          f"({args.chat_lr_scale:g} x {args.lr:.2e})")
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
            masks = []
            kept: list[int] = []
            dropped_this_batch = 0
            for index in picks:
                row = sequences[index][: args.seq_len]
                start = answer_starts[index]
                # If the answer begins past the window the mask is all zeros:
                # the pair contributes no gradient, and with per-pair weighting
                # it also averages a 0.0 into the loss, dragging it down and
                # diluting every other pair in the batch. Neither is visible in
                # the log. Such a pair is now skipped and counted.
                if start >= args.seq_len:
                    dropped_this_batch += 1
                    m = None
                else:
                    m = [0] * min(start, args.seq_len) + \
                        [1] * max(0, min(len(sequences[index]), args.seq_len) - start)
                    m = m[: args.seq_len]
                if m is None or sum(m) == 0:
                    continue
                rows.append(row + [tokenizer.pad] * (args.seq_len - len(row)))
                masks.append(m + [0] * (args.seq_len - len(m)))
                kept.append(int(index))
            if args.replay_fraction > 0 and replay_windows:
                take = int(args.batch * args.replay_fraction)
                for _ in range(take):
                    w = replay_windows[int(rng.integers(0, len(replay_windows)))]
                    row = list(w[: args.seq_len])
                    rows.append(row + [tokenizer.pad] * (args.seq_len - len(row)))
                    # every token is supervised: this is language modelling, not
                    # instruction following, so there is no question to mask
                    masks.append([1] * len(row) + [0] * (args.seq_len - len(row)))
                    stats["replay"] += 1
            if dropped_this_batch:
                stats["dropped_long"] += dropped_this_batch
            if not rows:
                continue
                rows.append(row + [tokenizer.pad] * (args.seq_len - len(row)))
                masks.append(m + [0] * (args.seq_len - len(m)))
            batch = np.array(rows, dtype=np.int64)
            mask_batch = np.array(masks, dtype=np.int64)
            x = torch.from_numpy(batch[:, :-1])
            y = torch.from_numpy(batch[:, 1:])
            # y is the sequence shifted by one, so its mask is shifted too
            y_mask = torch.from_numpy(mask_batch[:, 1:])
            optimiser.zero_grad()
            logits, aux_logits, hidden = model(x, return_aux=True)
            n_replay = len(rows) - len(kept)
            batch_weights = torch.tensor(
                [weights[i] for i in kept] + [1.0] * max(n_replay, 0))
            loss, parts = combined_loss(logits, aux_logits, hidden, y, tokenizer,
                                        args, weights=batch_weights,
                                        token_mask=y_mask)
            if not torch.isfinite(loss):
                # Skip the step and say so. Continuing would write NaN into every
                # weight, and the run would finish and save a checkpoint that
                # cannot produce anything.
                stats["nan_skipped"] = stats.get("nan_skipped", 0) + 1
                if stats["nan_skipped"] in (1, 10, 100):
                    print(f"  non-finite loss at step {step_offset + done + 1}; "
                          f"step skipped ({stats['nan_skipped']} so far)",
                          flush=True)
                optimiser.zero_grad()
                continue
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
            if global_step % max(args.checkpoint_every, 1) == 0:
                print(f"  step {global_step}  loss {last:.4f}  "
                      f"{time.time() - started:.0f}s", flush=True)
                save(model, tokenizer, args.out / "chat.pt", args)
        return time.time() - started, done

    budget, planned = bind_to_clock(args, measure=lambda: run_steps(15))
    target = planned or args.target_steps or args.steps
    print(f"running {target:,} steps")
    remaining = max(target - 15, 0) if planned else target
    elapsed, done = run_steps(remaining, step_offset=15 if planned else 0)
    total_steps = 15 + done if planned else done
    print(f"completed {total_steps:,} steps in {elapsed:.0f}s")
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
        "tokens_per_second": round(
            total_steps * args.batch * args.seq_len / max(elapsed, 1e-6), 1),
        "threads": args.threads,
        "seq_len": args.seq_len,
    }
    if model.controller is not None:
        brain["criticality"] = model.controller.report()
    Path(args.out / "brain_report.json").write_text(json.dumps(brain, indent=2))
    print("brain report:", json.dumps(brain["mechanisms"]))

    save(model, tokenizer, args.out / "chat.pt", args)


if __name__ == "__main__":
    main()
