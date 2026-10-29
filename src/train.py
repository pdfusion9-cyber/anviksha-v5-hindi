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

from model import HindiLM  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pretrain", "chat"], required=True)
    p.add_argument("--text", default="../hindi_corpus.txt", type=Path)
    p.add_argument("--chat", default="data/chat_seed.jsonl", type=Path)
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


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage == "pretrain":
        text = Path(args.text).read_text(encoding="utf-8")[:2_000_000]
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

    # chat fine-tune
    tokenizer = CharTokenizer.load(args.out / "tokenizer.json")
    model = load_model(args.out / "pretrain.pt", tokenizer, args)
    pairs = [json.loads(line) for line in Path(args.chat).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{len(pairs)} chat pairs, vocab {tokenizer.size}")

    def encode_pair(pair: dict) -> list[int]:
        return (
            [tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
            + tokenizer.encode(pair["user"])
            + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"]]
            + tokenizer.encode(pair["assistant"])
            + [tokenizer.eos]
        )

    sequences = [encode_pair(p) for p in pairs]
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5)
    rng = np.random.default_rng(1)
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        picks = rng.integers(0, len(sequences), size=args.batch)
        rows = []
        for index in picks:
            row = sequences[index][: args.seq_len]
            rows.append(row + [tokenizer.pad] * (args.seq_len - len(row)))
        batch = np.array(rows, dtype=np.int64)
        x = torch.from_numpy(batch[:, :-1])
        y = torch.from_numpy(batch[:, 1:])
        optimiser.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, tokenizer.size), y.reshape(-1),
                               ignore_index=tokenizer.pad)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % 25 == 0:
            print(f"  step {step}  loss {loss.item():.4f}  {time.time()-started:.0f}s", flush=True)
            save(model, tokenizer, args.out / "chat.pt", args)
        if args.time_budget and time.time() - started > args.time_budget:
            break
    save(model, tokenizer, args.out / "chat.pt", args)


if __name__ == "__main__":
    main()
