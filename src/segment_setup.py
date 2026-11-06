"""Prepare one training segment: build the starting checkpoint, or adopt one.

A long run is a chain of jobs, each resuming what the last one left. The shell
that drives that chain is fragile -- nested heredocs inside an if-block broke the
first attempt, because a heredoc terminator has to start at column zero and the
one inside the branch was indented. So the logic lives here, in Python, where it
can be tested, and the workflow just calls it.

Two paths:

    fresh    build the initial weights from one seed, so every copy starts
             identical and any difference later is something training did
    resume   adopt a checkpoint from the previous segment, keeping its step
             count, token count and optimiser state, and refuse it if it does
             not match the configuration we are about to train
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from moe_model import MoEConfig, MoELM


def fresh(out: Path, vocab: int, dim: int, layers: int, heads: int,
          experts: int, ffn: int, seq_len: int, seed: int) -> None:
    torch.manual_seed(seed)
    cfg = MoEConfig(vocab=vocab, dim=dim, layers=layers, heads=heads,
                    seq_len=seq_len, experts=experts, top_k=2, ffn_mult=ffn)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": MoELM(cfg).state_dict(), "config": cfg.__dict__,
                "args": {"stage": "init", "seed": seed}}, out)
    c = cfg.counts()
    print(f"fresh start from seed {seed}")
    print(f"  {c['total']:,} parameters, {c['active']:,} active")
    print(f"  written to {out}")


def resume(previous: Path, out: Path, vocab: int, dim: int, layers: int,
           heads: int, experts: int, ffn: int, seq_len: int) -> None:
    st = torch.load(previous, weights_only=False)
    if "model" not in st:
        raise SystemExit(f"{previous} has no model state")
    cfg = MoEConfig(**st["config"])

    want = dict(vocab=vocab, dim=dim, layers=layers, heads=heads,
                seq_len=seq_len, experts=experts, ffn_mult=ffn)
    got = dict(vocab=cfg.vocab, dim=cfg.dim, layers=cfg.layers, heads=cfg.heads,
               seq_len=cfg.seq_len, experts=cfg.experts, ffn_mult=cfg.ffn_mult)
    if want != got:
        raise SystemExit(
            f"the checkpoint does not match the configuration this segment "
            f"would train.\n  checkpoint {got}\n  this run   {want}\n"
            f"Continuing would load weights into a differently shaped model.")
    if not st.get("optimiser"):
        print("  WARNING: this checkpoint carries no optimiser state, so AdamW "
              "restarts its moments. Training will still work but the first "
              "steps of this segment are wasted re-learning them.")

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(previous, out)
    e = st.get("extra", {})
    print(f"resuming from {previous}")
    print(f"  step {st.get('step', 0):,}, tokens {e.get('tokens_seen', 0):,}, "
          f"docs {e.get('docs_read', 0):,}")
    print(f"  copied to {out} so the trainer finds it")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--previous", type=Path, default=None)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--experts", type=int, default=12)
    ap.add_argument("--ffn", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()

    if args.previous is not None and args.previous.is_file():
        resume(args.previous, args.out, args.vocab, args.dim, args.layers,
               args.heads, args.experts, args.ffn, args.seq_len)
    else:
        if args.previous is not None:
            print(f"  no checkpoint at {args.previous}; starting fresh")
        fresh(args.out, args.vocab, args.dim, args.layers, args.heads,
              args.experts, args.ffn, args.seq_len, args.seed)


if __name__ == "__main__":
    main()
