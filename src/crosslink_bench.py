"""Does cross-linking make a small model stronger without more parameters?

Trains the same architecture several ways on identical data, in identical order,
for identical steps, and reports the loss each reaches. The parameter count is
printed for every arm so the claim being tested -- more capability at the same
size -- is checked rather than assumed.

Arms:

    baseline          the model as it is today
    cross             each block also sees a running average of all earlier layer
                      outputs (zero new parameters)
    recurrent2        the block stack applied twice with a step embedding
                      (2 x dim extra parameters, a few hundred)
    cross+recurrent2  both

If cross-linking helps, `cross` reaches a lower loss than `baseline` at the same
parameter count and the same speed. If it does not, that is the answer, and it
costs one short run to know.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import HindiLM  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

ARMS = {
    "baseline": dict(),
    "cross": dict(cross_links=True),
    "recurrent2": dict(recurrent=2),
    "cross+recurrent2": dict(cross_links=True, recurrent=2),
}


def build_pairs(path: Path, tok: CharTokenizer, seq_len: int, limit: int) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids = ([tok.stoi["<bos>"], tok.stoi["<user>"]] + tok.encode(row["user"])
               + [tok.stoi["<sep>"], tok.stoi["<assistant>"]]
               + tok.encode(row["assistant"]) + [tok.eos])
        ids = ids[:seq_len]
        out.append(ids + [tok.pad] * (seq_len - len(ids)))
        if len(out) >= limit:
            break
    return out


def train_arm(name: str, kwargs: dict, sequences: list, vocab: int, seq_len: int,
              steps: int, batch: int, lr: float, seed: int,
              dim: int = 384, layers: int = 6, heads: int = 6,
              ffn: int = 1536) -> dict:
    torch.manual_seed(seed)
    model = HindiLM(vocab, dim, layers, heads, ffn, seq_len, share_blocks=True, **kwargs)
    params = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    losses, started = [], time.time()
    for step in range(steps):
        picks = torch.randint(0, len(sequences), (batch,), generator=torch.Generator().manual_seed(seed + step))
        batch_tokens = torch.tensor([sequences[int(i)] for i in picks], dtype=torch.long)
        logits = model(batch_tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab), batch_tokens[:, 1:].reshape(-1),
                               ignore_index=0)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        losses.append(float(loss.detach()))
    elapsed = time.time() - started

    tail = losses[-20:]
    mean_tail = sum(tail) / len(tail)
    try:
        ppl = math.exp(mean_tail)
    except OverflowError:
        ppl = float("inf")
    return {
        "arm": name,
        "parameters": params,
        "steps": steps,
        "seconds": round(elapsed, 1),
        "steps_per_second": round(steps / elapsed, 2),
        "first_loss": round(losses[0], 4),
        "last20_loss": round(mean_tail, 4),
        "perplexity": round(ppl, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/bench_train.jsonl", type=Path)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--seq-len", default=192, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--limit", default=1500, type=int)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--dim", default=384, type=int)
    parser.add_argument("--layers", default=6, type=int)
    parser.add_argument("--heads", default=6, type=int)
    parser.add_argument("--ffn", default=1536, type=int)
    parser.add_argument("--arms", default="baseline,cross")
    parser.add_argument("--out", default="runs/crosslink_bench.json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    rows = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]
    tok = CharTokenizer.train("\n".join(r["user"] + "\n" + r["assistant"] for r in rows[:800]))
    sequences = build_pairs(args.data, tok, args.seq_len, args.limit)
    print(f"{len(sequences)} sequences, vocab {tok.size}, seq {args.seq_len}")

    results = []
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if name not in ARMS:
            raise SystemExit(f"unknown arm {name}; known: {list(ARMS)}")
        result = train_arm(name, ARMS[name], sequences, tok.size, args.seq_len,
                           args.steps, args.batch, args.lr, seed=1234,
                           dim=args.dim, layers=args.layers, heads=args.heads,
                           ffn=args.ffn)
        results.append(result)
        print(f"  {name:18} params {result['parameters']:>9,}  "
              f"loss {result['last20_loss']:.4f}  ppl {result['perplexity']:.2f}  "
              f"{result['steps_per_second']:.2f} steps/s")

    base = next((r for r in results if r["arm"] == "baseline"), None)
    if base:
        print("\nagainst baseline:")
        for r in results:
            if r["arm"] == "baseline":
                continue
            gain = base["last20_loss"] - r["last20_loss"]
            verdict = "better" if gain > 0.005 else ("worse" if gain < -0.005 else "no change")
            print(f"  {r['arm']:18} loss {gain:+.4f}  ({verdict}), "
                  f"params {r['parameters'] - base['parameters']:+,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
