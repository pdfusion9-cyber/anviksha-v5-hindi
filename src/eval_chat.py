"""Held-out evaluation: perplexity, generation and safety screening.

Reports three things that a six-prompt eyeball check cannot:
  1. held-out perplexity on pairs the model never trained on
  2. the same perplexity on the training pairs, for the overfit gap
  3. safety screening over generations from held-out prompts
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import HindiLM  # noqa: E402
from safety import screen  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/chat.pt", type=Path)
    p.add_argument("--train", default="data/multi_train.jsonl", type=Path)
    p.add_argument("--heldout", default="data/chat_heldout.jsonl", type=Path)
    p.add_argument("--out", default="runs/eval.json")
    p.add_argument("--limit", default=300, type=int)
    p.add_argument("--generate", default=40, type=int)
    p.add_argument("--temperature", default=0.7, type=float)
    p.add_argument("--top-k", default=20, type=int)
    p.add_argument("--threads", default=2, type=int)
    return p.parse_args()


def encode_pair(pair: dict, tok: CharTokenizer) -> list[int]:
    return ([tok.stoi["<bos>"], tok.stoi["<user>"]]
            + tok.encode(pair["user"])
            + [tok.stoi["<sep>"], tok.stoi["<assistant>"]]
            + tok.encode(pair["assistant"]) + [tok.eos])


@torch.no_grad()
def split_perplexity(model, tok, pairs, seq_len, batch=8) -> tuple[float, int]:
    total_loss, total_tokens = 0.0, 0
    for start in range(0, len(pairs), batch):
        rows = []
        for pair in pairs[start : start + batch]:
            row = encode_pair(pair, tok)[:seq_len]
            rows.append(row + [tok.pad] * (seq_len - len(row)))
        tensor = torch.tensor(rows, dtype=torch.long)
        logits = model(tensor[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, tok.size), tensor[:, 1:].reshape(-1),
                               ignore_index=tok.pad, reduction="sum")
        tokens = int((tensor[:, 1:] != tok.pad).sum())
        total_loss += float(loss)
        total_tokens += tokens
    return math.exp(total_loss / max(total_tokens, 1)), total_tokens


@torch.no_grad()
def generate(model, tok, prompt: str, seq_len: int, temperature: float, top_k: int,
             max_new: int = 48) -> str:
    ids = ([tok.stoi["<bos>"], tok.stoi["<user>"]]
           + tok.encode(prompt)
           + [tok.stoi["<sep>"], tok.stoi["<assistant>"]])
    tokens = torch.tensor([ids])
    for _ in range(max_new):
        if tokens.shape[1] >= seq_len:
            break
        logits = model(tokens)[:, -1, :] / temperature
        values, _ = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
        logits[logits < values[:, -1:]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        tokens = torch.cat([tokens, nxt], dim=1)
        if int(nxt) == tok.eos:
            break
    return tok.decode(tokens[0, len(ids):].tolist())


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    tok = CharTokenizer.load(args.checkpoint.parent / "tokenizer.json")
    state = torch.load(args.checkpoint, weights_only=False)
    cfg = state["args"]
    model = HindiLM(tok.size, cfg["dim"], cfg["layers"], cfg["heads"], cfg["ffn"],
                    cfg["seq_len"])
    model.load_state_dict(state["model"])
    model.eval()

    def read(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    train_pairs = read(args.train)[: args.limit]
    held_pairs = read(args.heldout)[: args.limit]

    train_ppl, _ = split_perplexity(model, tok, train_pairs, cfg["seq_len"])
    held_ppl, _ = split_perplexity(model, tok, held_pairs, cfg["seq_len"])
    print(f"train perplexity  {train_ppl:.2f}   ({len(train_pairs)} pairs)")
    print(f"held-out perplexity {held_ppl:.2f}   ({len(held_pairs)} pairs)")
    print(f"gap {held_ppl - train_ppl:+.2f}")

    generations = []
    for pair in held_pairs[: args.generate]:
        reply = generate(model, tok, pair["user"], cfg["seq_len"],
                         args.temperature, args.top_k)
        generations.append({"prompt": pair["user"], "reply": reply,
                            "safety": screen(reply)})

    flagged = sum(1 for g in generations if not g["safety"]["clean"])
    issues: dict[str, int] = {}
    for g in generations:
        for issue in g["safety"]["issues"]:
            issues[issue] = issues.get(issue, 0) + 1

    print(f"screened {len(generations)} generations, {flagged} flagged")
    for name, count in sorted(issues.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {count}")
    for g in generations[:6]:
        print(f"U: {g['prompt'][:70]}\nA: {g['reply'][:120]}\n")

    Path(args.out).write_text(json.dumps({
        "train_perplexity": train_ppl,
        "heldout_perplexity": held_ppl,
        "overfit_gap": held_ppl - train_ppl,
        "train_pairs_evaluated": len(train_pairs),
        "heldout_pairs_evaluated": len(held_pairs),
        "generations_screened": len(generations),
        "generations_flagged": flagged,
        "safety_issues": issues,
        "samples": generations,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
