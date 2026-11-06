"""Supervised chat fine-tuning for the merged 574M KAVACH MoE checkpoint.

The five-copy pretraining job sees plain cybersecurity documents. It does not
teach chat boundary tokens or assistant-turn termination. This script adds the
exact prompt format used by KavachModel1 and computes loss only on assistant
answer tokens plus EOS. The source checkpoint is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bpe_tokenizer import load_any
from moe_model import MoEConfig, MoELM
from tokenizer import language_tag


def read_pairs(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"chat data not found: {path}")
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            user = row.get("user")
            assistant = row.get("assistant")
            if isinstance(user, str) and isinstance(assistant, str):
                user, assistant = user.strip(), assistant.strip()
                if user and assistant:
                    rows.append({"user": user, "assistant": assistant})
    return rows


def build_example(
    row: dict[str, str],
    tokenizer: Any,
    max_seq_len: int,
    max_answer_tokens: int,
) -> dict[str, list[int]] | None:
    """Encode [BOS USER prompt SEP ASSISTANT lang answer EOS].

    Prompt/header targets are masked. The model learns only the assistant
    continuation and its end marker, matching the inference adapter exactly.
    """
    required = ("<bos>", "<user>", "<sep>", "<assistant>", "<eos>", "<pad>")
    missing = [token for token in required if token not in tokenizer.stoi]
    if missing:
        raise ValueError(f"tokenizer missing required special tokens: {missing}")
    if max_seq_len < 16 or max_answer_tokens < 1:
        raise ValueError("max_seq_len must be >=16 and max_answer_tokens >=1")

    user_ids = list(tokenizer.encode(row["user"]))
    answer_ids = list(tokenizer.encode(row["assistant"]))
    if not answer_ids:
        return None

    tag = language_tag(row["user"])
    if tag not in tokenizer.stoi:
        tag = "<mix>" if "<mix>" in tokenizer.stoi else None
    if tag is None:
        raise ValueError("tokenizer has no usable language marker")

    header = [
        tokenizer.stoi["<bos>"],
        tokenizer.stoi["<user>"],
    ]
    suffix = [
        tokenizer.stoi["<sep>"],
        tokenizer.stoi["<assistant>"],
        tokenizer.stoi[tag],
    ]

    answer_budget = min(
        len(answer_ids),
        max_answer_tokens,
        max_seq_len - len(header) - len(suffix) - 1,
    )
    answer_ids = answer_ids[:answer_budget]
    prompt_budget = max_seq_len - len(header) - len(suffix) - len(answer_ids) - 1
    if prompt_budget < 1:
        return None
    if len(user_ids) > prompt_budget:
        user_ids = user_ids[-prompt_budget:]

    prefix = header + user_ids + suffix
    tokens = prefix + answer_ids + [tokenizer.stoi["<eos>"]]
    loss_mask = (
        [0] * (len(prefix) - 1)
        + [1] * (len(answer_ids) + 1)
    )
    if len(loss_mask) != len(tokens) - 1:
        raise AssertionError("loss mask does not align with next-token targets")
    return {"tokens": tokens, "loss_mask": loss_mask}


def make_batch(examples: list[dict[str, list[int]]], pad_id: int):
    if not examples:
        raise ValueError("cannot create a batch from zero examples")
    width = max(len(row["tokens"]) for row in examples) - 1
    x = torch.full((len(examples), width), pad_id, dtype=torch.long)
    y = torch.full((len(examples), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.float32)
    for i, row in enumerate(examples):
        tokens = row["tokens"]
        n = len(tokens) - 1
        x[i, :n] = torch.tensor(tokens[:-1], dtype=torch.long)
        y[i, :n] = torch.tensor(tokens[1:], dtype=torch.long)
        mask[i, :n] = torch.tensor(row["loss_mask"], dtype=torch.float32)
    return x, y, mask


def masked_loss(model: MoELM, x, y, mask, pad_id: int):
    logits = model(x)
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        ignore_index=pad_id,
        reduction="none",
    ).view_as(y)
    denom = mask.sum().clamp_min(1.0)
    token_loss = (per_token * mask).sum() / denom
    return token_loss, model.balance_loss()


@torch.inference_mode()
def evaluate_loss(model, pairs, tokenizer, max_seq_len, max_answer_tokens, limit):
    model.eval()
    values = []
    for row in pairs[:limit]:
        example = build_example(row, tokenizer, max_seq_len, max_answer_tokens)
        if example is None:
            continue
        x, y, mask = make_batch([example], tokenizer.pad)
        loss, _ = masked_loss(model, x, y, mask, tokenizer.pad)
        if torch.isfinite(loss):
            values.append(float(loss))
    return {
        "examples_scored": len(values),
        "mean_answer_loss": round(sum(values) / len(values), 6) if values else None,
    }


def load_base(checkpoint: Path):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"base checkpoint not found: {checkpoint}")
    # mmap + meta construction avoids a second randomly initialized 574M copy.
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if "model" not in state or "config" not in state:
        raise ValueError("checkpoint must contain 'model' and 'config'")
    config = MoEConfig(**state["config"])
    with torch.device("meta"):
        model = MoELM(config)
    missing, unexpected = model.load_state_dict(
        state["model"],
        strict=True,
        assign=True,
    )
    if missing or unexpected:
        raise ValueError(f"strict checkpoint load mismatch: {missing=}, {unexpected=}")
    model.head.weight = model.embed.weight
    actual = sum(parameter.numel() for parameter in model.parameters())
    expected = config.counts()["total"]
    if actual != expected:
        raise ValueError(f"parameter count mismatch: actual={actual}, expected={expected}")
    del state
    model.train()
    return model, config, actual


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/kavach_model1_chat_sft.pt"))
    parser.add_argument("--report", type=Path, default=Path("runs/kavach_model1_chat_sft.json"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-answer-tokens", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--balance-weight", type=float, default=0.001)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261106)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise SystemExit("--steps and --batch-size must be positive")
    if args.out.resolve() == args.checkpoint.resolve():
        raise SystemExit("refusing to overwrite the source checkpoint")

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    tokenizer = load_any(args.tokenizer)
    train_rows = read_pairs(args.train)
    heldout_rows = read_pairs(args.heldout)
    if len(train_rows) < 100:
        raise SystemExit(f"too few licensed SFT pairs: {len(train_rows)}; refusing to train")
    if not heldout_rows:
        raise SystemExit("held-out set is empty; refusing an unmeasured SFT run")

    model, config, parameter_count = load_base(args.checkpoint)
    if args.max_seq_len > config.seq_len:
        raise SystemExit(
            f"--max-seq-len {args.max_seq_len} exceeds checkpoint seq_len {config.seq_len}"
        )
    footprint_gb = config.footprint_gb()
    if footprint_gb > 12.5:
        raise SystemExit(
            f"full AdamW training footprint is {footprint_gb:.2f} GB; "
            "refusing a run with insufficient memory headroom"
        )

    before = evaluate_loss(
        model,
        heldout_rows,
        tokenizer,
        args.max_seq_len,
        args.max_answer_tokens,
        args.eval_limit,
    )
    model.train()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    usable = [
        example for row in train_rows
        if (example := build_example(
            row, tokenizer, args.max_seq_len, args.max_answer_tokens
        )) is not None
    ]
    if len(usable) < 100:
        raise SystemExit(f"too few encodable SFT pairs after tokenization: {len(usable)}")

    losses: list[float] = []
    for step in range(1, args.steps + 1):
        batch_examples = [rng.choice(usable) for _ in range(args.batch_size)]
        x, y, mask = make_batch(batch_examples, tokenizer.pad)
        optimiser.zero_grad(set_to_none=True)
        token_loss, balance = masked_loss(model, x, y, mask, tokenizer.pad)
        loss = token_loss + args.balance_weight * balance
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite SFT loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        losses.append(float(token_loss.detach()))
        if step == 1 or step % 10 == 0 or step == args.steps:
            recent = sum(losses[-10:]) / min(10, len(losses))
            print(
                f"step {step}/{args.steps} answer_loss={recent:.4f} "
                f"balance={float(balance.detach()):.4f} "
                f"answer_tokens={int(mask.sum().item())}",
                flush=True,
            )

    model.eval()
    after = evaluate_loss(
        model,
        heldout_rows,
        tokenizer,
        args.max_seq_len,
        args.max_answer_tokens,
        args.eval_limit,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "sft": {
                "steps": args.steps,
                "train_pairs": len(train_rows),
                "usable_pairs": len(usable),
                "heldout_pairs": len(heldout_rows),
                "answer_only_loss": True,
                "parameter_count": parameter_count,
            },
        },
        args.out,
    )
    report = {
        "classification": "experimental_chat_sft_not_quality_certification",
        "base_checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "parameters": parameter_count,
        "training_pairs": len(train_rows),
        "usable_pairs": len(usable),
        "heldout_pairs": len(heldout_rows),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "answer_only_loss": True,
        "heldout_answer_loss_before": before,
        "heldout_answer_loss_after": after,
        "last_10_train_answer_loss": round(
            sum(losses[-10:]) / min(10, len(losses)), 6
        ),
        "source_checkpoint_preserved": True,
        "notes": [
            "This run teaches the exact chat boundary and EOS tokens used at inference.",
            "A lower held-out loss is useful but does not establish factual accuracy.",
            "Run the six-case guarded smoke test before making any quality claim.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
