"""Quantize the trained model and measure what it costs in size and quality.

Three artefacts are produced from the same checkpoint:

  fp32  — the original weights
  int8  — per-tensor symmetric quantization, one scale per weight matrix
  int4  — the same at 4 bits, two values packed per byte

For each, the script reports the on-disk size and the held-out perplexity after
dequantization. That second number is the one that matters: a 4-bit model that is
small but useless is not a win.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import HindiLM  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def quantize_tensor(tensor: torch.Tensor, bits: int) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor quantization. Returns (packed bytes, scale)."""
    values = tensor.detach().float().cpu().numpy().ravel()
    limit = 2 ** (bits - 1) - 1
    scale = float(np.abs(values).max() / limit) if values.size else 0.0
    if scale == 0.0:
        return np.zeros(values.size, dtype=np.uint8), 0.0
    codes = np.clip(np.round(values / scale), -limit, limit).astype(np.int8)
    if bits == 8:
        return codes.view(np.uint8), scale
    # 4-bit: shift to 0..15 and pack two per byte
    unsigned = (codes + 8).astype(np.uint8)
    if unsigned.size % 2:
        unsigned = np.append(unsigned, np.uint8(0))
    packed = (unsigned[0::2] << 4) | unsigned[1::2]
    return packed, scale


def dequantize(packed: np.ndarray, scale: float, bits: int, shape) -> torch.Tensor:
    if bits == 8:
        codes = packed.view(np.int8).astype(np.float32)
    else:
        high = (packed >> 4).astype(np.uint8)
        low = (packed & 0x0F).astype(np.uint8)
        interleaved = np.empty(packed.size * 2, dtype=np.uint8)
        interleaved[0::2] = high
        interleaved[1::2] = low
        codes = (interleaved.astype(np.int16) - 8).astype(np.float32)
    values = codes[: int(np.prod(shape))] * scale
    return torch.from_numpy(values.reshape(shape))


def roundtrip(model: HindiLM, bits: int) -> tuple[dict, int]:
    """Return a state dict quantized to `bits` and the payload size in bytes."""
    quantized: dict[str, torch.Tensor] = {}
    payload = 0
    for name, tensor in model.state_dict().items():
        if tensor.dim() < 2:  # norms and biases stay exact
            quantized[name] = tensor
            payload += tensor.numel() * 4
            continue
        packed, scale = quantize_tensor(tensor, bits)
        payload += packed.nbytes + 4  # +4 bytes for the float32 scale
        quantized[name] = dequantize(packed, scale, bits, tuple(tensor.shape))
    return quantized, payload


@torch.no_grad()
def perplexity(model, tok, pairs, seq_len, batch=8) -> float:
    total, tokens = 0.0, 0
    for start in range(0, len(pairs), batch):
        rows = []
        for pair in pairs[start : start + batch]:
            row = ([tok.stoi["<bos>"], tok.stoi["<user>"]] + tok.encode(pair["user"])
                   + [tok.stoi["<sep>"], tok.stoi["<assistant>"]]
                   + tok.encode(pair["assistant"]) + [tok.eos])[:seq_len]
            rows.append(row + [tok.pad] * (seq_len - len(row)))
        tensor = torch.tensor(rows, dtype=torch.long)
        logits = model(tensor[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, tok.size), tensor[:, 1:].reshape(-1),
                               ignore_index=tok.pad, reduction="sum")
        total += float(loss)
        tokens += int((tensor[:, 1:] != tok.pad).sum())
    return math.exp(total / max(tokens, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/chat.pt", type=Path)
    parser.add_argument("--heldout", default="data/chat_heldout.jsonl", type=Path)
    parser.add_argument("--out", default="runs/quantization.json", type=Path)
    parser.add_argument("--limit", default=100, type=int)
    args = parser.parse_args()

    tok = CharTokenizer.load(args.checkpoint.parent / "tokenizer.json")
    state = torch.load(args.checkpoint, weights_only=False)
    cfg = state["args"]
    model = HindiLM(tok.size, cfg["dim"], cfg["layers"], cfg["heads"], cfg["ffn"],
                    cfg["seq_len"])
    model.load_state_dict(state["model"])
    model.eval()

    pairs = [json.loads(line) for line in args.heldout.read_text(encoding="utf-8").splitlines()
             if line.strip()][: args.limit]

    parameters = sum(p.numel() for p in model.parameters())
    report = {"parameters": parameters, "evaluated_pairs": len(pairs), "variants": {}}

    baseline = perplexity(model, tok, pairs, cfg["seq_len"])
    report["variants"]["fp32"] = {
        "bytes": parameters * 4,
        "megabytes": round(parameters * 4 / 1e6, 2),
        "perplexity": round(baseline, 3),
    }
    print(f"fp32: {parameters * 4 / 1e6:.2f} MB, perplexity {baseline:.3f}")

    for bits in (8, 4):
        quantized, payload = roundtrip(model, bits)
        model.load_state_dict(quantized)
        model.eval()
        ppl = perplexity(model, tok, pairs, cfg["seq_len"])
        report["variants"][f"int{bits}"] = {
            "bytes": payload,
            "megabytes": round(payload / 1e6, 2),
            "perplexity": round(ppl, 3),
            "degradation": round(ppl - baseline, 3),
        }
        print(f"int{bits}: {payload / 1e6:.2f} MB, perplexity {ppl:.3f} "
              f"(+{ppl - baseline:.3f})")
        model.load_state_dict(state["model"])  # restore for the next variant

    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
