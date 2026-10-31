"""Build a model from a checkpoint's own saved configuration.

Every evaluation script used to construct HindiLM from a hand-written config,
which silently stopped matching the moment the architecture gained optional
pieces (shared blocks, multi-level heads, the criticality controller). Loading
then raised a state_dict mismatch, and because the workflow runs those steps
with `|| true`, the failure produced no numbers and no error in the summary.

Reading `args` out of the checkpoint removes that whole class of bug: the model
is always rebuilt exactly as it was trained.
"""
from __future__ import annotations

from pathlib import Path

import torch

from model import HindiLM


def build_from_checkpoint(path: Path, tokenizer) -> tuple[HindiLM, dict]:
    state = torch.load(path, weights_only=False)
    args = state.get("args", {})
    model = HindiLM(
        tokenizer.size,
        args.get("dim", 192),
        args.get("layers", 4),
        args.get("heads", 4),
        args.get("ffn", 768),
        args.get("seq_len", 192),
        share_blocks=bool(args.get("share_blocks", False)),
        multi_level=bool(args.get("multi_level", False)),
        criticality=bool(args.get("criticality", False)),
    )
    model.load_state_dict(state["model"])
    model.eval()
    return model, args
