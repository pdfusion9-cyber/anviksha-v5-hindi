"""Guarded inference adapter for the merged ANVIKSHA KAVACH Model 1.

The deterministic KAVACH layers inspect both the incoming request and the
candidate response. This module does not grant tools, shell execution or network
access to the language model. It is an inference adapter, not a claim that the
model has passed a quality benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kavach import Kavach
from moe_model import MoEConfig, MoELM
from bpe_tokenizer import load_any
from tokenizer import language_tag


class KavachModel1:
    def __init__(self, checkpoint: Path, tokenizer_path: Path | None = None,
                 audit_dir: Path | None = None, threads: int = 2):
        torch.set_num_threads(threads)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model" not in state or "config" not in state:
            raise ValueError("checkpoint must contain model and config")
        self.config = MoEConfig(**state["config"])
        self.tokenizer_path = tokenizer_path or checkpoint.parent / "tokenizer.json"
        self.tokenizer = load_any(self.tokenizer_path)
        if self.tokenizer.size != self.config.vocab:
            raise ValueError(
                f"tokenizer size {self.tokenizer.size} != model vocab {self.config.vocab}")
        self.model = MoELM(self.config)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.guard = Kavach(
            "cybersecurity_assistant",
            state_dir=audit_dir,
        )

    def _prompt_ids(self, prompt: str) -> list[int]:
        tok = self.tokenizer
        required = ("<bos>", "<user>", "<sep>", "<assistant>")
        missing = [s for s in required if s not in tok.stoi]
        if missing:
            raise ValueError(f"tokenizer missing chat specials: {missing}")
        tag = language_tag(prompt)
        tag_id = tok.stoi.get(tag, tok.stoi.get("<mix>"))
        ids = [tok.stoi["<bos>"], tok.stoi["<user>"]]
        ids.extend(tok.encode(prompt))
        ids.extend([tok.stoi["<sep>"], tok.stoi["<assistant>"]])
        if tag_id is not None:
            ids.append(tag_id)
        return ids

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 48) -> dict:
        incoming = self.guard.judge(prompt, {"source": "kavach-model1-user"})
        if not incoming.allow:
            return {"accepted": False, "stage": "input_guard",
                    "action": incoming.action, "layer": incoming.layer,
                    "reasons": incoming.reasons, "answer": ""}

        ids = self._prompt_ids(incoming.text)
        max_new_tokens = max(1, min(int(max_new_tokens), self.config.seq_len - 1))
        ids = ids[-max(1, self.config.seq_len - max_new_tokens):]
        tokens = torch.tensor([ids], dtype=torch.long)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            if tokens.shape[1] >= self.config.seq_len:
                break
            logits = self.model(tokens)[:, -1, :]
            nxt = int(torch.argmax(logits, dim=-1).item())
            generated.append(nxt)
            tokens = torch.cat([tokens, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
            if nxt == self.tokenizer.eos:
                break

        answer = self.tokenizer.decode(generated).strip()
        if not answer:
            return {"accepted": False, "stage": "generation",
                    "action": "refuse", "layer": "empty_generation",
                    "reasons": ["model produced an empty response"], "answer": ""}

        outgoing = self.guard.judge(answer, {"source": "kavach-model1-output"})
        if not outgoing.allow:
            return {"accepted": False, "stage": "output_guard",
                    "action": outgoing.action, "layer": outgoing.layer,
                    "reasons": outgoing.reasons, "answer": ""}
        return {"accepted": True, "stage": "complete", "action": "allow",
                "layer": "allowed", "reasons": [], "answer": answer}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--audit-dir", type=Path, default=Path("runs/kavach_model1_audit"))
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    model = KavachModel1(args.checkpoint, args.tokenizer, args.audit_dir, args.threads)
    result = model.generate(args.prompt, args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
