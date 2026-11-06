"""Guarded inference adapter for the merged ANVIKSHA KAVACH Model 1.

The deterministic KAVACH layers inspect both the incoming request and the
candidate response. This module does not grant tools, shell execution or network
access to the language model. It is an inference adapter, not a claim that the
model has passed a quality benchmark.
"""
from __future__ import annotations

import argparse
import codecs
import json
from pathlib import Path

import torch

from kavach import Kavach
from moe_model import MoEConfig, MoELM
from bpe_tokenizer import load_any
from tokenizer import language_tag


def utf8_valid_token_mask(tokenizer, generated: list[int], device):
    """Mask BPE tokens that would make the generated UTF-8 byte stream invalid.

    Incomplete multi-byte characters are allowed while generation continues;
    EOS is allowed only when the byte stream ends at a character boundary.
    Special tokens other than EOS are never emitted as user-facing text.
    """
    import torch

    vocab_size = tokenizer.size
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if not hasattr(tokenizer, "token_bytes"):
        return mask

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        for token_id in generated:
            decoder.decode(tokenizer.token_bytes(token_id), final=False)
    except (UnicodeDecodeError, ValueError):
        return mask

    pending = decoder.getstate()[0]
    for token_id in range(tokenizer.base):
        state = decoder.getstate()
        try:
            decoder.decode(tokenizer.token_bytes(token_id), final=False)
            mask[token_id] = True
        except UnicodeDecodeError:
            pass
        finally:
            decoder.setstate(state)

    if not pending:
        mask[tokenizer.eos] = True
    return mask


class KavachModel1:
    def __init__(self, checkpoint: Path, tokenizer_path: Path | None = None,
                 audit_dir: Path | None = None, threads: int = 2):
        torch.set_num_threads(threads)
        # Memory-map checkpoint tensors and construct the module on meta to
        # avoid holding checkpoint + randomly initialized model copies at once.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if "model" not in state or "config" not in state:
            raise ValueError("checkpoint must contain model and config")
        self.config = MoEConfig(**state["config"])
        self.tokenizer_path = tokenizer_path or checkpoint.parent / "tokenizer.json"
        self.tokenizer = load_any(self.tokenizer_path)
        if self.tokenizer.size != self.config.vocab:
            raise ValueError(
                f"tokenizer size {self.tokenizer.size} != model vocab {self.config.vocab}")
        with torch.device("meta"):
            self.model = MoELM(self.config)
        self.model.load_state_dict(state["model"], strict=True, assign=True)
        self.model.head.weight = self.model.embed.weight
        actual = sum(p.numel() for p in self.model.parameters())
        expected = self.config.counts()["total"]
        if actual != expected or actual != 574_391_040:
            raise ValueError(
                f"unexpected parameter count: actual={actual}, expected={expected}")
        self.model.eval()
        del state
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
            logits = self.model(tokens)[:, -1, :].squeeze(0)
            if hasattr(self.tokenizer, "token_bytes"):
                valid = utf8_valid_token_mask(self.tokenizer, generated, logits.device)
                if not bool(valid.any()):
                    break
                logits = logits.masked_fill(~valid, float("-inf"))
            nxt = int(torch.argmax(logits, dim=-1).item())
            generated.append(nxt)
            tokens = torch.cat([tokens, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
            if nxt == self.tokenizer.eos:
                break

        if hasattr(self.tokenizer, "token_bytes"):
            # A max-token cutoff can land inside a multi-byte UTF-8 character.
            # Drop only the unfinished suffix; never emit U+FFFD.
            answer = ""
            while generated:
                raw = b"".join(
                    self.tokenizer.token_bytes(t)
                    for t in generated if t < self.tokenizer.base
                )
                try:
                    answer = raw.decode("utf-8", errors="strict").strip()
                    break
                except UnicodeDecodeError:
                    generated.pop()
        else:
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
