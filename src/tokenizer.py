"""Devanagari-aware character tokenizer.

Why this replaces V5's byte tokenizer
-------------------------------------
V5 mapped raw UTF-8 bytes to a 227-symbol vocabulary. A Hindi character such as
'क' is three bytes (E0 A4 95), so one letter costs three predictions and the
model has to learn UTF-8 continuation rules before it can learn any language at
all. That is a large part of why V5 could not produce coherent text.

Here every Devanagari codepoint is one token, plus matras, punctuation, digits
and a small set of special tokens. The vocabulary is built from the corpus and
stored with the checkpoint, so training and inference always agree.
"""
from __future__ import annotations

import json
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<sep>",
            "<hi>", "<en>", "<mix>"]
MAX_VOCAB = 1024


class CharTokenizer:
    def __init__(self, vocab: list[str] | None = None):
        self.itos = vocab or []
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad = self.stoi.get("<pad>", 0)
        self.bos = self.stoi.get("<bos>", 1)
        self.eos = self.stoi.get("<eos>", 2)

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        counts: dict[str, int] = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        # keep the most frequent characters, plus every special token
        common = sorted(counts.items(), key=lambda kv: -kv[1])[: MAX_VOCAB - len(SPECIALS)]
        return cls(SPECIALS + [ch for ch, _ in common])

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [self.stoi[ch] for ch in text if ch in self.stoi]
        if add_eos:
            ids.append(self.eos)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            ch = self.itos[i]
            if skip_special and ch in SPECIALS:
                continue
            out.append(ch)
        return "".join(out)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"vocab": self.itos}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["vocab"])

    @property
    def size(self) -> int:
        return len(self.itos)
