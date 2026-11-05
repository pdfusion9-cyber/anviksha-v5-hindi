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

# <think>/</think> and <ans> force every reply through a reasoning stage before
# an answer is emitted, so the model cannot answer without a trace and the answer
# can be extracted unambiguously.
SPECIALS = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<sep>",
            "<hi>", "<en>", "<mix>", "<think>", "</think>", "<ans>", "</ans>"]
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


def language_tag(text: str) -> str:
    """Which language marker this text should carry.

    The trainer applies this to the answer; the generator has only the question,
    so it applies the same rule there. The first version of the generator hard
    coded <hi>, which for an English answer primed the model with a tag it was
    never trained to see in that position -- it then emitted the tag again
    instead of an answer, and the reply decoded to the empty string.
    """
    dev = sum(1 for ch in text if "\u0900" <= ch <= "\u097f")
    lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if dev and dev >= lat:
        return "<hi>"
    if lat and dev == 0:
        words = {w.lower() for w in text.replace("?", " ").split()}
        markers = words & {"hai", "hain", "ka", "ki", "ke", "ko", "se", "me",
                           "kya", "mujhe", "aap", "nahi", "kare", "karo"}
        return "<mix>" if markers else "<en>"
    return "<mix>"
