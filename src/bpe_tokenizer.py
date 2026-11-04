"""BPE behind the same interface as CharTokenizer, so training can just switch.

The character tokenizer caps at 1024 symbols, and Devanagari is three bytes per
character in UTF-8, so a 192-token window holds roughly 85 Hindi characters. The
BPE trained on the real corpus encodes Hindi at 6.26 bytes per token, which puts
about 612 characters in the same window -- 6.2x more text for free, at the same
parameter count.

That gain has been sitting unused: train.py only ever built a CharTokenizer, so
every run so far has trained on the short window. This adapter gives BPE the same
methods so the trainer can pick either without other changes.

Layout of the vocabulary:

    0..255              the raw bytes, so any text can be encoded
    256..256+M-1        BPE merges, in the order they were learned
    then                the special tokens

Specials sit at the top so they never collide with a merge id, and a byte-level
base means no character is ever unencodable -- including text the tokenizer has
never seen, which matters more for Hindi than for English.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpe import BPE  # noqa: E402
from tokenizer import SPECIALS  # noqa: E402


class BPETokenizer:
    def __init__(self, bpe: BPE, specials: list[str] | None = None) -> None:
        self.bpe = bpe
        self.specials = list(specials or SPECIALS)
        self.base = 256 + len(bpe.merges)
        self._cache: dict[str, list[int]] = {}
        self.itos = ([f"<byte:{i}>" for i in range(256)]
                     + [f"<merge:{i}>" for i in range(len(bpe.merges))]
                     + self.specials)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.special_stoi = {s: self.base + i for i, s in enumerate(self.specials)}
        self.pad = self.special_stoi.get("<pad>", 0)
        self.bos = self.special_stoi.get("<bos>", 1)
        self.eos = self.special_stoi.get("<eos>", 2)

    @classmethod
    def train(cls, text: str, vocab_size: int = 4096) -> "BPETokenizer":
        # the requested size counts the specials, since the model's output layer
        # is sized from this tokenizer's total
        merges_target = max(257, vocab_size - len(SPECIALS))
        return cls(BPE.train(text, vocab_size=merges_target, verbose=False))

    @property
    def size(self) -> int:
        return self.base + len(self.specials)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Encode, caching per word.

        Hindi text repeats its words heavily, so a cache over word forms turns
        encoding a multi-gigabyte corpus from hours into minutes. The cache is
        bounded so a long tail of rare words cannot grow it without limit.
        """
        cache = self._cache
        if len(cache) > 400_000:
            cache.clear()
        ids = self.bpe.encode(text, cache=cache)
        if add_eos:
            ids.append(self.eos)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        plain = []
        for i in ids:
            if i >= self.base:
                if not skip_special:
                    plain_text = self.specials[i - self.base] if i - self.base < len(self.specials) else ""
                    plain.append(plain_text)
                continue
            plain.append(i)
        return self.bpe.decode([i for i in plain if isinstance(i, int)])

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "kind": "bpe",
            "merges": self.bpe.merges,
            "specials": self.specials,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("kind") != "bpe":
            raise ValueError(f"{path} is not a BPE tokenizer")
        return cls(BPE([tuple(m) for m in data["merges"]]), data.get("specials"))


def load_any(path: Path):
    """Load either tokenizer kind, so callers do not have to care which."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("kind") == "bpe":
        return BPETokenizer.load(path)
    from tokenizer import CharTokenizer
    return CharTokenizer.load(path)


if __name__ == "__main__":
    text = ("भारत की राजधानी नई दिल्ली है। यह देश का सबसे बड़ा शहर नहीं है। "
            "ताजमहल आगरा में यमुना नदी के किनारे है। " * 60)
    tok = BPETokenizer.train(text, vocab_size=1024)
    print(f"vocab {tok.size} (base {tok.base} + {len(tok.specials)} specials)")

    probe = "भारत की राजधानी नई दिल्ली है।"
    ids = tok.encode(probe)
    back = tok.decode(ids)
    print(f"encode/decode exact: {back == probe}")
    print(f"{len(probe.encode('utf-8'))} bytes -> {len(ids)} tokens")
    print(f"chars per 192-token window: {192 * len(probe) / len(ids):.0f}")
    print(f"special ids: pad {tok.pad}, bos {tok.bos}, eos {tok.eos}")
