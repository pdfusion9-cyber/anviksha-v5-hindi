"""A byte-level BPE that works on Hindi without special-casing any script.

The model's vocabulary is 256, which means it reads raw bytes. Devanagari is
three bytes per character in UTF-8, so a 256-token context holds about 85 Hindi
characters -- less than one sentence. That is the single biggest limit on the
Hindi side, and it is a tokenizer problem, not a model problem.

Byte-level BPE keeps the 256 bytes as its base and learns merges on top, so it
still encodes anything -- including text the model has never seen -- while giving
common Devanagari sequences their own token.

The merge loop is incremental: pair counts are kept up to date and only the words
that contain the winning pair are rewritten. Recomputing every count on every
merge is the obvious version and it does not finish -- 4096 merges over millions
of word types is billions of operations.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


class BPE:
    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges: list[tuple[int, int]] = merges or []
        self.rank = {pair: i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ---------------------------------------------------------------- train --
    @classmethod
    def train(cls, text: str, vocab_size: int = 4096, verbose: bool = True,
              max_word_types: int = 60_000) -> "BPE":
        if vocab_size < 257:
            raise ValueError("vocab_size must be at least 257")

        word_freq = Counter(text.split())
        if len(word_freq) > max_word_types:
            word_freq = Counter(dict(word_freq.most_common(max_word_types)))
            if verbose:
                print(f"  kept {max_word_types:,} most frequent word types")

        words: list[list[int]] = []
        freqs: list[int] = []
        for word, freq in word_freq.items():
            ids = list(word.encode("utf-8"))
            if ids:
                words.append(ids)
                freqs.append(freq)

        # pair -> total weighted count, and pair -> the words containing it
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            f = freqs[wi]
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += f
                pair_words[pair].add(wi)

        merges: list[tuple[int, int]] = []
        target = vocab_size - 256
        while len(merges) < target:
            if not pair_counts:
                break
            best = max(pair_counts, key=lambda p: pair_counts[p])
            count = pair_counts[best]
            if count < 2:
                break

            new_id = 256 + len(merges)
            merges.append(best)
            a, b = best

            # rewrite only the words that actually contain the winning pair
            for wi in list(pair_words.pop(best, ())):
                ids = words[wi]
                f = freqs[wi]
                old_pairs = Counter(zip(ids, ids[1:]))
                out: list[int] = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                new_pairs = Counter(zip(out, out[1:]))
                words[wi] = out
                for pair, c in old_pairs.items():
                    pair_counts[pair] -= c * f
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                        pair_words.pop(pair, None)
                for pair, c in new_pairs.items():
                    pair_counts[pair] += c * f
                    pair_words[pair].add(wi)

            pair_counts.pop(best, None)
            if verbose and (len(merges) % 512 == 0 or len(merges) == target):
                print(f"  merges {len(merges):>5}  pair {best} x{count}")

        return cls(merges)

    # --------------------------------------------------------------- encode --
    def _encode_word(self, ids: list[int]) -> list[int]:
        ids = list(ids)
        while len(ids) > 1:
            best_rank = None
            best_at = -1
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_at = i
            if best_at < 0:
                break
            pair = (ids[best_at], ids[best_at + 1])
            ids = ids[:best_at] + [256 + self.rank[pair]] + ids[best_at + 2:]
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for word in text.split():
            ids = self._encode_word(list(word.encode("utf-8")))
            if out:
                out.append(32)  # keep the space as a plain byte
            out.extend(ids)
        return out

    # --------------------------------------------------------------- decode --
    def _expand(self, tid: int, out: list[int]) -> None:
        if tid < 256:
            out.append(tid)
        else:
            a, b = self.merges[tid - 256]
            self._expand(a, out)
            self._expand(b, out)

    def decode(self, ids: list[int]) -> str:
        raw: list[int] = []
        for tid in ids:
            self._expand(tid, raw)
        return bytes(raw).decode("utf-8", errors="replace")

    # ------------------------------------------------------------ persistence --
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPE":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]])


def measure(text: str, bpe: BPE, label: str) -> dict:
    """How much text fits in a 256-token context, before and after."""
    n_bytes = len(text.encode("utf-8"))
    n_bpe = len(bpe.encode(text))
    n_chars = len(text)
    return {
        "label": label,
        "chars": n_chars,
        "bytes": n_bytes,
        "bpe_tokens": n_bpe,
        "bytes_per_token": round(n_bytes / max(n_bpe, 1), 2),
        "chars_fit_256_byte_level": round(256 / max(n_bytes / max(n_chars, 1), 1e-9)),
        "chars_fit_256_bpe": round(256 * n_chars / max(n_bpe, 1)),
        "compression_vs_bytes": round(n_bytes / max(n_bpe, 1), 2),
    }


if __name__ == "__main__":
    hindi = (
        "भारत की राजधानी नई दिल्ली है। यह देश का सबसे बड़ा शहर नहीं है, "
        "लेकिन राजनीतिक और प्रशासनिक दृष्टि से सबसे महत्वपूर्ण है। "
        "यहाँ संसद, राष्ट्रपति भवन और सुप्रीम कोर्ट स्थित हैं। "
    )
    bpe = BPE.train(hindi * 50, vocab_size=512)
    print(f"\nvocab {bpe.vocab_size}")
    for text, label in ((hindi, "Hindi"),):
        m = measure(text, bpe, label)
        print(f"{label}: {m['chars_fit_256_byte_level']} chars (bytes) -> "
              f"{m['chars_fit_256_bpe']} chars (BPE)")
        back = bpe.decode(bpe.encode(text))
        print("round trip:", "exact" if back.strip() == text.strip() else "MISMATCH")
