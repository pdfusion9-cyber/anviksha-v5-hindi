"""A byte-level BPE that works on Hindi without special-casing any script.

The model's vocabulary is 256, which means it reads raw bytes. Devanagari is
three bytes per character in UTF-8, so a 256-token context holds about 85 Hindi
characters -- less than one sentence. That is the single biggest limit on the
Hindi side, and it is a tokenizer problem, not a model problem.

Byte-level BPE keeps the 256 bytes as its base and learns merges on top, so it
still encodes anything -- including text the model has never seen -- while giving
common Devanagari sequences their own token.

Training is on word types with frequencies rather than on raw positions. That is
the standard approach and it is what makes this finish in minutes on a CPU
instead of days.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class BPE:
    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        self.merges: list[tuple[int, int]] = merges or []
        # rank drives encode: lower rank means merged earlier, so applied first
        self.rank = {pair: i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ---------------------------------------------------------------- train --
    @classmethod
    def train(cls, text: str, vocab_size: int = 4096, verbose: bool = True) -> "BPE":
        if vocab_size < 257:
            raise ValueError("vocab_size must be at least 257")

        words = Counter(text.split())
        # each word type as a tuple of byte ids, carried with its frequency
        corpus: dict[tuple[int, ...], int] = {}
        for word, freq in words.items():
            ids = tuple(word.encode("utf-8"))
            if ids:
                corpus[ids] = corpus.get(ids, 0) + freq

        merges: list[tuple[int, int]] = []
        target = vocab_size - 256
        while len(merges) < target:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for ids, freq in corpus.items():
                for a, b in zip(ids, ids[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            best, count = pair_counts.most_common(1)[0]
            if count < 2:
                break

            new_id = 256 + len(merges)
            merges.append(best)

            merged: dict[tuple[int, ...], int] = {}
            a, b = best
            for ids, freq in corpus.items():
                if len(ids) < 2:
                    merged[ids] = merged.get(ids, 0) + freq
                    continue
                out: list[int] = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                key = tuple(out)
                merged[key] = merged.get(key, 0) + freq
            corpus = merged

            if verbose and (len(merges) % 512 == 0 or len(merges) == target):
                total = sum(len(k) * v for k, v in corpus.items())
                print(f"  merges {len(merges):>5}  pair {best} x{count:<8} "
                      f"corpus tokens {total:,}")

        return cls(merges)

    # --------------------------------------------------------------- encode --
    def _encode_word(self, ids: tuple[int, ...]) -> list[int]:
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
            new_id = 256 + self.rank[pair]
            ids = ids[:best_at] + [new_id] + ids[best_at + 2:]
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for word in text.split():
            ids = self._encode_word(tuple(word.encode("utf-8")))
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
        "दिल्ली का इतिहास हज़ारों साल पुराना है और इसमें कई राजवंशों की छाप मिलती है। "
    )
    english = (
        "The capital of India is New Delhi. It is not the largest city in the "
        "country, but it is the most important politically and administratively. "
        "Parliament, the President's residence and the Supreme Court are here. "
        "Delhi's history spans thousands of years and bears the mark of many dynasties. "
    )

    print("training on the sample text...")
    bpe = BPE.train(hindi + "\n" + english, vocab_size=1024)
    print(f"\nvocab size: {bpe.vocab_size}")

    for text, label in ((hindi, "Hindi"), (english, "English")):
        m = measure(text, bpe, label)
        print(f"\n{label}:")
        print(f"  chars {m['chars']}, bytes {m['bytes']}, bpe tokens {m['bpe_tokens']}")
        print(f"  chars that fit in 256 tokens:")
        print(f"    byte-level (vocab 256): {m['chars_fit_256_byte_level']}")
        print(f"    BPE (vocab {bpe.vocab_size}): {m['chars_fit_256_bpe']}")

    # round trip must be exact, or the model is training on corrupted text
    for text, label in ((hindi, "Hindi"), (english, "English")):
        back = bpe.decode(bpe.encode(text))
        print(f"\n{label} round trip: {'exact' if back.strip() == text.strip() else 'MISMATCH'}")
