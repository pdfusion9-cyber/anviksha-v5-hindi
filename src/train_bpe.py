"""Train a byte-level BPE on the corpus and report what it buys for Hindi.

The model's vocabulary is 256, so it reads raw bytes, and Devanagari is three
bytes per character. This trains a larger vocabulary on the real corpus and
measures the one number that matters: how many Hindi characters fit in a
256-token context before and after.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpe import BPE, measure  # noqa: E402

HINDI_SAMPLE = (
    "भारत की राजधानी नई दिल्ली है। यह देश का सबसे बड़ा शहर नहीं है, लेकिन "
    "राजनीतिक और प्रशासनिक दृष्टि से सबसे महत्वपूर्ण है। यहाँ संसद, राष्ट्रपति "
    "भवन और सुप्रीम कोर्ट स्थित हैं। दिल्ली का इतिहास हज़ारों साल पुराना है। "
)
ENGLISH_SAMPLE = (
    "The capital of India is New Delhi. It is not the largest city in the country, "
    "but it is the most important politically and administratively. Parliament, the "
    "President's residence and the Supreme Court are here. Delhi's history spans "
    "thousands of years and bears the mark of many dynasties."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/multi_train.jsonl"))
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--max-chars", type=int, default=12_000_000,
                    help="cap the training text so this finishes on a CPU")
    ap.add_argument("--out", type=Path, default=Path("runs/bpe4096.json"))
    ap.add_argument("--report", type=Path, default=Path("runs/bpe_report.json"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    text = "\n".join(r["user"] + "\n" + r["assistant"] for r in rows)
    if len(text) > args.max_chars:
        text = text[: args.max_chars]
    hindi_chars = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    print(f"corpus: {len(rows):,} rows, {len(text):,} chars "
          f"({hindi_chars:,} Devanagari, {hindi_chars / max(len(text), 1):.1%})")

    bpe = BPE.train(text, vocab_size=args.vocab)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpe.save(args.out)
    print(f"trained vocab {bpe.vocab_size}, saved to {args.out}")

    report = {"vocab_size": bpe.vocab_size, "corpus_rows": len(rows),
              "corpus_chars": len(text), "devanagari_share": round(hindi_chars / max(len(text), 1), 3)}
    for sample, label in ((HINDI_SAMPLE, "hindi"), (ENGLISH_SAMPLE, "english")):
        m = measure(sample, bpe, label)
        report[label] = m
        print(f"\n{label}: {m['chars']} chars -> {m['bpe_tokens']} tokens "
              f"({m['bytes_per_token']} bytes/token)")
        print(f"  fits in 256 tokens: byte-level {m['chars_fit_256_byte_level']} chars, "
              f"BPE {m['chars_fit_256_bpe']} chars "
              f"({m['chars_fit_256_bpe'] / max(m['chars_fit_256_byte_level'], 1):.1f}x)")
        back = bpe.decode(bpe.encode(sample))
        report[label]["round_trip_exact"] = back.strip() == sample.strip()
        print(f"  round trip: {'exact' if report[label]['round_trip_exact'] else 'MISMATCH'}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {args.report}")


if __name__ == "__main__":
    main()
