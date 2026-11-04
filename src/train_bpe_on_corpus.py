"""Learn the BPE merges on a sample of the corpus and report what it bought.

Learning merges does not need the whole corpus -- a few tens of megabytes covers
the frequent word forms, and the merges generalise to the rest. Encoding does
use the whole corpus, and that is fast because the encoder caches per word.

The report answers the one question that matters here: how many Hindi characters
now fit in the model's 192-token window, against the 85 the character tokenizer
managed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpe_tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=Path("runs/bpe4096.json"))
    ap.add_argument("--report", type=Path, default=Path("runs/bpe_train.json"))
    ap.add_argument("--window", type=int, default=192)
    args = ap.parse_args()

    text = args.text.read_text(encoding="utf-8")
    print(f"learning merges from {len(text):,} characters")

    t0 = time.time()
    tok = BPETokenizer.train(text, vocab_size=args.vocab)
    train_s = time.time() - t0

    probe_hi = "भारत की राजधानी नई दिल्ली है। यह देश का सबसे बड़ा शहर नहीं है।"
    probe_en = "The capital of India is New Delhi."
    hi_ids = tok.encode(probe_hi)
    en_ids = tok.encode(probe_en)

    report = {
        "vocab": tok.size,
        "merges": len(tok.bpe.merges),
        "train_seconds": round(train_s, 1),
        "sample_chars": len(text),
        "round_trip_hindi": tok.decode(hi_ids) == probe_hi,
        "round_trip_english": tok.decode(en_ids) == probe_en,
        "hindi_bytes_per_token": round(len(probe_hi.encode("utf-8")) / max(len(hi_ids), 1), 2),
        "hindi_chars_per_window": int(args.window * len(probe_hi) / max(len(hi_ids), 1)),
        "english_chars_per_window": int(args.window * len(probe_en) / max(len(en_ids), 1)),
        "char_tokenizer_chars_per_window": args.window,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nHindi per {args.window}-token window: "
          f"{report['hindi_chars_per_window']} chars "
          f"(character tokenizer managed {report['char_tokenizer_chars_per_window']})")


if __name__ == "__main__":
    main()
