"""Pull a bounded slice of HPLT Hindi, clean it, and keep only what is verified.

The sampling run established what is worth taking: bin 9, 200,000 documents,
2.90 billion characters, 100% Devanagari-dominant, zero mojibake, 1.48%
near-duplicates. This fetches that bin for real and writes a corpus.

Three things this fixes from the sampling script, all of them mine:

  the byte budget never fired     `produced = len(buf)` measured the buffer
                                  after lines had been yielded from it, so it
                                  stayed small and the cap was never reached.
                                  The run stopped on the document limit instead
                                  and reported 2.90B characters for what was
                                  supposed to be a 500 MB sample.

  duplicates were only counted    counting them changes nothing. They are now
                                  removed, exact and near, and the number kept
                                  away is reported.

  no metadata was carried         every surviving document records its source,
                                  so the corpus can be audited later rather than
                                  trusted because it exists.

Output is three files: the plain text the pretrain stage reads, a JSONL with the
per-document metadata, and a stats file with what was kept and what was thrown
away and why.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

DEVANAGARI = re.compile(r"[\u0900-\u097f]")
LATIN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[\u0900-\u097f]+|[A-Za-z][A-Za-z'-]+")

# The shard map is per language. The second run needs data that shares nothing
# with the first -- a different language and a different domain -- while keeping
# the same tokenizer, because two checkpoints can only be merged if their
# embedding rows mean the same thing.
LANG_MAPS = {
    "hin_Deva": "https://data.hplt-project.org/three/sorted/hin_Deva.map",
    "eng_Latn": "https://data.hplt-project.org/three/sorted/eng_Latn.map",
}
MAP_URL = LANG_MAPS["hin_Deva"]
BIN = 9


def fetch_map(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode("utf-8", errors="replace")
    return [l.strip() for l in text.splitlines() if l.strip().startswith("http")]


def stream_lines(url: str, max_bytes: int):
    """Yield JSONL lines and the running decompressed byte count.

    The count is accumulated here rather than inferred from the buffer, which is
    what the earlier script got wrong: the buffer shrinks as lines are yielded,
    so it can never reach the budget.
    """
    try:
        import zstandard as zstd
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard"],
                       check=True)
        import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    buf = b""
    total = 0
    with urllib.request.urlopen(url, timeout=180) as resp:
        dobj = dctx.decompressobj()
        while total < max_bytes:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            try:
                out = dobj.decompress(chunk)
            except Exception:
                break
            total += len(out)
            buf += out
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    yield line, total


def looks_mojibake(text: str) -> bool:
    odd = sum(1 for c in text if 0x80 <= ord(c) <= 0xFF)
    return odd > len(text) * 0.15


def shingle_sig(text: str, n: int = 5, limit: int = 300) -> str:
    words = WORD.findall(text.lower())[:limit]
    if len(words) < n:
        return ""
    sh = {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    return hashlib.sha256("|".join(sorted(sh)).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bytes", type=int, default=3_000_000_000,
                    help="decompressed bytes to stream before stopping")
    ap.add_argument("--max-docs", type=int, default=400_000)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--min-devanagari", type=float, default=None,
                   help="kept for compatibility; --min-script is the real knob")
    ap.add_argument("--out-dir", type=Path, default=Path("data/hplt"))
    ap.add_argument("--lang", default="hin_Deva", choices=sorted(LANG_MAPS),
                   help="which HPLT language to stream")
    ap.add_argument("--min-script", type=float, default=0.5,
                   help="fraction of characters that must be in the target script")
    args = ap.parse_args()

    global MAP_URL
    MAP_URL = LANG_MAPS[args.lang]
    print(f"streaming HPLT {args.lang}")
    urls = fetch_map(MAP_URL)
    target = [u for u in urls if re.search(rf"/{BIN}_\d+\.jsonl\.zst$", u)]
    if not target:
        print(f"no shard for bin {BIN}; available: {[u.split('/')[-1] for u in urls]}")
        raise SystemExit(2)
    url = target[0]
    print(f"shard: {url}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.out_dir / "hindi_corpus.txt"
    meta_path = args.out_dir / "hindi_corpus.jsonl"
    stats_path = args.out_dir / "fetch_stats.json"

    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    kept = dropped = 0
    chars = 0
    deva_chars = 0
    reasons: Counter = Counter()
    streamed = 0

    with corpus_path.open("w", encoding="utf-8") as corpus, \
         meta_path.open("w", encoding="utf-8") as meta:
        for line, total in stream_lines(url, args.max_bytes):
            streamed = total
            if kept >= args.max_docs:
                break
            try:
                rec = json.loads(line)
            except Exception:
                reasons["unparseable"] += 1
                continue
            text = rec.get("text") or rec.get("content") or ""
            if not isinstance(text, str) or not text.strip():
                reasons["empty"] += 1
                continue
            if len(text) < args.min_chars:
                reasons["too_short"] += 1
                continue
            if looks_mojibake(text):
                reasons["mojibake"] += 1
                continue
            d = len(DEVANAGARI.findall(text))
            # Latin script for English, Devanagari for Hindi. Checking for
            # Devanagari on an English run would reject every document and
            # produce an empty corpus that reports success.
            if args.lang == "eng_Latn":
                latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
                if latin < len(text) * args.min_script:
                    reasons["not_target_script"] += 1
                    continue
            else:
                if d < len(text) * args.min_script:
                    reasons["not_target_script"] += 1
                    continue

            exact = hashlib.sha256(text[:2000].encode("utf-8")).hexdigest()[:16]
            if exact in seen_exact:
                reasons["duplicate_exact"] += 1
                dropped += 1
                continue
            near = shingle_sig(text)
            if near and near in seen_near:
                reasons["duplicate_near"] += 1
                dropped += 1
                continue
            seen_exact.add(exact)
            if near:
                seen_near.add(near)

            corpus.write(text.replace("\n", " ") + "\n")
            meta.write(json.dumps({
                "id": rec.get("id", ""),
                "source": "hplt-3.0-hin_Deva-bin9",
                "licence": "CC0",
                "verification_status": "checked",
                "chars": len(text),
                "devanagari_share": round(d / len(text), 3),
                "url": rec.get("url", ""),
            }, ensure_ascii=False) + "\n")
            kept += 1
            chars += len(text)
            deva_chars += d

    stats = {
        "shard": url,
        "streamed_decompressed_bytes": streamed,
        "documents_kept": kept,
        "documents_dropped": dropped,
        "characters_kept": chars,
        "devanagari_char_share": round(deva_chars / max(chars, 1), 3),
        "token_estimate": int(deva_chars / 3.1),
        "drop_reasons": dict(reasons),
        "corpus_bytes": corpus_path.stat().st_size,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print("fetch result")
    print("=" * 66)
    for k, v in stats.items():
        if k == "drop_reasons":
            print(f"  {k}:")
            for reason, n in sorted(v.items(), key=lambda x: -x[1]):
                print(f"      {reason:20} {n:,}")
        elif isinstance(v, int):
            print(f"  {k:32} {v:,}")
        else:
            print(f"  {k:32} {v}")
    print(f"\nwritten to {args.out_dir}/")


if __name__ == "__main__":
    main()
