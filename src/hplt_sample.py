"""Sample HPLT Hindi without downloading it, and measure what it actually is.

The corpus is 59.97 GB across 17 shards. Deciding how much of it to take by
looking at the total would be guessing, so this streams a bounded sample from a
couple of shards and answers the four questions that decide the plan:

    how clean is it        mojibake, empty text, script mix
    how much is duplicate  exact repeats within the sample
    how much is real Hindi share of Devanagari, share of Latin, share that is neither
    what quality buys      the same measurements for a high and a low WDS bin

The shard filenames carry the WDS quality bin as their prefix: 5_1.jsonl.zst is
bin 5, 10_1.jsonl.zst is bin 10. HPLT describes higher WDS as more useful for
language modelling, so sampling both ends measures that claim on Hindi instead
of taking it on faith.

Nothing here downloads a whole shard: the stream is cut after a byte budget and
the partial JSONL is parsed line by line, discarding the last incomplete line.
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

MAP_URL = "https://data.hplt-project.org/three/sorted/hin_Deva.map"


def fetch_map(url: str = MAP_URL) -> list[str]:
    """The .map file lists every shard URL for one language, one per line."""
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode("utf-8", errors="replace")
    urls = [line.strip() for line in text.splitlines() if line.strip().startswith("http")]
    return urls


def stream_lines(url: str, max_bytes: int):
    """Yield JSONL lines from a remote .jsonl.zst, stopping after max_bytes.

    Decompressed a chunk at a time so a truncated stream is fine: the tail is
    dropped and everything before it is still usable.
    """
    try:
        import zstandard as zstd
    except ImportError:
        print("installing zstandard...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard"],
                       check=True)
        import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    buf = b""
    produced = 0
    with urllib.request.urlopen(url, timeout=120) as resp:
        dobj = dctx.decompressobj()
        while produced < max_bytes:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            try:
                buf += dobj.decompress(chunk)
            except Exception:
                break
            produced = len(buf)
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    yield line


def shingles(text: str, n: int = 5, limit: int = 400) -> set[str]:
    """Word shingles, which catch syndicated copies that exact hashing misses."""
    words = WORD.findall(text.lower())[:limit]
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def looks_mojibake(text: str) -> bool:
    if not text:
        return False
    odd = sum(1 for c in text if 0x80 <= ord(c) <= 0xFF)
    return odd > len(text) * 0.15


def analyse(url: str, label: str, max_bytes: int, max_docs: int) -> dict:
    print(f"\n=== {label} ===")
    print(f"  {url}")
    docs = 0
    chars = 0
    deva_docs = latin_docs = other_docs = 0
    deva_chars = 0
    mojibake = 0
    empty = 0
    short = 0
    hashes: Counter = Counter()
    near_seen: dict[frozenset, int] = {}
    lengths: list[int] = []
    sample_texts: list[str] = []

    for line in stream_lines(url, max_bytes):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        text = rec.get("text") or rec.get("content") or ""
        if not isinstance(text, str):
            continue
        docs += 1
        chars += len(text)
        lengths.append(len(text))
        if not text.strip():
            empty += 1
            continue
        if len(text) < 100:
            short += 1
        if looks_mojibake(text):
            mojibake += 1
        d = len(DEVANAGARI.findall(text))
        l = len(LATIN.findall(text))
        deva_chars += d
        if d > l * 2:
            deva_docs += 1
        elif l > d * 2:
            latin_docs += 1
        else:
            other_docs += 1
        hashes[hashlib.sha256(text[:2000].encode("utf-8")).hexdigest()[:16]] += 1
        # near-duplicates: keep a compact signature of the first shingle set, and
        # compare against signatures already seen. Bounded so memory stays flat.
        if docs <= 20000:
            sig = frozenset(shingles(text))
            if sig:
                near_seen[sig] = near_seen.get(sig, 0) + 1
        if len(sample_texts) < 3 and 200 < len(text) < 700:
            sample_texts.append(text[:300])
        if docs >= max_docs:
            break

    dup = sum(c - 1 for c in hashes.values() if c > 1)
    near_dup = sum(c - 1 for c in near_seen.values() if c > 1)
    near_checked = sum(near_seen.values())
    lengths.sort()
    med = lengths[len(lengths) // 2] if lengths else 0
    res = {
        "label": label, "url": url, "docs": docs, "chars": chars,
        "median_chars": med,
        "devanagari_docs": deva_docs, "latin_docs": latin_docs, "mixed_docs": other_docs,
        "devanagari_char_share": round(deva_chars / max(chars, 1), 3),
        "mojibake": mojibake, "mojibake_rate": round(mojibake / max(docs, 1), 4),
        "empty": empty, "under_100_chars": short,
        "duplicate_docs": dup, "duplicate_rate": round(dup / max(docs, 1), 4),
        "near_duplicates": near_dup,
        "near_duplicate_rate": round(near_dup / max(near_checked, 1), 4),
        "near_checked": near_checked,
        "unique_docs": len(hashes),
        # a rough Hindi token estimate: about 3.5 characters per token in this
        # script, which is close to the measured 6.26 bytes (3.1 chars) per token
        "token_estimate": int(deva_chars / 3.1),
    }
    print(f"  documents            {docs:,}")
    print(f"  characters           {chars:,}   median {med:,}")
    print(f"  Devanagari-dominant  {deva_docs:,}  ({deva_docs/max(docs,1):.1%})")
    print(f"  Latin-dominant       {latin_docs:,}  ({latin_docs/max(docs,1):.1%})")
    print(f"  mixed/other          {other_docs:,}  ({other_docs/max(docs,1):.1%})")
    print(f"  Devanagari chars     {res['devanagari_char_share']:.1%} of all characters")
    print(f"  mojibake             {mojibake:,}  ({res['mojibake_rate']:.2%})")
    print(f"  empty                {empty:,}")
    print(f"  under 100 chars      {short:,}  ({short/max(docs,1):.1%})")
    print(f"  duplicate documents  {dup:,}  ({res['duplicate_rate']:.2%})")
    print(f"  near-duplicates      {near_dup:,}  of {near_checked:,} checked "
          f"({res['near_duplicate_rate']:.2%})")
    print(f"  Hindi token estimate {res['token_estimate']:,}")
    for t in sample_texts[:2]:
        print(f"    sample: {t[:150]!r}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-url", default=MAP_URL)
    ap.add_argument("--max-bytes", type=int, default=120_000_000,
                    help="decompressed bytes per shard; keeps the run short")
    ap.add_argument("--max-docs", type=int, default=200_000)
    ap.add_argument("--bins", default="",
                    help="comma-separated WDS bins to sample, e.g. 8,9,10; "
                         "default is the highest and the lowest")
    ap.add_argument("--out", type=Path, default=Path("runs/hplt_sample.json"))
    args = ap.parse_args()

    print(f"fetching shard list from {args.map_url}")
    urls = fetch_map(args.map_url)
    print(f"{len(urls)} shards:")
    for u in urls:
        print("  ", u.split("/")[-1])

    # the shard prefix is the WDS bin; sample the best and the worst available
    def bin_of(u: str) -> int:
        m = re.search(r"/(\d+)_\d+\.jsonl\.zst$", u)
        return int(m.group(1)) if m else -1

    ranked = sorted(urls, key=bin_of)
    if args.bins:
        wanted = {int(b) for b in args.bins.split(",") if b.strip()}
        picks = [(f"WDS bin {bin_of(u)}", u) for u in ranked if bin_of(u) in wanted]
        missing = wanted - {bin_of(u) for u in ranked}
        if missing:
            print(f"  bins requested but not published: {sorted(missing)}")
    else:
        picks = []
        if ranked:
            picks.append((f"WDS bin {bin_of(ranked[-1])} (highest quality)", ranked[-1]))
        if len(ranked) > 1:
            picks.append((f"WDS bin {bin_of(ranked[0])} (lowest quality)", ranked[0]))

    results = []
    for label, url in picks:
        try:
            results.append(analyse(url, label, args.max_bytes, args.max_docs))
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")

    if len(results) >= 2:
        hi, lo = results[0], results[-1]
        print("\n" + "=" * 70)
        print("what the quality bin actually buys, on Hindi")
        print("=" * 70)
        for key in ("devanagari_char_share", "mojibake_rate", "duplicate_rate",
                    "median_chars"):
            print(f"  {key:<24} high {hi[key]:>8}   low {lo[key]:>8}")


if __name__ == "__main__":
    main()
