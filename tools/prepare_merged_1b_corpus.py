from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

DEVANAGARI = re.compile(r"[\u0900-\u097f]")
EN_SECURITY = {
    "cryptography", "encryption", "cipher", "firewall", "packet", "network",
    "malware", "ransomware", "phishing", "vulnerability", "exploit", "injection",
    "authentication", "authorization", "access control", "incident response",
    "forensics", "threat", "security", "password", "certificate", "tls", "ssh",
    "sql", "xss", "csrf", "ddos", "hash", "salt", "zero trust", "privacy",
}
HI_SECURITY = {
    "साइबर", "सुरक्षा", "नेटवर्क", "फायरवॉल", "मैलवेयर", "रैंसमवेयर",
    "फ़िशिंग", "फिशिंग", "कूटलेखन", "एन्क्रिप्शन", "पासवर्ड", "प्रमाणीकरण",
    "अभिगम नियंत्रण", "डेटा", "गोपनीयता", "भेद्यता", "दुर्भावनापूर्ण",
    "हमला", "घुसपैठ", "सुरक्षा घटना", "प्रतिक्रिया", "क्रिप्टोग्राफी",
    "सर्वर", "ट्रैफिक", "ट्रैफ़िक", "एप्लिकेशन", "इंजेक्शन",
}


def iter_lines(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--english", type=Path, required=True,
                        help="HPLT eng_Latn text output from fetch_hplt.py")
    parser.add_argument("--hindi", type=Path, required=True,
                        help="HPLT hin_Deva text output from fetch_hplt.py")
    parser.add_argument("--seed", type=Path, default=Path("data/chat_seed.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-english", type=int, default=12000)
    parser.add_argument("--max-hindi", type=int, default=12000)
    parser.add_argument("--min-english-docs", type=int, default=500)
    parser.add_argument("--min-hindi-docs", type=int, default=500)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = Counter()
    char_counts = Counter()
    security_hits = Counter()

    def write_doc(handle, text: str, language: str, source: str,
                  require_security: bool = False) -> bool:
        text = " ".join(text.split())
        if len(text) < 120:
            return False
        if language == "hi":
            dev_share = len(DEVANAGARI.findall(text)) / max(len(text), 1)
            if dev_share < 0.35:
                return False
            terms = HI_SECURITY
        else:
            low = text.lower()
            terms = EN_SECURITY
        hits = sum(1 for term in terms if term in (text if language == "hi" else low))
        if require_security and hits < 1:
            return False
        digest = hashlib.sha256(text[:3000].encode("utf-8")).hexdigest()
        if digest in seen:
            return False
        seen.add(digest)
        handle.write(json.dumps({
            "text": text, "language": language, "source": source,
            "security_keyword_hits": hits,
        }, ensure_ascii=False) + "\n")
        counts[language] += 1
        char_counts[language] += len(text)
        if hits:
            security_hits[language] += 1
        return True

    with args.out.open("w", encoding="utf-8") as output:
        for text in iter_lines(args.english):
            if counts["en"] >= args.max_english:
                break
            low = text.lower()
            hits = sum(1 for term in EN_SECURITY if term in low)
            if hits >= 1:
                write_doc(output, text, "en", "HPLT-3.0-eng_Latn-CC0")
        for text in iter_lines(args.hindi):
            if counts["hi"] >= args.max_hindi:
                break
            # Keep quality Hindi prose for language regularisation; mark and
            # report security-related items separately rather than discarding
            # all non-technical Hindi writing.
            write_doc(output, text, "hi", "HPLT-3.0-hin_Deva-CC0")

        # Small in-repository conversational Hindi seed, never the held-out file.
        if args.seed.is_file():
            for line in iter_lines(args.seed):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user = row.get("user")
                assistant = row.get("assistant")
                if isinstance(user, str) and isinstance(assistant, str):
                    write_doc(
                        output,
                        f"प्रश्न: {user} उत्तर: {assistant}",
                        "hi", "repository-chat-seed",
                    )

    manifest = {
        "dataset": "merged-1b-bilingual-eos-continuation",
        "sources": {
            "english": str(args.english),
            "hindi": str(args.hindi),
            "seed": str(args.seed),
            "heldout_used_for_training": False,
        },
        "documents": dict(counts),
        "characters": dict(char_counts),
        "security_keyword_documents": dict(security_hits),
        "unique_document_hashes": len(seen),
        "corpus_bytes": args.out.stat().st_size,
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if counts["en"] < args.min_english_docs:
        raise SystemExit(
            f"English security corpus too small: {counts['en']} < {args.min_english_docs}"
        )
    if counts["hi"] < args.min_hindi_docs:
        raise SystemExit(
            f"Hindi continuation corpus too small: {counts['hi']} < {args.min_hindi_docs}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
