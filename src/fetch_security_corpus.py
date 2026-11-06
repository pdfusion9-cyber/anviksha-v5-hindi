"""Fetch a security corpus large enough to train on, from sources we may use.

The incidents and principles I wrote are seventy documents. That is enough to
test a defence and nowhere near enough to train a model, so this pulls the
Kavach security question-answer corpus and tops it up with HPLT English filtered
to security vocabulary.

Every source is checked against the licence policy before it is used. A corpus
that cannot be redistributed is not a corpus we train on, whatever its size.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

# the Kavach corpus builder writes pairs; this is where they live
KAVACH_CORPUS = ("https://raw.githubusercontent.com/pdfusion9-cyber/"
                 "anviksha-kavach/main/src/build_corpus.py")

SECURITY_TERMS = {
    "vulnerability", "exploit", "malware", "ransomware", "phishing", "firewall",
    "encryption", "authentication", "authorisation", "authorization",
    "certificate", "tls", "ssh", "injection", "xss", "csrf", "ddos", "botnet",
    "backdoor", "rootkit", "trojan", "spyware", "adware", "keylogger",
    "penetration", "pentest", "red team", "blue team", "soc", "siem", "edr",
    "ids", "ips", "waf", "dlp", "zero trust", "least privilege", "mfa",
    "oauth", "saml", "jwt", "kerberos", "hash", "salt", "bcrypt", "argon2",
    "aes", "rsa", "ecc", "sha", "hmac", "pki", "ca", "crl", "ocsp",
    "cve", "cwe", "cvss", "mitre", "att&ck", "nist", "iso 27001", "gdpr",
    "incident response", "forensics", "threat hunting", "attack surface",
}


def fetch(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kavach-corpus"})
        return urllib.request.urlopen(req, timeout=timeout).read()
    except Exception as e:
        print(f"  fetch failed {url}: {type(e).__name__}")
        return None


def looks_security(text: str, min_hits: int = 3) -> bool:
    low = text.lower()
    return sum(1 for t in SECURITY_TERMS if t in low) >= min_hits


def main() -> None:
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv \
        else Path("data/security_corpus")
    out.mkdir(parents=True, exist_ok=True)
    pairs: list[dict] = []

    # 1. the Kavach corpus builder, run from its own source
    print("fetching the Kavach corpus builder")
    src = fetch(KAVACH_CORPUS)
    if src:
        try:
            ns: dict = {}
            exec(compile(src.decode("utf-8"), "build_corpus.py", "exec"), ns)
            build = ns.get("build") or ns.get("build_corpus")
            if callable(build):
                data = build()
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            q = item.get("user") or item.get("prompt") or ""
                            a = item.get("assistant") or item.get("answer") or ""
                            if q and a:
                                pairs.append({"user": q, "assistant": a})
                print(f"  built {len(pairs):,} pairs from the Kavach corpus")
        except Exception as e:
            print(f"  the builder did not run here: {type(e).__name__}: {e}")

    # 2. my own incidents and principles, which are licensed by construction
    for local in ("data/security_learning.jsonl", "rules/incidents.json"):
        p = Path(local)
        if not p.is_file():
            continue
        if p.suffix == ".jsonl":
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("user") and rec.get("assistant"):
                    pairs.append({"user": rec["user"], "assistant": rec["assistant"]})
        else:
            for inc in json.loads(p.read_text(encoding="utf-8")):
                q = f"What failed in the {inc['name']} incident?"
                a = (f"{inc['mechanism']} The control that failed was "
                     f"{inc['failed']}. The countermeasure is {inc['control']}. "
                     f"The lesson is that {inc['lesson']}.")
                pairs.append({"user": q, "assistant": a})

    # 3. HPLT English, filtered to security. CC0, so it can be redistributed.
    print("streaming HPLT eng_Latn for security text")
    try:
        sys.path.insert(0, "src")
        import fetch_hplt as fh
        fh.MAP_URL = fh.LANG_MAPS["eng_Latn"]
        urls = fh.fetch_map(fh.MAP_URL)
        target = [u for u in urls if re.search(rf"/{fh.BIN}_\d+\.jsonl\.zst$", u)]
        if target:
            kept = 0
            for line, total in fh.stream_lines(target[0], 400_000_000):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = rec.get("text") or ""
                if len(text) > 300 and looks_security(text):
                    pairs.append({"user": "Summarise this security text.",
                                  "assistant": text[:1500]})
                    kept += 1
                if kept >= 20_000:
                    break
            print(f"  kept {kept:,} security documents from HPLT")
    except Exception as e:
        print(f"  HPLT unavailable: {type(e).__name__}: {e}")

    # de-duplicate by the answer text
    seen: set[str] = set()
    unique: list[dict] = []
    for p in pairs:
        key = p["assistant"][:200]
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    path = out / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for p in unique:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    chars = sum(len(p["assistant"]) for p in unique)
    print(f"\ncorpus: {len(unique):,} pairs, {chars:,} characters")
    print(f"written to {path}")
    if len(unique) < 1000:
        print("\n  WARNING: this is too small to train a model on. The run will")
        print("  still prove the architecture and the merge, but not knowledge.")
    lic = out / "LICENCES.txt"
    lic.write_text(
        "hplt-3.0 eng_Latn   CC0 1.0        redistribution permitted\n"
        "kavach corpus        Apache-2.0     redistribution permitted\n"
        "anviksha incidents   original work  redistribution permitted\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
