"""Split the security corpus into two halves that share nothing.

The requirement is two models trained on different data so the merge has
something to combine. That only works if the halves are genuinely disjoint: if
both see the same documents, the merge adds two copies of one thing and the
expert count doubles while the knowledge does not.

So the split is by subject, not by row. Half A is how to defend -- controls,
architecture, configuration. Half B is how defences fail -- incidents, post
mortems, what broke and why. A document goes to exactly one side, decided by
which vocabulary it uses, and the overlap is measured and reported rather than
assumed to be zero.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# vocabulary that marks a document as being about how to defend
DEFEND = {
    "control", "controls", "mitigate", "mitigation", "harden", "hardening",
    "configure", "configuration", "policy", "policies", "baseline", "framework",
    "architecture", "segmentation", "least privilege", "mfa", "encryption",
    "patch", "patching", "monitor", "monitoring", "logging", "audit",
    "authentication", "authorisation", "authorization", "certificate",
    "firewall", "allowlist", "deny", "rotate", "rotation", "backup",
    "recovery", "incident response", "playbook", "standard", "nist", "iso",
    "cis", "mitre", "att&ck", "soc", "siem", "edr", "dlp", "waf",
}

# vocabulary that marks a document as being about how defences fail
FAIL = {
    "breach", "incident", "attack", "attacker", "exploit", "exploited",
    "vulnerability", "cve", "zero-day", "leak", "leaked", "exfiltrate",
    "exfiltration", "compromise", "compromised", "hijack", "phishing",
    "malware", "ransomware", "worm", "trojan", "backdoor", "rootkit",
    "injection", "jailbreak", "bypass", "evade", "evasion", "poison",
    "poisoned", "spoof", "impersonate", "deepfake", "stolen", "exposed",
    "post-mortem", "postmortem", "root cause", "lessons learned",
}


def score(text: str) -> tuple[int, int]:
    low = text.lower()
    d = sum(1 for w in DEFEND if w in low)
    f = sum(1 for w in FAIL if w in low)
    return d, f


def split(documents: list[str], min_score: int = 1) -> tuple[list[str], list[str], dict]:
    a: list[str] = []
    b: list[str] = []
    skipped = 0
    for doc in documents:
        d, f = score(doc)
        if d == 0 and f == 0:
            skipped += 1
            continue
        if d > f and d >= min_score:
            a.append(doc)
        elif f > d and f >= min_score:
            b.append(doc)
        else:
            skipped += 1          # a tie belongs to neither half
    # overlap: documents that would have qualified for both
    overlap = sum(1 for doc in a if score(doc)[1] >= min_score)
    report = {
        "total": len(documents), "half_a": len(a), "half_b": len(b),
        "skipped": skipped,
        "a_docs_that_also_look_like_b": overlap,
        "overlap_fraction": round(overlap / max(len(a), 1), 3),
        "a_chars": sum(len(d) for d in a),
        "b_chars": sum(len(d) for d in b),
    }
    return a, b, report


def load_corpus(paths: list[Path]) -> list[str]:
    docs: list[str] = []
    for p in paths:
        if not p.is_file():
            continue
        if p.suffix == ".jsonl":
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = (rec.get("assistant") or rec.get("text")
                        or rec.get("content") or "")
                if isinstance(text, str) and len(text) > 40:
                    docs.append(text)
        else:
            for para in re.split(r"\n\s*\n", p.read_text(encoding="utf-8",
                                                       errors="ignore")):
                if len(para) > 120:
                    docs.append(para.strip())
    return docs


def main() -> None:
    ap_paths = [Path(a) for a in sys.argv[1:]]
    docs = load_corpus(ap_paths)
    if not docs:
        print("no documents found; pass one or more corpus files")
        return
    a, b, report = split(docs)
    out = Path("data/security")
    out.mkdir(parents=True, exist_ok=True)
    (out / "half_a_defend.txt").write_text("\n\n".join(a), encoding="utf-8")
    (out / "half_b_failures.txt").write_text("\n\n".join(b), encoding="utf-8")
    (out / "split_report.json").write_text(json.dumps(report, indent=2))

    print("=" * 70)
    print("security corpus split: defending versus failing")
    print("=" * 70)
    for k, v in report.items():
        print(f"  {k:<32} {v}")
    print()
    if report["overlap_fraction"] > 0.2:
        print("  WARNING: the halves overlap heavily. A merge of two models")
        print("  trained on the same material doubles the parameters and not")
        print("  the knowledge, which is the failure this split exists to avoid.")
    else:
        print("  the halves are distinct enough for a merge to add knowledge")
    print(f"\n  written to {out}/half_a_defend.txt and half_b_failures.txt")


if __name__ == "__main__":
    main()
