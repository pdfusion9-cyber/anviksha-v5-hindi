"""A knowledge base that refuses a pair it cannot vouch for.

Adding pairs to raise coverage is only safe if every pair carries the evidence
that makes it trustworthy. Four fields are required, and a pair without them does
not enter:

    source               where this came from, specifically enough to check
    verification_status  checked | unverified | disputed
    topic                what it is about, so contradictions can be found
    duplicate_check      the fingerprint used to spot repeats

Contradictions matter more than they look. If the base holds two different
answers to one question, then whichever retrieval happens to return is a coin
flip, and no threshold can fix that -- the answer is not unknown, it is
ambiguous. Those are recorded as disputed and excluded from answering until a
person settles them.

The point of this file is that growth cannot quietly lower quality. A pair that
fails a check is counted and reported, never silently accepted.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

WORD = re.compile(r"[\u0900-\u097f]+|[A-Za-z][A-Za-z'-]+|[0-9]+")
VALID_STATUS = {"checked", "unverified", "disputed"}


def fingerprint(question: str, answer: str) -> str:
    norm = " ".join(WORD.findall(question.lower())) + "|" + \
           " ".join(WORD.findall(answer.lower()))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:20]


def question_key(question: str) -> str:
    return " ".join(WORD.findall(question.lower()))


def answer_key(answer: str) -> str:
    return " ".join(WORD.findall(answer.lower()))


@dataclass
class Pair:
    question: str
    answer: str
    source: str
    verification_status: str
    topic: str
    duplicate_check: str = ""
    disputed_with: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.duplicate_check:
            self.duplicate_check = fingerprint(self.question, self.answer)

    def as_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer,
                "source": self.source, "verification_status": self.verification_status,
                "topic": self.topic, "duplicate_check": self.duplicate_check,
                "disputed_with": self.disputed_with}


class VerifiedBase:
    def __init__(self) -> None:
        self.pairs: dict[str, Pair] = {}       # duplicate_check -> pair
        self.by_question: dict[str, list[str]] = defaultdict(list)
        self.rejected: list[dict] = []
        self.contradictions: dict[str, list[str]] = {}

    # ------------------------------------------------------------ ingest --
    def add(self, question: str, answer: str, *, source: str,
            verification_status: str, topic: str) -> bool:
        """Add a pair, or record why it was refused. Never silently accepts."""
        def reject(reason: str) -> bool:
            self.rejected.append({"question": question[:80], "reason": reason})
            return False

        if not question.strip() or not answer.strip():
            return reject("empty question or answer")
        if not source.strip():
            return reject("no source")
        if verification_status not in VALID_STATUS:
            return reject(f"bad verification_status {verification_status!r}")
        if not topic.strip():
            return reject("no topic")
        if verification_status == "unverified":
            return reject("unverified pairs do not enter the answerable base")

        fp = fingerprint(question, answer)
        if fp in self.pairs:
            return reject("duplicate")

        qk = question_key(question)
        ak = answer_key(answer)
        # same question, different answer: a contradiction, not a duplicate
        for existing_fp in self.by_question.get(qk, []):
            existing = self.pairs[existing_fp]
            if answer_key(existing.answer) != ak:
                self.contradictions.setdefault(qk, [])
                if existing_fp not in self.contradictions[qk]:
                    self.contradictions[qk].append(existing_fp)
                if fp not in self.contradictions[qk]:
                    self.contradictions[qk].append(fp)
                existing.verification_status = "disputed"
                existing.disputed_with.append(fp)

        pair = Pair(question, answer, source, verification_status, topic, fp)
        if qk in self.contradictions:
            pair.verification_status = "disputed"
            pair.disputed_with = [f for f in self.contradictions[qk] if f != fp]
        self.pairs[fp] = pair
        self.by_question[qk].append(fp)
        return True

    def add_many(self, rows: list[dict]) -> dict:
        accepted = 0
        for r in rows:
            ok = self.add(r.get("question", r.get("user", "")),
                          r.get("answer", r.get("assistant", "")),
                          source=r.get("source", ""),
                          verification_status=r.get("verification_status", ""),
                          topic=r.get("topic", ""))
            accepted += int(ok)
        return {"accepted": accepted, "rejected": len(self.rejected)}

    # ------------------------------------------------------------ output --
    def answerable(self) -> list[Pair]:
        """Only checked pairs, never disputed ones."""
        return [p for p in self.pairs.values() if p.verification_status == "checked"]

    def report(self) -> dict:
        status = defaultdict(int)
        for p in self.pairs.values():
            status[p.verification_status] += 1
        reasons = defaultdict(int)
        for r in self.rejected:
            reasons[r["reason"]] += 1
        return {"pairs": len(self.pairs), "by_status": dict(status),
                "answerable": len(self.answerable()),
                "contradicted_questions": len(self.contradictions),
                "rejected": len(self.rejected),
                "rejection_reasons": dict(reasons)}

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps([p.as_dict() for p in self.pairs.values()],
                       ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    print("=" * 72)
    print("verified base: what it accepts and what it refuses")
    print("=" * 72)
    kb = VerifiedBase()

    kb.add("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।",
           source="ncert-class-8-geography-p42", verification_status="checked",
           topic="geography.india")
    kb.add("भारत की राजधानी क्या है?", "भारत की राजधानी मुंबई है।",
           source="forum-post-2021", verification_status="checked",
           topic="geography.india")
    kb.add("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।",
           source="duplicate", verification_status="checked",
           topic="geography.india")
    kb.add("ताजमहल कहाँ है?", "ताजमहल आगरा में है।",
           source="", verification_status="checked", topic="geography.india")
    kb.add("पानी कैसे उबालें?", "बर्तन में पानी डालकर गरम करें।",
           source="cookbook-p12", verification_status="unverified",
           topic="cooking.basics")

    r = kb.report()
    print(f"\n  pairs held            {r['pairs']}")
    print(f"  by status             {r['by_status']}")
    print(f"  answerable (checked)  {r['answerable']}")
    print(f"  contradicted questions {r['contradicted_questions']}")
    print(f"  rejected              {r['rejected']}")
    for reason, n in r["rejection_reasons"].items():
        print(f"     {reason}: {n}")

    print("\n  what this means: the capital question now has two answers, so both")
    print("  are marked disputed and neither will be answered. That is correct --")
    print("  returning either one would be a coin flip dressed as knowledge.")
    for p in kb.pairs.values():
        print(f"   [{p.verification_status:9}] {p.question[:40]:42} <- {p.source}")


if __name__ == "__main__":
    main()
