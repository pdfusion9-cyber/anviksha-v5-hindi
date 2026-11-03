"""Answer through retrieval and rules first, the model only to phrase.

The model cannot hold facts at this size, so it is not asked to. Correctness
comes from outside:

    1. look the question up in a knowledge base of verified answers
    2. if a stored answer matches, return it -- it is the checked answer, so it
       is correct by construction, not by the model's opinion
    3. if nothing matches, let the model write, then check the result against the
       rule files and the security judge
    4. if the check fails, refuse rather than ship something unverified

That is the "control from outside, once" arrangement: rules and facts are loaded
from files, and every answer afterwards passes through them automatically.

Retrieval is deliberately simple -- token overlap plus character n-grams. It is
inspectable, it has no parameters, and when it is not confident it says so
instead of guessing.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rule_engine import check as check_rules  # noqa: E402

DEVANAGARI = re.compile(r"[\u0900-\u097f]")
WORD = re.compile(r"[\u0900-\u097f]+|[A-Za-z][A-Za-z'-]+")
STOPWORDS = {
    "का", "की", "के", "को", "में", "से", "पर", "और", "है", "हैं", "था", "थी",
    "क्या", "कौन", "कैसे", "कहाँ", "कब", "क्यों", "कि", "यह", "वह", "एक", "हो",
    "the", "a", "an", "is", "are", "what", "which", "how", "of", "to", "in",
}


def tokens(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text or "") if w.lower() not in STOPWORDS}


def ngrams(text: str, n: int = 3) -> set[str]:
    text = re.sub(r"\s+", " ", (text or "").lower())
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def similarity(a: str, b: str, idf: dict[str, float] | None = None) -> float:
    """Similarity that ignores the instruction template.

    Instruction corpora repeat the same wording across thousands of rows --
    "Translate the following from English into Hinglish." appears with every
    payload. Plain token overlap then scores two different questions as nearly
    identical, and retrieval confidently returns the wrong stored answer. So
    tokens are weighted by how rare they are: the template words that appear
    everywhere count for almost nothing, and the payload that differs counts.
    """
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0

    def weight(tok: str) -> float:
        return (idf or {}).get(tok, 1.0)

    shared = ta & tb
    union = ta | tb
    w_shared = sum(weight(t) for t in shared)
    w_union = sum(weight(t) for t in union)
    weighted_jaccard = w_shared / w_union if w_union else 0.0

    ga, gb = ngrams(a), ngrams(b)
    dice = 2 * len(ga & gb) / max(len(ga) + len(gb), 1)
    return 0.65 * weighted_jaccard + 0.35 * dice


NUMBER = re.compile(r"\d+")


def consistent_with_question(question: str, answer: str,
                             idf: dict[str, float] | None = None) -> tuple[bool, str]:
    """Would this stored answer actually be answering *this* question?

    Similarity alone cannot tell "cancel alarm for 3" from "set alarm for 5":
    the wording matches and the payload does not. Two deterministic checks catch
    that class of error, and both are facts about the text:

      * every number in the question appears in the answer
      * the answer mentions at least half of the question's distinctive words

    A stored answer that fails either is not an answer to this question, however
    similar the question looked.
    """
    q_numbers = set(NUMBER.findall(question))
    a_numbers = set(NUMBER.findall(answer))
    missing = q_numbers - a_numbers
    if missing:
        return False, f"question asks about {sorted(missing)}, answer never mentions it"

    q_tokens = tokens(question)
    if not q_tokens:
        return True, ""
    # only the distinctive words matter; template words are shared by everything
    distinctive = {t for t in q_tokens if (idf or {}).get(t, 1.0) >= 1.5} or q_tokens
    present = {t for t in distinctive if t in tokens(answer)}
    if len(present) / len(distinctive) < 0.5:
        return False, (f"answer covers {len(present)}/{len(distinctive)} of the "
                       f"question's distinctive words")
    return True, ""


def build_idf(questions: list[str]) -> dict[str, float]:
    """Inverse document frequency over the knowledge base questions."""
    import math
    df: dict[str, int] = {}
    for q in questions:
        for tok in tokens(q):
            df[tok] = df.get(tok, 0) + 1
    n = max(len(questions), 1)
    return {tok: math.log((n + 1) / (count + 1)) + 1.0 for tok, count in df.items()}


@dataclass
class Hit:
    question: str
    answer: str
    score: float
    source: str = ""


@dataclass
class KnowledgeBase:
    rows: list[dict] = field(default_factory=list)
    idf: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return cls(rows, build_idf([r["user"] for r in rows]))

    def search(self, question: str, k: int = 3) -> list[Hit]:
        scored = [(similarity(question, r["user"], self.idf), r) for r in self.rows]
        scored.sort(key=lambda x: -x[0])
        return [Hit(r["user"], r["assistant"], round(s, 3), r.get("source", ""))
                for s, r in scored[:k]]


@dataclass
class Answer:
    text: str | None
    via: str                     # "knowledge_base" | "model" | "refused"
    confidence: float
    hits: list[Hit] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"answer": self.text, "via": self.via,
                "confidence": self.confidence,
                "hits": [h.__dict__ for h in self.hits],
                "violations": self.violations}


def answer(question: str, kb: KnowledgeBase | None, generate=None,
           match_threshold: float = 0.62, exact_threshold: float = 0.95,
           judge=None) -> Answer:
    """Retrieve, then rules, then the model. Never return something unchecked."""
    hits = kb.search(question, k=3) if kb else []

    # 1. a stored answer that matches well enough is the answer: it was checked
    #    before it was stored, so returning it is correct by construction
    if hits and hits[0].score >= match_threshold:
        best = hits[0]
        report = check_rules(best.answer)
        if not report.passed:
            return Answer(None, "refused", best.score, hits,
                          [v.detail for v in report.errors])
        # A near-identical question is the same question, so its stored answer is
        # the answer: "2022 मास्टर्स गोल्फ किसने जीता?" -> "स्कॉटी शेफलर" is
        # correct even though the answer never repeats "2022". Below that, the
        # match is a lookalike and the answer has to prove it addresses the
        # question before it is returned as a checked fact.
        if best.score >= exact_threshold:
            return Answer(best.answer, "knowledge_base", best.score, hits)
        ok, why = consistent_with_question(question, best.answer,
                                           kb.idf if kb else None)
        if not ok:
            return Answer(None, "refused", best.score, hits, [why])
        return Answer(best.answer, "knowledge_base", best.score, hits)

    # 2. nothing good enough stored: let the model write, then check it
    if generate is None:
        return Answer(None, "refused", hits[0].score if hits else 0.0, hits,
                      ["no matching stored answer and no model available"])

    text = generate(question)
    if not text:
        return Answer(None, "refused", 0.0, hits, ["model produced nothing"])

    violations = [v.detail for v in check_rules(text).errors]
    if judge is not None:
        verdict = judge(text, hits[0].answer if hits else None)
        if not verdict.passed:
            violations.extend(verdict.reasons)

    if violations:
        return Answer(None, "refused", hits[0].score if hits else 0.0, hits,
                      violations)

    return Answer(text, "model", hits[0].score if hits else 0.0, hits)


if __name__ == "__main__":
    kb_path = Path("data/multi_train.jsonl")
    if not kb_path.exists():
        raise SystemExit(f"need a knowledge base at {kb_path}")

    kb = KnowledgeBase.load(kb_path)
    print(f"knowledge base: {len(kb.rows)} verified answers\n")

    probes = [r["user"] for r in kb.rows[:5]]
    probes += ["भारत की राजधानी क्या है?", "यह सवाल बिल्कुल नया है जो कभी नहीं पूछा गया"]

    for q in probes:
        result = answer(q, kb)
        top = result.hits[0].score if result.hits else 0.0
        print(f"Q: {q[:56]}")
        print(f"   via={result.via:14} confidence={result.confidence:.2f} (top hit {top:.2f})")
        if result.text:
            print(f"   A: {result.text[:90]}")
        else:
            print(f"   refused: {'; '.join(result.violations[:2])[:90]}")
        print()
