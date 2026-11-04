"""The retrieval modern systems actually use, as a small honest test.

What modern models do for "answer from a known set" is not exotic: BM25 to get
candidates, then rerank the candidates against the question, then answer only if
the reranker is confident. Dense embeddings are the fashionable variant of the
same pipeline, and BM25 remains the baseline they are measured against because
it is hard to beat on short factual text.

This exists to be compared against the similarity measure this project already
had. Same split, same questions, so the comparison means something.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

WORD = re.compile(r"[\u0900-\u097f]+|[A-Za-z][A-Za-z'-]+|[0-9]+")
NUMBER = re.compile(r"\d+")

STOP = {
    "का", "की", "के", "को", "में", "से", "पर", "और", "है", "हैं", "था", "थी",
    "क्या", "कौन", "कैसे", "कहाँ", "कब", "क्यों", "कि", "यह", "वह", "एक", "हो",
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "what", "which", "who", "how", "when", "where", "why", "do", "does", "did",
    "please", "following", "given", "translate", "sentence",
}


def terms(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text or "") if w.lower() not in STOP]


class BM25:
    """Okapi BM25. k1 controls term-frequency saturation, b the length norm."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [terms(d) for d in docs]
        self.tf = [Counter(d) for d in self.docs]
        self.len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0)
                    for t, c in df.items()}

    def score(self, query: str, i: int) -> float:
        q = terms(query)
        if not q:
            return 0.0
        s = 0.0
        tf, dl = self.tf[i], self.len[i]
        for t in q:
            f = tf.get(t)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
        return s

    def top_k(self, query: str, k: int = 5) -> list[tuple[float, int]]:
        scored = [(self.score(query, i), i) for i in range(len(self.docs))]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


def rerank(question: str, answer: str) -> tuple[float, list[str]]:
    """Score the pair, not just the question.

    BM25 compares question to question. A reranker looks at the answer too, which
    is the step that stops a near-identical question from donating its answer: if
    the answer does not carry the question's specifics, the pair scores low no
    matter how well the questions matched.
    """
    notes: list[str] = []
    score = 1.0

    q_nums = set(NUMBER.findall(question))
    a_nums = set(NUMBER.findall(answer))
    if q_nums - a_nums:
        return 0.0, [f"answer lacks {sorted(q_nums - a_nums)} -- veto"]

    q_terms = set(terms(question))
    if q_terms:
        cover = len(q_terms & set(terms(answer))) / len(q_terms)
        if cover < 0.2:
            score -= 0.45
            notes.append(f"answer covers {cover:.0%} of question terms")

    # an answer with almost no content cannot be verified, so it cannot be trusted
    if len(set(terms(answer))) < 2:
        score -= 0.35
        notes.append("answer has too little content to verify")

    return max(0.0, min(1.0, score)), notes


@dataclass
class Decision:
    answer: str | None
    bm25: float
    rerank: float
    reason: str

    @property
    def answered(self) -> bool:
        return self.answer is not None


def answer_question(question: str, rows: list[dict], bm25: BM25, *,
                    rerank_floor: float = 0.55, k: int = 5) -> Decision:
    cands = bm25.top_k(question, k=k)
    if not cands or cands[0][0] <= 0:
        return Decision(None, 0.0, 0.0, "no candidate shares a term")

    best_score, best_i = cands[0]
    answer = rows[best_i]["assistant"]
    rr, notes = rerank(question, answer)
    if rr < rerank_floor:
        return Decision(None, best_score, rr, "; ".join(notes) or "rerank low")
    return Decision(answer, best_score, rr, "reranked")


def main() -> None:
    import json

    kb_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/multi_train.jsonl")
    test_path = Path(sys.argv[2] if len(sys.argv) > 2 else "data/multi_heldout.jsonl")
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    if not kb_path.is_file():
        print(f"missing {kb_path}")
        raise SystemExit(2)

    rows = [json.loads(l) for l in kb_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"knowledge base: {len(rows):,} rows")
    bm25 = BM25([r["user"] for r in rows])

    # known questions: sample from the base itself
    import random
    rng = random.Random(0)
    known = rng.sample(rows, min(limit, len(rows)))
    ok = wrong = refused = 0
    for r in known:
        d = answer_question(r["user"], rows, bm25)
        if not d.answered:
            refused += 1
        elif d.answer == r["assistant"]:
            ok += 1
        else:
            wrong += 1
    print(f"\nknown questions ({len(known)}):")
    print(f"  correct {ok} ({ok/len(known):.1%})  wrong {wrong}  refused {refused}")

    if test_path.is_file():
        unseen = [json.loads(l) for l in test_path.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
        u_wrong = u_ok = u_ref = 0
        for r in unseen:
            d = answer_question(r["user"], rows, bm25)
            if not d.answered:
                u_ref += 1
            elif d.answer == r["assistant"]:
                u_ok += 1
            else:
                u_wrong += 1
        print(f"\nunseen questions ({len(unseen)}):")
        print(f"  correct {u_ok}  WRONG {u_wrong}  refused {u_ref}")
        print(f"  wrong rate {u_wrong/len(unseen):.2%}")


if __name__ == "__main__":
    main()
