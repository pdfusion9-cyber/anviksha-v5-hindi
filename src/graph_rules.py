"""Enforce the universal graph rules, for any corpus and any domain.

Every rule here is a measurement. None of them is a list of banned words, and
that is the point: the recurring failure in this project was a hand-written list
that missed the obvious case -- "monitoring" was listed and "monitor" was not,
"approach" was listed and "this approach" was not. A rule stated as a
measurement cannot miss a word it was not told about.

The central one is informativeness, measured as inverse document frequency
against the corpus in front of it. A term that appears in most documents
distinguishes nothing, whatever the term happens to be. "implementation" is
everywhere in security prose and carries no information; "MITRE" and "YARA" are
not and do. The same measurement handles a domain the model has never seen,
because it is computed from that corpus rather than from a list someone wrote
for another one.

This module is the single gate in front of every path that can add an edge:
automatic extraction, a user prompt that states a fact, a model answer being
promoted, and a relation written by hand.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

def _find_rules() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "rules" / "graph_rules.json",
                 here / "rules" / "graph_rules.json",
                 Path("rules/graph_rules.json")):
        if cand.is_file():
            return cand
    return here.parent / "rules" / "graph_rules.json"


DEFAULT_RULES = _find_rules()


@dataclass
class Verdict:
    ok: bool
    rule: str = ""
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class CorpusStats:
    """What the rules need to know about the text they are judging."""
    documents: int = 0
    doc_freq: Counter = field(default_factory=Counter)
    term_freq: Counter = field(default_factory=Counter)
    pos_counts: dict = field(default_factory=dict)

    def add(self, doc_tokens: list[str], pos: list[str] | None = None) -> None:
        self.documents += 1
        self.doc_freq.update(set(doc_tokens))
        self.term_freq.update(doc_tokens)
        if pos:
            for word, tag in zip(doc_tokens, pos):
                slot = self.pos_counts.setdefault(word, Counter())
                slot[tag] += 1

    def verb_share(self, word: str) -> float:
        """How much of this word's use in the corpus is as a verb."""
        counts = self.pos_counts.get(word)
        if not counts:
            return 0.0
        total = sum(counts.values())
        return (counts.get("VERB", 0) + counts.get("AUX", 0)) / total

    def idf(self, term: str) -> float:
        """Inverse document frequency, smoothed.

        A term in every document scores 0. A term in one document out of a
        thousand scores about 6.9. The threshold sits between those, so the
        judgement is made by the corpus and not by me.
        """
        if self.documents == 0:
            return 1.0
        df = self.doc_freq.get(term, 0)
        return math.log((self.documents + 1) / (df + 1)) + 1.0

    def document_share(self, phrase: str) -> float:
        """How much of the corpus this exact phrase appears in."""
        if self.documents == 0:
            return 0.0
        return self.doc_freq.get(phrase, 0) / self.documents


class GraphRules:
    def __init__(self, rules_path: Path | None = None):
        path = rules_path or DEFAULT_RULES
        self.rules = json.loads(path.read_text(encoding="utf-8"))
        t = self.rules.get("thresholds", {})
        self.min_idf = float(t.get("min_idf", 1.2))
        self.min_support = int(t.get("min_support", 3))
        self.max_share = float(t.get("max_document_share", 0.15))
        self.min_chars = int(t.get("min_chars", 3))
        self.max_tokens = int(t.get("max_tokens", 6))
        self.max_verb_share = float(t.get("max_verb_share", 0.5))
        self.stats = CorpusStats()
        self.rejected: Counter = Counter()

    # ---------------------------------------------------------- entity ----
    def check_entity(self, text: str, pos: list[str] | None = None,
                     verb_lemma: str | None = None) -> Verdict:
        t = (text or "").strip().lower()
        tokens = t.split()

        # E5: specific enough to name something
        if len(t) < self.min_chars:
            return self._no("E5", f"{text!r} is too short to name anything")
        if not (1 <= len(tokens) <= self.max_tokens):
            return self._no("E5", f"{text!r} has {len(tokens)} tokens, "
                                  f"outside 1..{self.max_tokens}")

        # E1: an entity is a noun phrase. Acronyms are exempt: a parser guessing
        # at an unfamiliar word tags TLS as a VERB, and rejecting every acronym
        # the model has not seen would throw away exactly the specific terms the
        # rules exist to keep.
        if pos and not _is_acronym(text):
            head = pos[-1]
            if head not in ("NOUN", "PROPN"):
                # A hyphenated compound is one entity and its head is its first
                # element: "man-in-the-middle" ends in a word the parser calls an
                # adjective, but the phrase is a noun. Rejecting it would drop a
                # term the rules exist to keep.
                if "-" in text and pos[0] in ("NOUN", "PROPN"):
                    pass
                elif "-" in text and "NOUN" in pos:
                    pass
                else:
                    return self._no("E1", f"{text!r} ends in {head}, not a noun")

        # E2: not a closed-class word
        closed = {"it", "this", "that", "these", "those", "they", "them", "he",
                  "she", "we", "you", "i", "who", "which", "what", "some", "any",
                  "all", "each", "every", "both", "either", "neither", "the", "a",
                  "an", "and", "or", "but", "if", "when", "while", "because"}
        if tokens and all(w in closed for w in tokens):
            return self._no("E2", f"{text!r} is a referring word, not a thing")

        # E3: not the relation's own verb, compared by stem. "detection detects
        # anomalies" says nothing, and exact-lemma comparison missed it because
        # detect != detection.
        if verb_lemma:
            v = verb_lemma.lower()
            if tokens and (tokens[0] == v or _shares_stem(tokens[0], v)):
                return self._no("E3", f"{text!r} and the verb {v!r} are the "
                                      f"same word in different form")

        # E6: a word this corpus mostly uses as a verb is not an entity, whatever
        # the parser called it here. spaCy tagged the bare "monitor" as PROPN, so
        # a POS check alone let it through; the corpus says the word is a verb,
        # and that measurement is what a hand-written list kept getting wrong.
        if tokens and self.stats.documents > 0:
            share = self.stats.verb_share(tokens[0])
            if share > self.max_verb_share:
                return self._no("E6", f"{text!r} is used as a verb in {share:.0%} "
                                      f"of its appearances (limit "
                                      f"{self.max_verb_share:.0%})")

        # E4: carries information about this corpus
        if self.stats.documents > 0:
            score = self.stats.idf(tokens[0]) if tokens else 0.0
            if score < self.min_idf:
                return self._no("E4", f"{text!r} appears in most documents "
                                      f"(idf {score:.2f} < {self.min_idf}); "
                                      f"it distinguishes nothing here")
        return Verdict(True)

    # -------------------------------------------------------- relation ----
    def check_relation(self, src: str, rel: str, dst: str,
                       support: int = 1,
                       src_pos: list[str] | None = None,
                       dst_pos: list[str] | None = None,
                       verb_lemma: str | None = None) -> Verdict:
        a, b = (src or "").strip().lower(), (dst or "").strip().lower()

        # R1: both endpoints must be entities
        va = self.check_entity(a, src_pos, verb_lemma)
        if not va:
            return self._no("R1", f"subject fails {va.rule}: {va.reason}")
        vb = self.check_entity(b, dst_pos, verb_lemma)
        if not vb:
            return self._no("R1", f"object fails {vb.rule}: {vb.reason}")

        # R2: the endpoints must differ
        if a == b:
            return self._no("R2", "the endpoints are the same")

        # R3: asserted more than once
        if support < self.min_support:
            return self._no("R3", f"support {support} < {self.min_support}")

        # R4: frequency alone is not evidence
        share = max(self.stats.document_share(a), self.stats.document_share(b))
        if share > self.max_share:
            return self._no("R4", f"one endpoint appears in {share:.0%} of "
                                  f"documents (limit {self.max_share:.0%}); "
                                  f"this pair describes the writing, not the subject")

        # R6: not a paraphrase of its own endpoints
        overlap = len(set(a.split()) & set(b.split()))
        if overlap and overlap >= min(len(a.split()), len(b.split())):
            return self._no("R6", f"{a!r} and {b!r} are the same words reordered")
        return Verdict(True)

    # ------------------------------------------------------ preference ----
    @staticmethod
    def prefer_specific(edges: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
        """R5: drop a general relation when a more specific one implies it.

        A--rel-->B is dropped if A--rel-->B-and-more exists, because the longer
        object names what the shorter one gestures at.
        """
        keep: list[tuple[str, str, str]] = []
        for src, rel, dst in edges:
            general = any(
                s == src and r == rel and d != dst
                and d.startswith(dst) and len(d) > len(dst)
                for s, r, d in edges)
            if not general:
                keep.append((src, rel, dst))
        return keep

    def _no(self, rule: str, reason: str) -> Verdict:
        self.rejected[rule] += 1
        return Verdict(False, rule, reason)

    # ----------------------------------------------------------- batch ----
    def learn(self, documents: list[list[str]],
              pos: list[list[str]] | None = None) -> None:
        for i, tokens in enumerate(documents):
            self.stats.add(tokens, pos[i] if pos and i < len(pos) else None)

    def report(self) -> dict:
        return {"documents": self.stats.documents,
                "vocabulary": len(self.stats.term_freq),
                "rejected_by_rule": dict(self.rejected),
                "thresholds": {"min_idf": self.min_idf,
                               "min_support": self.min_support,
                               "max_document_share": self.max_share,
                               "min_chars": self.min_chars,
                               "max_tokens": self.max_tokens}}


def _is_acronym(text: str) -> bool:
    """TLS, MITRE, YARA, SPF: short and all capitals, or dotted."""
    core = text.replace("-", "").replace(".", "").replace("_", "").strip()
    return 2 <= len(core) <= 8 and core.isalpha() and core.isupper()


def _shares_stem(a: str, b: str) -> bool:
    """Cheap stem comparison: a long enough common prefix means the same word."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= max(5, min(len(a), len(b)) - 2)


def tokens_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


if __name__ == "__main__":
    # the rules judged against the thirty-seven relations the real Kavach run
    # kept, so the effect is visible rather than asserted
    import sys
    rules = GraphRules()
    sample = [
        ("implementation", "requires", "careful consideration", 16),
        ("detect function", "requires", "continuous monitoring", 12),
        ("detect function", "requires", "failed login monitoring", 12),
        ("multi-layered approach", "requires", "comprehensive coverage", 9),
        ("threat hunters", "detects", "multi-layered detection strategies", 8),
        ("network segmentation", "prevents", "lateral movement", 7),
        ("monitor", "causes", "proactive identification", 7),
        ("security professionals", "causes", "yara rules", 6),
        ("machine learning algorithms", "detects", "subtle deviations", 5),
        ("accounts", "requires", "authentication", 5),
        ("this approach", "causes", "security teams", 5),
        ("misconfigurations", "causes", "unauthorized access", 4),
        ("security teams", "detects", "malware", 4),
        ("tls", "prevents", "man-in-the-middle", 3),
        ("mitre attck framework", "detects", "as-rep roasting", 3),
        ("forensic tools", "detects", "systematic analysis", 3),
    ]
    # a corpus where the filler terms are everywhere and the real ones are not
    docs = []
    for _ in range(40):
        docs.append(tokens_of(
            "the implementation of this approach requires careful consideration "
            "and comprehensive coverage of security teams monitoring accounts "
            "authentication multi-layered approach"))
    for _ in range(3):
        docs.append(tokens_of(
            "mitre attck framework detects as-rep roasting yara rules lateral "
            "movement network segmentation tls man-in-the-middle"))
    rules.learn(docs)

    print(f"corpus: {rules.stats.documents} documents\n")
    print(f"{'kept':>5}  {'support':>7}  relation")
    kept = rejected = 0
    for src, rel, dst, sup in sample:
        v = rules.check_relation(src, rel, dst, support=sup)
        if v:
            kept += 1
            print(f"  yes  {sup:>7}  {src} --{rel}--> {dst}")
        else:
            rejected += 1
            print(f"   no  {sup:>7}  {src} --{rel}--> {dst}   [{v.rule}] {v.reason[:60]}")
    print(f"\nkept {kept}, rejected {rejected}, precision "
          f"{kept / max(kept + rejected, 1):.2f}")
    print(f"rejected by rule: {dict(rules.rejected)}")
