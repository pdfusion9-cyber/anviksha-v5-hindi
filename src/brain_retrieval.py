"""Find the answer the way recognition memory does: two signals, not one.

A brain does not decide "have I seen this?" with a single number. Recognition
memory is generally modelled with two processes, and the distinction matters
here because it is exactly the failure this system still has:

    familiarity   fast, vague, "this feels like something I know". Cheap and
                  always available. On its own it produces false memories: the
                  feeling is genuine, the content is wrong.

    recollection  slower, specific, "I can retrieve what actually happened".
                  Expensive, and sometimes absent even when familiarity is high.

A confident wrong answer is what you get when familiarity is high and
recollection is absent. That is déjà vu, and it is also what happens when a
stored answer is returned for a question that was never seen: the question looks
familiar, the answer belongs to a different question.

So the fix is not a higher threshold on familiarity. It is to require
recollection before speaking, and to let competing memories suppress each other
the way lateral inhibition does in the hippocampus -- if two candidates are
almost equally familiar, neither has actually been recalled, and the honest
output is silence.

Everything here is computed, not simulated: familiarity is the similarity the
retrieval already produces, and recollection is a set of checks on whether the
answer's content actually belongs to this question.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer_pipeline import KnowledgeBase, Hit  # noqa: E402

NUMBER = re.compile(r"\d+")
WORD = re.compile(r"[\u0900-\u097f]+|[A-Za-z][A-Za-z'-]+")

# words that carry no identity: if the answer shares only these with the
# question, nothing has actually been recalled
HOLLOW = {
    "का", "की", "के", "को", "में", "से", "पर", "और", "है", "हैं", "था", "थी",
    "क्या", "कौन", "कैसे", "कहाँ", "कब", "क्यों", "कि", "यह", "वह", "एक", "हो",
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "what", "which", "who", "how", "when", "where", "why", "do", "does", "did",
}


@dataclass
class Recall:
    """One memory probe, with both signals kept separate."""
    answer: str | None
    familiarity: float
    recollection: float
    margin: float
    decided_by: str
    notes: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.answer is not None

    def as_dict(self) -> dict:
        return {"answer": self.answer,
                "familiarity": round(self.familiarity, 3),
                "recollection": round(self.recollection, 3),
                "margin": round(self.margin, 3),
                "decided_by": self.decided_by, "notes": self.notes}


def distinctive(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text or "")
            if w.lower() not in HOLLOW and len(w) > 1}


def answer_agreement(a: str, b: str) -> float:
    """Do two stored answers say the same thing?

    This is the lateral-inhibition signal. Two memories that agree reinforce each
    other; two that disagree mean the cue matched neither properly, however
    familiar it felt.
    """
    ta, tb = distinctive(a), distinctive(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def recollection_of(question: str, answer: str) -> tuple[float, list[str]]:
    """Is this answer actually supported by this question?

    The checks a person makes before trusting a memory: are the specific details
    carried across, and is the overlap real rather than the grammar every
    sentence in the corpus shares.
    """
    notes: list[str] = []
    score = 1.0

    q_numbers = set(NUMBER.findall(question))
    a_numbers = set(NUMBER.findall(answer))
    missing = q_numbers - a_numbers
    if missing:
        # A specific detail that contradicts the question is decisive, not a
        # deduction. A person asked about 6 o'clock does not accept a memory of
        # 8 o'clock because it feels familiar -- the detail is what settles it.
        # Scoring it as -0.5 let agreement between two competing memories push
        # it back over the line, which is how a false memory survives a check.
        notes.append(f"question says {sorted(missing)}, answer does not -- veto")
        return 0.0, notes

    q_dist = distinctive(question)
    if q_dist:
        coverage = len(q_dist & distinctive(answer)) / len(q_dist)
        if coverage < 0.34:
            score -= 0.4
            notes.append(f"only {coverage:.0%} of the question's specific words "
                         f"appear in the answer")

    # a bare number or single word has no context to check, so it cannot earn
    # recollection on its own
    if len(distinctive(answer)) < 2 and not q_numbers:
        score -= 0.3
        notes.append("answer is too bare to verify")

    return max(0.0, min(1.0, score)), notes


def recall(question: str, kb: KnowledgeBase, *, familiarity_floor: float = 0.55,
           recollection_floor: float = 0.5, margin_floor: float = 0.02,
           k: int = 3) -> Recall:
    """Retrieve the way recognition memory does, and refuse when unsure."""
    hits = kb.search(question, k=k)
    if not hits:
        return Recall(None, 0.0, 0.0, 0.0, "no_match", ["nothing retrieved"])

    top = hits[0]
    familiarity = top.score
    if familiarity < familiarity_floor:
        return Recall(None, familiarity, 0.0, 0.0, "no_match",
                      ["nothing familiar enough"], hits)

    recollection, notes = recollection_of(question, top.answer)

    margin = 0.0
    if len(hits) > 1:
        runner = hits[1]
        margin = familiarity - runner.score
        agreement = answer_agreement(top.answer, runner.answer)
        if agreement >= 0.5:
            recollection = min(1.0, recollection + 0.15)
            notes.append("runner-up agrees, memory reinforced")
        elif margin < margin_floor:
            recollection *= 0.5
            notes.append("runner-up competes with no clear winner, "
                         "so neither was recalled")
        else:
            notes.append(f"runner-up differs but is {margin:.2f} behind")

    if recollection < recollection_floor:
        # high familiarity with weak recollection is precisely a false memory
        kind = "familiar_only" if familiarity >= 0.85 else "no_match"
        return Recall(None, familiarity, recollection, margin, kind,
                      notes + ["familiar but not recollected"], hits)

    return Recall(top.answer, familiarity, recollection, margin, "recollected",
                  notes, hits)


if __name__ == "__main__":
    import json
    from pathlib import Path as P

    kb_path = P(sys.argv[1] if len(sys.argv) > 1 else "data/idx_test.jsonl")
    if not kb_path.is_file():
        print(f"missing {kb_path}")
        raise SystemExit(2)
    kb = KnowledgeBase.load(kb_path)

    probes = [("in the base", kb.rows[0]["user"]),
              ("in the base", kb.rows[min(1, len(kb.rows) - 1)]["user"]),
              ("unseen, reworded", kb.rows[0]["user"].replace("क्या", "कौन")),
              ("unseen, new topic", "मंगल ग्रह पर पानी कहाँ मिला है?")]
    print(f"knowledge base: {len(kb.rows)} rows\n")
    for label, q in probes:
        r = recall(q, kb)
        print(f"[{'ANSWER' if r.answered else 'REFUSE'}] {label}")
        print(f"   Q: {q[:70]}")
        print(f"   familiarity {r.familiarity:.2f}  recollection {r.recollection:.2f}"
              f"  margin {r.margin:.2f}  -> {r.decided_by}")
        if r.answered:
            print(f"   A: {r.answer[:70]}")
        for n in r.notes[:2]:
            print(f"   · {n}")
        print()
