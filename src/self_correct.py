"""Answer with self-correction before the user ever sees anything.

The rewards so far only judge text after it exists. This module is the loop that
uses them at answer time:

    1. generate several candidate replies
    2. score each on path and form (and on agreement with a reference, if one exists)
    3. if the best one clears the threshold, return it
    4. otherwise go back: find where the best candidate went wrong, cut it there,
       and regenerate the remainder with the failure written into the prompt
    5. if nothing clears the threshold after the allowed rounds, say so plainly
       instead of returning a wrong answer

Step 4 is the part that matters. "Going back" is concrete here: the path checks
name the failure, and the failing candidate tells us how far it got before it
broke, so the next round continues from the last good point rather than starting
over and hoping.

This is inference-time search with a verifier. It costs more compute per answer
-- k candidates times the number of rounds -- which is the honest trade: a wrong
answer delivered confidently is worse than a slower right one.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from grammar_reward import score_hindi
from meaning_reward import score_against_reference
from path_reward import score_path

Generator = Callable[[str, int], str]


@dataclass
class Candidate:
    text: str
    combined: float
    path: float
    form: float
    answer: float | None
    notes: list[str] = field(default_factory=list)


@dataclass
class CorrectionResult:
    answer: str | None
    accepted: bool
    rounds: int
    candidates_tried: int
    best: Candidate | None
    history: list[dict] = field(default_factory=list)
    refusal_reason: str | None = None


def _score(prompt: str, reply: str, reference: str | None) -> Candidate:
    path = score_path(prompt, reply)
    form = score_hindi(reply)
    answer = score_against_reference(reply, reference) if reference else None

    if answer is None:
        combined = 0.6 * path.score + 0.4 * form.score
    else:
        combined = 0.35 * path.score + 0.25 * form.score + 0.40 * answer.total

    notes = list(path.notes)
    if answer is not None and answer.total < 40:
        notes.append("answer disagrees with the reference")

    return Candidate(text=reply, combined=combined, path=path.score,
                     form=form.score,
                     answer=None if answer is None else answer.total,
                     notes=notes)


def _longest_grounded_prefix(prompt: str, reply: str) -> str:
    """The part of the reply that still shares content with the question.

    Cutting here is what "going back along the path" means in practice: keep the
    words that were still on topic, drop the tail that wandered off.
    """
    from path_reward import _content_words  # local import: private helper, same package

    prompt_words = _content_words(prompt)
    if not prompt_words:
        return reply

    words = reply.split()
    if len(words) < 4:
        return ""

    seen_hit = False
    for i, w in enumerate(words):
        if _content_words(w) & prompt_words:
            seen_hit = True
        # once we have been on topic, stop at a long run with no overlap
        elif seen_hit and i >= 3:
            tail = words[i:i + 4]
            if not any(_content_words(t) & prompt_words for t in tail):
                return " ".join(words[:i]).strip()
    return reply


def answer_with_self_correction(
    prompt: str,
    generate: Generator,
    reference: str | None = None,
    candidates_per_round: int = 4,
    threshold: float = 65.0,
    max_rounds: int = 3,
    seed: int = 0,
) -> CorrectionResult:
    """Return an answer only if it clears `threshold`; otherwise refuse."""
    history: list[dict] = []
    tried = 0
    best: Candidate | None = None
    constraint = ""

    for round_index in range(max_rounds):
        round_best: Candidate | None = None
        for c in range(candidates_per_round):
            # the constraint from the previous round is folded into the prompt,
            # so the generator sees what went wrong last time
            effective_prompt = prompt if not constraint else f"{prompt}\n{constraint}"
            text = generate(effective_prompt, seed + round_index * 100 + c)
            tried += 1
            cand = _score(prompt, text, reference)
            if round_best is None or cand.combined > round_best.combined:
                round_best = cand
            if cand.combined >= threshold:
                history.append({"round": round_index, "accepted": True,
                                "best_combined": round(cand.combined, 1)})
                return CorrectionResult(answer=cand.text, accepted=True,
                                        rounds=round_index + 1, candidates_tried=tried,
                                        best=cand, history=history)

        assert round_best is not None
        best = round_best if best is None or round_best.combined > best.combined else best
        history.append({"round": round_index, "accepted": False,
                        "best_combined": round(round_best.combined, 1),
                        "notes": round_best.notes})

        # go back: keep the grounded prefix, and tell the next round what broke
        prefix = _longest_grounded_prefix(prompt, round_best.text)
        reasons = "; ".join(round_best.notes) if round_best.notes else "उत्तर सटीक नहीं था"
        constraint = (f"पिछली कोशिश में यह गलती हुई: {reasons}. "
                      f"इसे सुधारें और सीधे उत्तर दें।")
        if prefix and len(prefix.split()) >= 3:
            constraint = f"शुरुआत इस प्रकार ठीक थी: {prefix}. " + constraint

    return CorrectionResult(
        answer=None, accepted=False, rounds=max_rounds, candidates_tried=tried,
        best=best, history=history,
        refusal_reason=(f"सबसे अच्छा स्कोर {best.combined:.0f} था, जो "
                        f"{threshold:.0f} की सीमा से कम है" if best else "कोई उत्तर नहीं बना"))


if __name__ == "__main__":
    # a stub generator so the loop can be exercised without a checkpoint
    good = "भारत की राजधानी नई दिल्ली है।"
    bad = "आज मौसम अच्छा है और बाज़ार खुला है।"
    junk = "का का कीटेका क्यम ना के"

    def stub(prompt: str, seed: int) -> str:
        pool = [bad, junk, good, bad]
        return pool[seed % len(pool)]

    r = answer_with_self_correction(
        "भारत की राजधानी क्या है?", stub, reference="नई दिल्ली",
        candidates_per_round=4, threshold=60, max_rounds=2)
    print(f"accepted={r.accepted} rounds={r.rounds} tried={r.candidates_tried}")
    print(f"answer={r.answer!r}")
    print(f"best combined={r.best.combined:.1f} path={r.best.path:.1f} "
          f"form={r.best.form:.1f} answer={r.best.answer}")

    # a generator that can never satisfy the threshold must refuse, not guess
    def always_bad(prompt: str, seed: int) -> str:
        return junk

    r2 = answer_with_self_correction(
        "भारत की राजधानी क्या है?", always_bad, reference="नई दिल्ली",
        threshold=80, max_rounds=2)
    print(f"\naccepted={r2.accepted} refusal={r2.refusal_reason!r}")
    print(f"answer={r2.answer!r}  (must be None)")
