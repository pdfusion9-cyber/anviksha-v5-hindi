"""A reference-based "did it say the right thing" reward.

Grammar scoring can only judge form. This adds a second axis: does the reply
carry the same content as a known-good answer?

It is measured, not judged:

    chrF-like score   character n-gram precision/recall against the reference
    word F1           overlap of content words
    length ratio      a one-word answer to a paragraph is penalised

That is a proxy for meaning, and an honest one only in a specific sense: it
rewards agreeing with the reference, not understanding it. A model that parrots
the reference scores well; a model that says something true in different words
scores badly. So this is useful for training data filtering and for rejection
sampling against known answers, and it is not a general "is this a good answer"
judge.

For open-ended prompts with no single right answer there is no reference, and
this reward must not be used.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


def _char_ngrams(text: str, n: int = 3) -> Counter:
    text = text.replace(" ", "")
    return Counter(text[i : i + n] for i in range(max(len(text) - n + 1, 0)))


def _f_score(hyp_counts: Counter, ref_counts: Counter) -> float:
    overlap = sum((hyp_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(hyp_counts.values()), 1)
    recall = overlap / max(sum(ref_counts.values()), 1)
    return 2 * precision * recall / (precision + recall)


def _words(text: str) -> list[str]:
    return [w for w in text.replace("।", " ").split() if w]


@dataclass
class MeaningScore:
    total: int          # 0-100
    chrf: float
    word_f1: float
    length_ratio: float
    notes: list[str]

    @property
    def passed(self) -> bool:
        return self.total >= 50


def score_against_reference(reply: str, reference: str) -> MeaningScore:
    reply, reference = reply.strip(), reference.strip()
    notes: list[str] = []
    if not reply or not reference:
        return MeaningScore(0, 0.0, 0.0, 0.0, ["empty reply or reference"])

    chrf = 0.5 * (_f_score(_char_ngrams(reply, 2), _char_ngrams(reference, 2))
                  + _f_score(_char_ngrams(reply, 3), _char_ngrams(reference, 3)))

    reply_words, ref_words = _words(reply), _words(reference)
    word_f1 = _f_score(Counter(reply_words), Counter(ref_words))

    length_ratio = min(len(reply), len(reference)) / max(len(reply), len(reference), 1)
    if length_ratio < 0.4:
        notes.append(f"length ratio {length_ratio:.0%}")

    # chrF carries most of the signal; word overlap guards against character
    # coincidence, and the length ratio stops one-word answers from scoring well
    raw = 0.60 * chrf + 0.25 * word_f1 + 0.15 * length_ratio
    total = int(round(raw * 100))
    return MeaningScore(total, round(chrf, 3), round(word_f1, 3),
                        round(length_ratio, 3), notes)


def combined_reward(reply: str, reference: str | None, grammar_score: int) -> dict:
    """Form and content together. Content is only counted when a reference exists."""
    meaning = score_against_reference(reply, reference) if reference else None
    if meaning is None:
        total = grammar_score
    else:
        total = int(round(0.4 * grammar_score + 0.6 * meaning.total))
    return {
        "total": total,
        "grammar": grammar_score,
        "meaning": meaning.total if meaning else None,
        "chrf": meaning.chrf if meaning else None,
        "word_f1": meaning.word_f1 if meaning else None,
    }
