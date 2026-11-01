"""Path reward -- score the route, not just the destination.

The existing rewards answer two questions:

    grammar_reward.py   is the reply well-formed?
    meaning_reward.py   does the reply carry the reference answer's content?

Neither says anything about *how* the reply got there. This module scores the
route: does it stay on the user's question, does it progress instead of
repeating, does it keep the shape the answer needs (steps for a how-to, a
number for a quantity), does it stop cleanly.

That distinction is the whole point. An answer can be right by luck on a bad
path, and wrong on a good path -- and those two cases should not get the same
reward. The combined score below keeps them separate so the two can be tracked
and tuned independently.

Every check is computable from the text alone: no judge model, no step labels,
no API. That also bounds what it can claim. It measures shape and grounding, not
correctness of reasoning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
WORD = re.compile(r"[\w\u0900-\u097F]+")
NUMBER = re.compile(r"[\d\u0966-\u096F]+")

STOPWORDS = {
    "का", "की", "के", "को", "से", "में", "पर", "और", "है", "हैं", "था", "थे",
    "एक", "यह", "वह", "कि", "जो", "भी", "नहीं", "हो", "कर", "लिए", "साथ",
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "for", "with", "that", "this", "it", "as", "at", "by",
}

STEP_MARKERS = ("1", "2", "3", "१", "२", "३", "पहले", "फिर", "अंत में",
                "first", "then", "finally", "next", "step")


def _words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def _content_words(text: str) -> set[str]:
    return {w for w in _words(text) if len(w) > 1 and w not in STOPWORDS}


@dataclass
class PathScore:
    score: float
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def score_path(prompt: str, reply: str) -> PathScore:
    """Score the route from `prompt` to `reply`, 0-100."""
    reply = reply.strip()
    prompt = prompt.strip()
    notes: list[str] = []
    checks: dict[str, bool] = {}

    # An empty reply passes every check vacuously -- no words means no loop, no
    # shape mismatch, no dangling end -- and scored 95/100 when this was first
    # tested. It has to be caught before the checks, not after.
    if len(reply) < 2:
        return PathScore(score=0.0, checks={"non_empty": False},
                         notes=["empty reply"])

    words = _words(reply)
    prompt_words = _content_words(prompt)
    reply_words = _content_words(reply)

    # 1. grounding: a reply that shares nothing with the question has wandered off
    overlap = len(prompt_words & reply_words) / max(len(prompt_words), 1)
    checks["grounded"] = overlap > 0.0 or len(prompt_words) < 3
    if not checks["grounded"]:
        notes.append("shares no content word with the prompt")

    # 2. progress: unique words should keep arriving, not recycle
    if len(words) >= 8:
        first_half = set(words[: len(words) // 2])
        second_half = set(words[len(words) // 2:])
        new_later = len(second_half - first_half) / max(len(second_half), 1)
        checks["progresses"] = new_later > 0.2
        if not checks["progresses"]:
            notes.append("second half adds no new words")
    else:
        checks["progresses"] = True

    # 3. no phrase recycling
    checks["no_loop"] = not re.search(r"(.{5,})\1{2,}", reply)
    if not checks["no_loop"]:
        notes.append("repeats a phrase")

    # 4. shape match: a how-to question needs steps; a quantity question needs a number
    wants_steps = any(m in prompt.lower() for m in ("कैसे", "steps", "तरीका", "how to"))
    has_steps = any(m in reply.lower() for m in STEP_MARKERS)
    checks["shape_matches"] = (not wants_steps) or has_steps
    if wants_steps and not has_steps:
        notes.append("question asked how-to but reply has no steps")

    wants_number = bool(NUMBER.search(prompt))
    has_number = bool(NUMBER.search(reply))
    checks["answers_quantity"] = (not wants_number) or has_number
    if wants_number and not has_number:
        notes.append("question contains a number, reply does not")

    # 5. clean stop: should not end mid-word or on a dangling joiner
    checks["stops_cleanly"] = not reply.endswith(("का", "के", "की", "और", "से", "of", "and", "the"))
    if not checks["stops_cleanly"]:
        notes.append("ends on a dangling word")

    # 6. length plausibility relative to the question
    ratio = len(reply) / max(len(prompt), 1)
    checks["length_plausible"] = 0.2 <= ratio <= 8.0 and len(reply) >= 8
    if not checks["length_plausible"]:
        notes.append(f"reply/question length ratio {ratio:.1f}")

    weights = {
        "grounded": 30,
        "progresses": 20,
        "no_loop": 15,
        "shape_matches": 15,
        "answers_quantity": 10,
        "stops_cleanly": 5,
        "length_plausible": 5,
    }
    assert sum(weights.values()) == 100, sum(weights.values())
    score = sum(w for k, w in weights.items() if checks.get(k, False))

    # Two words pass nearly every check by having nothing to fail. A route that
    # short cannot be judged, so it must not score like a real answer.
    if len(words) < 4:
        notes.append(f"only {len(words)} words -- too short to judge the route")
        score = min(score, 40)

    return PathScore(score=score, checks=checks, notes=notes)


def score_path_and_answer(prompt: str, reply: str, reference: str | None = None,
                    path_weight: float = 0.4) -> dict:
    """Path reward and answer reward, kept separate.

    With no reference there is no answer reward to compute, so the path score
    stands alone and `answer` is reported as None rather than guessed.
    """
    from grammar_reward import score_hindi
    from meaning_reward import score_against_reference

    path = score_path(prompt, reply)
    form = score_hindi(reply)
    answer = score_against_reference(reply, reference) if reference else None

    if answer is None:
        combined = path_weight * path.score + (1 - path_weight) * form.score
    else:
        combined = (path_weight * path.score
                    + (1 - path_weight) * (0.5 * form.score + 0.5 * answer.total))
    return {
        "combined": round(combined, 1),
        "path": round(path.score, 1),
        "form": round(form.score, 1),
        "answer": None if answer is None else round(answer.total, 1),
        "path_notes": path.notes,
        "form_notes": form.notes,
        "answer_notes": [] if answer is None else answer.notes,
    }


if __name__ == "__main__":
    cases = [
        ("पानी कैसे उबालें?", "1. बर्तन में पानी डालें। 2. गैस जलाएँ। 3. उबाल आने तक रखें।",
         "बर्तन में पानी डालकर गरम करें जब तक उबाल न आए।"),
        ("पानी कैसे उबालें?", "पानी पानी पानी पानी पानी पानी पानी पानी पानी", None),
        ("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।", "नई दिल्ली"),
        ("भारत की राजधानी क्या है?", "आज मौसम अच्छा है और बाज़ार खुला है।", "नई दिल्ली"),
    ]
    for prompt, reply, ref in cases:
        r = score_path_and_answer(prompt, reply, ref)
        print(f"{r['combined']:5.1f}  path {r['path']:5.1f}  form {r['form']:5.1f}  "
              f"answer {r['answer']}  | {reply[:44]}")
        if r["path_notes"]:
            print("        path:", "; ".join(r["path_notes"]))
