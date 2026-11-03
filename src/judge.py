"""A strict judge: a wrong answer must not pass, whatever else it gets right.

The previous scoring was a weighted average, and that was the flaw. An answer
can be badly wrong and still average above the bar, because good form and a
loose notion of "on topic" contribute enough to carry it:

    "मंगोलिया की राजधानी?" -> "दुनिया की राजधानी की हवा के पीत्रतीय अद्भुत है।"
    answer 3, form 100, path 70  ->  weighted average 79.6  ->  accepted

Averaging lets one strong component hide a fatal one. So this module does not
average. It applies hard gates first, and only if every gate passes does a score
get computed at all. A reply that fails any gate is rejected outright, with the
reason recorded.

The gates, in the order they matter:

    1. it must say something             (not empty, not too short)
    2. it must agree with the reference  (answer score >= ANSWER_BAR)
    3. its numbers must be the reference's numbers
    4. it must not invent proper nouns the reference and question never mention
    5. it must share the reference's content words

Gates 3 and 4 exist because the reward let through "20 minutes -> 250 minutes"
and "अताबक -> गुनवर तार है". Both are wrong in a way a character-overlap score
cannot see, and both would teach the model to invent facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from grammar_reward import score_hindi
from meaning_reward import score_against_reference
from path_reward import score_path
from rule_engine import check as check_rules  # noqa: E402

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
WORD = re.compile(r"[\w\u0900-\u097F]+")
NUMBER = re.compile(r"\d+")

# a reply must reach this answer score to be usable at all
ANSWER_BAR = 65
# of the reference's content words, this share must appear in the reply
GROUNDING_BAR = 0.45
MIN_WORDS = 4

STOPWORDS = {
    "का", "की", "के", "को", "से", "में", "पर", "और", "है", "हैं", "था", "थे",
    "एक", "यह", "वह", "कि", "जो", "भी", "नहीं", "हो", "कर", "लिए", "साथ",
    "हुआ", "हुई", "होता", "किया", "करने", "करें", "दें", "आप", "मैं", "अपने",
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "for", "with", "that", "this", "it", "as", "at", "by", "you",
}


def _words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def _content(text: str) -> set[str]:
    return {w for w in _words(text) if len(w) > 1 and w not in STOPWORDS}


def _numbers(text: str) -> set[str]:
    return set(NUMBER.findall(text))


@dataclass
class Verdict:
    passed: bool
    score: float
    answer: int | None
    path: float
    form: float
    gates: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": round(self.score, 1),
            "answer": self.answer,
            "path": round(self.path, 1),
            "form": round(self.form, 1),
            "gates": self.gates,
            "reasons": self.reasons,
        }


def judge(prompt: str, reply: str, reference: str | None,
          require_reference: bool = True) -> Verdict:
    """Gate first, score second. A failed gate means the reply is not usable.

    Without a reference there is no way to know whether the reply is right, and
    every wrong reply the old reward accepted would pass again -- all 30 of them
    did, scoring 88-100 on form and shape alone. So a missing reference is
    itself a failed gate. Rejecting a good reply is recoverable; training on a
    wrong one is not.
    """
    reply = (reply or "").strip()
    path = score_path(prompt, reply)
    form = score_hindi(reply)
    answer = score_against_reference(reply, reference) if reference else None

    gates: dict[str, bool] = {}
    reasons: list[str] = []

    # 1. says something
    gates["not_empty"] = len(_words(reply)) >= MIN_WORDS
    if not gates["not_empty"]:
        reasons.append(f"fewer than {MIN_WORDS} words")

    # 2. agrees with the reference -- this is the gate that matters.
    # A short reference against a longer reply is penalised by the length ratio,
    # so "नई दिल्ली" vs "भारत की राजधानी नई दिल्ली है।" scores only 44 despite
    # being correct. If the reply contains every content word the reference
    # asserts, it is at least as informative and the gate is satisfied.
    contains_reference = False
    if reference:
        ref_content_all = _content(reference)
        contains_reference = bool(ref_content_all) and ref_content_all <= _content(reply)

    if answer is None:
        gates["matches_reference"] = not require_reference
        reasons.append("no reference: correctness cannot be checked")
    elif contains_reference:
        gates["matches_reference"] = True
    else:
        gates["matches_reference"] = answer.total >= ANSWER_BAR
        if not gates["matches_reference"]:
            reasons.append(f"answer score {answer.total} < {ANSWER_BAR}")

    # 3. numbers must be the reference's numbers
    if reference:
        ref_nums, rep_nums = _numbers(reference), _numbers(reply)
        if ref_nums:
            gates["numbers_match"] = rep_nums == ref_nums or rep_nums <= ref_nums
            if not gates["numbers_match"]:
                reasons.append(f"numbers {sorted(rep_nums)} not in reference "
                               f"{sorted(ref_nums)}")
        else:
            gates["numbers_match"] = True
    else:
        gates["numbers_match"] = True

    # 4. no invented proper nouns
    if reference:
        allowed = _content(reference) | _content(prompt)
        introduced = {w for w in _content(reply) - allowed if len(w) > 2}
        # a few new words are normal phrasing; a crowd of them is invention
        gates["no_invented_names"] = len(introduced) <= max(2, len(allowed) // 3)
        if not gates["no_invented_names"]:
            reasons.append(f"{len(introduced)} words appear in neither question "
                           f"nor reference")
    else:
        gates["no_invented_names"] = True

    # 5. obeys the rule files. This is the gate that tells the model which answer
    # is right and which is wrong: a grammar violation is a fact about the text,
    # not an opinion, so it vetoes rather than deducts.
    rule_report = check_rules(reply)
    gates["rules"] = rule_report.passed
    if not gates["rules"]:
        reasons.append("rules: " + "; ".join(
            v.detail for v in rule_report.errors[:2]))

    # 6. shares the reference's content words
    if reference:
        ref_content = _content(reference)
        if ref_content:
            shared = len(ref_content & _content(reply)) / len(ref_content)
            gates["grounded_in_reference"] = shared >= GROUNDING_BAR
            if not gates["grounded_in_reference"]:
                reasons.append(f"shares only {shared:.0%} of reference content words")
        else:
            gates["grounded_in_reference"] = True
    else:
        gates["grounded_in_reference"] = True

    passed = all(gates.values())
    if passed:
        # answer dominates; form and path are tie-breakers, not saviours
        if answer is None:
            score = 0.75 * path.score + 0.25 * form.score
        else:
            score = 0.70 * answer.total + 0.20 * path.score + 0.10 * form.score
    else:
        # a failed gate caps the score, so it can never look acceptable
        score = min(30.0, 0.70 * (answer.total if answer else 0) + 0.2 * path.score)

    return Verdict(passed=passed, score=score,
                   answer=None if answer is None else answer.total,
                   path=path.score, form=form.score, gates=gates, reasons=reasons)


if __name__ == "__main__":
    cases = [
        # the replies the old reward accepted, all wrong
        ("मंगोलिया की राजधानी क्या है?",
         "दुनिया की राजधानी की हवा के पीत्रतीय अद्भुत है।", "उलानबातर"),
        ("स्पेन की राजधानी क्या है", "एक परिचर्ड की राजधानी है।", "मैड्रिड"),
        ("Snooze alarm for 20 minutes", "250 minutes ke liye snooze kare",
         "20 minutes ke liye snooze kare"),
        ("पहचानें कि कौन सा वाद्य तार या तालवाद्य हैः अताबक, अजेंग",
         "गुनवर तार है, ताक्रा ताल है।", "अताबक तार है, अजेंग ताल है।"),
        # and one that is genuinely right
        ("नमस्ते", "हैलो! आज मैं आपकी कैसे मदद कर सकता हूँ?",
         "हैलो! आज मैं आपकी कैसे मदद कर सकता हूँ?"),
        ("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।", "नई दिल्ली"),
        ("पानी कैसे उबालें?",
         "बर्तन में पानी डालें और गैस पर रखें। उबाल आने तक इंतज़ार करें।",
         "बर्तन में पानी डालकर गैस पर रखें और उबाल आने तक इंतज़ार करें।"),
        # right shape, right words, wrong place -- must still fail
        ("फ्रांस की राजधानी क्या है?", "फ्रांस की राजधानी बर्लिन है।", "पेरिस"),
    ]
    for prompt, reply, ref in cases:
        v = judge(prompt, reply, ref)
        mark = "PASS" if v.passed else "FAIL"
        print(f"{mark}  score {v.score:5.1f}  answer {v.answer:>4}  "
              f"{prompt[:26]} -> {reply[:34]}")
        if v.reasons:
            print(f"        {', '.join(v.reasons)}")
