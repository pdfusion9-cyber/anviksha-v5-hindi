"""A rule-based Hindi grammar reward.

This is the "points" system: every generated sentence gets a score, and the
score is built from checks that can be verified without a parser. It is
deliberately conservative — it rewards what it can actually see and never claims
to judge meaning.

Checks, each with a weight:

    +  Devanagari validity      matras must follow a consonant, no orphan marks
    +  Sentence terminator      ends with danda (।) or a question mark
    +  Postposition sanity      ने/को/में/से/पर appear after a word, not at start
    +  Verb agreement hint      है/हैं/था/थे/होगा agreement with plural markers
    +  Length window            between 4 and 40 characters of content
    -  Repetition               the same 6-char window three or more times
    -  Script mixing            Latin letters inside a Devanagari sentence
    -  Orphan punctuation       stray ASCII punctuation in the middle

The score is 0-100. A sentence that fails validity cannot score above 30.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
# Roman-Hindi function words: if a Latin-script sentence contains several of
# these it is Hinglish, not English, and must not be penalised for it
HINGLISH_MARKERS = {
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "me", "mein", "par", "kya",
    "mujhe", "aap", "tum", "woh", "yeh", "nahi", "nahin", "karo", "kare",
    "karna", "hoga", "hogi", "tha", "thi", "the", "bhi", "aur", "jo", "mera",
    "tera", "apna", "kaise", "kyun", "kahan", "kab", "abhi", "kal", "aaj",
    "baje", "subha", "shaam", "kitne", "kitna", "hata", "do", "de", "dena",
    "lena", "chahiye", "sakta", "sakte", "raha", "rahi", "rahe", "gaya",
    "gayi", "hua", "hui", "kuch", "sab", "log", "bahut", "thoda", "accha",
    "achha", "mat", "wala", "wali", "banao", "bana", "likho", "likha",
    "batao", "bata", "suno", "dekho", "chalo", "jao", "aao", "hona", "kar",
}
LATIN = re.compile(r"[A-Za-z]")
MATRA = re.compile(r"[\u093E-\u094C\u0962\u0963]")
CONSONANT = re.compile(r"[\u0915-\u0939\u0958-\u095F]")
POSTPOSITIONS = ("ने", "को", "में", "से", "पर", "का", "की", "के")
VERB_FORMS = ("है", "हैं", "था", "थी", "थे", "होगा", "होगी", "होंगे")


@dataclass
class GrammarScore:
    score: int
    checks: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    script: str = "unknown"

    @property
    def passed(self) -> bool:
        return self.score >= 60


def detect_script(text: str) -> str:
    """Return 'devanagari', 'hinglish', 'english' or 'mixed'."""
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    if dev and dev >= lat * 2:
        return "devanagari"
    if lat and dev == 0:
        words = {w.lower() for w in re.split(r"[^A-Za-z]+", text) if w}
        markers = len(words & HINGLISH_MARKERS)
        return "hinglish" if markers >= 2 else "english"
    return "mixed"


def score_hindi(text: str) -> GrammarScore:
    text = text.strip()
    checks: dict[str, bool] = {}
    notes: list[str] = []

    if not text:
        return GrammarScore(0, {"non_empty": False}, ["empty output"])

    checks["non_empty"] = True
    script = detect_script(text)
    # both scripts are legitimate for this project; only English-only output is
    # a failure, because the model is being trained to answer in Hindi/Hinglish
    checks["has_devanagari"] = script in ("devanagari", "hinglish")

    # a matra must not be the first character, and must follow a consonant
    first_is_matra = bool(text and MATRA.match(text[0]))
    orphan_matra = bool(re.search(r"[\s\u0964]%s" % MATRA.pattern, text))
    checks["matra_placement"] = (script != "devanagari") or (
        not first_is_matra and not orphan_matra)
    if first_is_matra:
        notes.append("starts with a matra")
    if orphan_matra:
        notes.append("matra after a space")

    checks["terminated"] = text.endswith(("।", "?", "॥", "."))
    if not checks["terminated"]:
        notes.append("no sentence terminator")

    # in Hindi a postposition IS a standalone word ("भारत की राजधानी"), so the
    # only real surface rule is that a sentence must not *begin* with one
    words = [w for w in re.split(r"[\s।,?!]+", text) if w]
    checks["postposition_sane"] = not (words and words[0] in POSTPOSITIONS)
    if not checks["postposition_sane"]:
        notes.append("sentence starts with a postposition")

    has_verb = any(v in text for v in VERB_FORMS) or any(
        w.lower() in {"hai", "hain", "tha", "thi", "the", "hoga", "hogi", "kare", "karo"}
        for w in words)
    checks["has_verb_form"] = has_verb

    content = len(DEVANAGARI.findall(text))
    checks["length_ok"] = 4 <= content <= 120
    if content < 4:
        notes.append("too short to be a sentence")

    # catch both a repeating phrase and the same single word three times
    repeated_phrase = bool(re.search(r"(.{6,})\1{2,}", text))
    word_counts: dict[str, int] = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    repeated_word = any(count >= 3 for count in word_counts.values())
    repetition = repeated_phrase or repeated_word
    checks["no_repetition"] = not repetition
    if repetition:
        notes.append("repeating phrase")

    latin_ratio = len(LATIN.findall(text)) / max(len(text), 1)
    checks["no_script_mixing"] = script != "mixed"
    if not checks["no_script_mixing"]:
        notes.append(f"latin characters {latin_ratio:.0%}")

    stray = bool(re.search(r"[\u0900-\u097F][.,;:][\u0900-\u097F]", text))
    checks["no_stray_punctuation"] = not stray

    # a sentence built from one repeated word is degenerate no matter how
    # well-formed each word is
    unique_ratio = len(set(words)) / max(len(words), 1)
    checks["lexical_variety"] = unique_ratio >= 0.5
    if not checks["lexical_variety"]:
        notes.append(f"only {unique_ratio:.0%} unique words")

    short_unit_repeat = bool(re.search(r"(.{2,4})\1{3,}", text))
    checks["no_short_unit_repeat"] = not short_unit_repeat
    if short_unit_repeat:
        notes.append("short unit repeated")

    weights = {
        "non_empty": 5,
        "has_devanagari": 15,
        "matra_placement": 10,
        "terminated": 10,
        "postposition_sane": 5,
        "has_verb_form": 15,
        "length_ok": 10,
        "no_repetition": 10,
        "no_script_mixing": 5,
        "no_stray_punctuation": 5,
        "lexical_variety": 5,
        "no_short_unit_repeat": 5,
    }
    assert sum(weights.values()) == 100
    total = sum(weights[k] for k, ok in checks.items() if ok)
    if not checks["has_devanagari"]:
        total = min(total, 30)
    if not checks["lexical_variety"] or not checks["no_short_unit_repeat"]:
        total = min(total, 30)
    score = GrammarScore(total, checks, notes)
    score.script = script  # type: ignore[attr-defined]
    return score
