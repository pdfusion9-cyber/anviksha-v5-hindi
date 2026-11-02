"""Force every reply through a reasoning stage before it answers.

The requirement is that the model always thinks before it answers, never answers
directly. That is enforced structurally rather than asked for in a prompt: every
training reply is rewritten as

    <think> ...a trace... </think> <ans> ...the answer... </ans>

so the first thing the model can learn to emit after <assistant> is <think>, and
the answer always sits in a place that can be parsed exactly. Generation refuses
to accept a reply with no <ans> block, and the judge scores only the text inside
it -- a trace cannot earn credit for a wrong answer.

What a trace can and cannot be here: this is a 12M-parameter character model, so
the trace is a structured restatement of the question and what kind of answer it
needs -- a plan, not private reasoning. It gives the model somewhere to condition
its answer on the question and the expected shape, which is real. It does not
make the model capable of logic it does not have, and claiming otherwise would be
dishonest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANS_OPEN, ANS_CLOSE = "<ans>", "</ans>"

QUESTION_TYPES = (
    (("क्या है", "कौन है", "क्या हैं", "कितने", "कितनी", "what is", "who is"),
     "यह एक तथ्यात्मक सवाल है। सीधा और छोटा उत्तर देना है।"),
    (("कैसे", "how to", "तरीका"), "यह एक तरीका पूछने वाला सवाल है। क्रम में चरण देने हैं।"),
    (("क्यों", "why"), "यह कारण पूछने वाला सवाल है। कारण सीधे बताना है।"),
    (("translate", "अनुवाद", "hinglish"), "यह अनुवाद का काम है। अर्थ बदले बिना भाषा बदलनी है।"),
    (("पहचानें", "identify", "कौन सा"), "यह वर्गीकरण का सवाल है। हर वस्तु का वर्ग अलग बताना है।"),
)


@dataclass
class Parsed:
    trace: str
    answer: str
    complete: bool


def _question_kind(prompt: str) -> str:
    low = prompt.lower()
    for needles, description in QUESTION_TYPES:
        if any(n in low for n in needles):
            return description
    return "यह एक सामान्य सवाल है। सीधा उत्तर देना है।"


COMPACT_KINDS = {
    "fact": "तथ्यात्मक",
    "how": "क्रम में चरण",
    "why": "कारण",
    "translate": "अनुवाद",
    "classify": "वर्गीकरण",
    "general": "सामान्य",
}


def _kind_key(prompt: str) -> str:
    low = prompt.lower()
    if any(n in low for n in ("क्या है", "कौन है", "कितने", "कितनी", "what is", "who is")):
        return "fact"
    if any(n in low for n in ("कैसे", "how to", "तरीका")):
        return "how"
    if any(n in low for n in ("क्यों", "why")):
        return "why"
    if any(n in low for n in ("translate", "अनुवाद", "hinglish")):
        return "translate"
    if any(n in low for n in ("पहचानें", "identify", "कौन सा")):
        return "classify"
    return "general"


def build_trace_compact(prompt: str, reference: str | None = None) -> str:
    """A short trace: the question type, and the target when one is known.

    Kept to roughly 45 characters so a full reasoning example still fits a
    192-token context. The long trace did not fit one, and a trace that gets
    truncated is worse than a short one -- the model would learn a cut-off plan.
    """
    kind = COMPACT_KINDS[_kind_key(prompt)]
    if reference:
        return f"{kind}। उत्तर: {reference.strip()[:40]}।"
    return f"{kind}। सीधा उत्तर।"


def build_trace(prompt: str, reference: str | None = None) -> str:
    """A short, structured plan derived from the question.

    Built from the question's own shape, so the trace is honest about what is
    being asked. When a reference exists the trace also states the target, which
    is what lets the answer stage copy it rather than invent.
    """
    prompt = prompt.strip().replace("\n", " ")
    kind = _question_kind(prompt)
    parts = [f"सवाल: {prompt[:120]}", kind]
    if reference:
        parts.append(f"सही उत्तर: {reference.strip()[:120]}")
    parts.append("अब सीधा उत्तर दूँ।")
    return " ".join(parts)


def wrap(prompt: str, answer: str, reference: str | None = None,
         compact: bool = False) -> str:
    """The exact string the model is trained to produce for this prompt."""
    trace = build_trace_compact(prompt, reference) if compact else build_trace(prompt, reference)
    return f"{THINK_OPEN}{trace}{THINK_CLOSE}{ANS_OPEN}{answer.strip()}{ANS_CLOSE}"


def parse(reply: str) -> Parsed:
    """Pull the trace and the answer apart. An answer with no <ans> is not an answer."""
    reply = reply or ""
    trace = ""
    if THINK_OPEN in reply:
        start = reply.index(THINK_OPEN) + len(THINK_OPEN)
        end = reply.find(THINK_CLOSE, start)
        trace = reply[start:end if end != -1 else len(reply)].strip()

    if ANS_OPEN in reply:
        start = reply.index(ANS_OPEN) + len(ANS_OPEN)
        end = reply.find(ANS_CLOSE, start)
        answer = reply[start:end if end != -1 else len(reply)].strip()
        return Parsed(trace=trace, answer=answer, complete=end != -1)

    # no answer block: the reply is a trace or nothing, never an answer
    return Parsed(trace=trace, answer="", complete=False)


def answer_only(reply: str) -> str:
    """What the judge and the user should see."""
    return parse(reply).answer


if __name__ == "__main__":
    samples = [
        ("भारत की राजधानी क्या है?", "नई दिल्ली"),
        ("पानी कैसे उबालें?", None),
        ("Translate: Set a timer for 10 minutes", "10 minutes ke liye timer set kare"),
    ]
    for prompt, ref in samples:
        wrapped = wrap(prompt, ref or "उत्तर", ref)
        p = parse(wrapped)
        print(f"prompt : {prompt}")
        print(f"wrapped: {wrapped[:110]}")
        print(f"trace  : {p.trace[:70]}")
        print(f"answer : {p.answer}  complete={p.complete}\n")

    # an answer with no <ans> must yield nothing
    bad = parse("<think>सोचा</think> दुनिया की राजधानी की हवा अद्भुत है।")
    print(f"no <ans> block -> answer {bad.answer!r} complete={bad.complete} "
          f"(must be empty and False)")
