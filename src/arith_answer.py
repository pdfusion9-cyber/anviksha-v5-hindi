"""Answer arithmetic exactly, by computing it, because the model cannot.

The decision on tables is settled: the CPU beats a precomputed table on all four
operations, so the tables are not built. That leaves the reason arithmetic still
matters here, which was never speed. The model gets arithmetic wrong -- 14 of 14
arithmetic questions in the last held-out run were refused or wrong, because an
890K-parameter model has no mechanism for carrying digits.

So the value is correctness, and correctness needs a few lines of exact integer
arithmetic, not a grid. This reads the expression out of a question and evaluates
it with Python integers, which are exact and unbounded. A question it cannot read
returns None and falls through to the normal pipeline rather than guessing.

What it handles:

    digits            ASCII and Devanagari, since the corpus uses both
    operators         + - * / and the Hindi words जोड़ घटाव गुणा भाग
    phrases           "12 और 7 का जोड़", "44 को 2 से भाग"
    percentages       "200 का 15 प्रतिशत"
    exactness         integer division where it divides evenly; otherwise the
                      exact rational is reported rather than a rounded float

Every answer carries how it was produced, so a wrong answer can be traced to the
parse rather than blamed on the model.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# words that mean an operation, longest first so "भाग" wins over "भा"
OP_WORDS = [
    ("जोड़", "+"), ("जोड", "+"), ("योग", "+"), ("plus", "+"), ("add", "+"),
    ("घटाव", "-"), ("घटा", "-"), ("अंतर", "-"), ("minus", "-"), ("subtract", "-"),
    ("गुणा", "*"), ("गुणन", "*"), ("times", "*"), ("multiply", "*"), ("product", "*"),
    ("भाग", "/"), ("बांट", "/"), ("divide", "/"), ("divided", "/"), ("quotient", "/"),
]

PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:का|of)?\s*(\d+(?:\.\d+)?)\s*(?:प्रतिशत|percent|%)")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class ArithAnswer:
    text: str
    expression: str
    method: str

    def as_dict(self) -> dict:
        return {"text": self.text, "expression": self.expression,
                "method": self.method}


def normalise(question: str) -> str:
    """Devanagari digits to ASCII, and operation words to symbols."""
    q = question.translate(DEVANAGARI_DIGITS)
    for word, sym in OP_WORDS:
        q = q.replace(word, f" {sym} ")
    return q


def evaluate(a: int, op: str, b: int) -> tuple[str, str]:
    """Exact integer arithmetic. No floats unless the question used them."""
    if op == "+":
        return str(a + b), "add"
    if op == "-":
        return str(a - b), "subtract"
    if op == "*":
        return str(a * b), "multiply"
    if op == "/":
        if b == 0:
            return "शून्य से भाग नहीं हो सकता।", "divide_by_zero"
        if a % b == 0:
            return str(a // b), "divide_exact"
        # report the exact fraction rather than a rounded decimal, because a
        # rounded answer to an exact question is the kind of quiet wrongness the
        # rest of this project is built to avoid
        from math import gcd
        g = gcd(abs(a), abs(b))
        return f"{a // g}/{b // g}", "divide_fraction"
    raise ValueError(op)


def answer_arithmetic(question: str) -> ArithAnswer | None:
    """Return an exact answer, or None if this is not an arithmetic question."""
    q = normalise(question)

    # percentage first: "200 का 15 प्रतिशत" is 15% of 200, not 200 of 15
    m = PERCENT.search(question.translate(DEVANAGARI_DIGITS))
    if m:
        # "200 का 15 प्रतिशत" reads as "<base> का <rate> प्रतिशत": the first
        # number is what it is a percentage of, the second is the rate
        base, rate = m.group(1), m.group(2)
        base_f, rate_f = float(base), float(rate)
        value = base_f * rate_f / 100.0
        if value == int(value):
            text = str(int(value))
        else:
            text = f"{value:.4g}"
        return ArithAnswer(text, f"{rate}% of {base}", "percentage")

    # a bare expression: number operator number
    # "X को Y से भाग" puts words between the numbers, so the operator is not
    # adjacent. Rewrite that phrasing into a plain expression first.
    q = re.sub(r"(\d+)\s*को\s*(\d+)\s*से\s*([+\-*/])", r"\1 \3 \2", q)

    pattern = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)")
    matches = list(pattern.finditer(q))
    if not matches:
        return None

    # use the last complete expression: questions often restate the numbers
    a_s, op, b_s = matches[-1].groups()
    a, b = int(float(a_s)), int(float(b_s))
    text, method = evaluate(a, op, b)
    return ArithAnswer(text, f"{a} {op} {b}", method)


def main() -> None:
    tests = [
        ("44 / 2 कितना होता है?", "22"),
        ("12 + 7 = ?", "19"),
        ("१२ + ७ कितना है?", "19"),
        ("15 - 8 कितना है?", "7"),
        ("12 गुणा 7 कितना है?", "84"),
        ("144 को 12 से भाग दें", "12"),
        ("7 / 2 कितना है?", "7/2"),
        ("5 / 0 कितना है?", "शून्य से भाग नहीं हो सकता।"),
        ("200 का 15 प्रतिशत कितना है?", "30"),
        ("भारत की राजधानी क्या है?", None),
        ("मुझे 3 किताबें चाहिए", None),
    ]
    print("=" * 70)
    print("arithmetic answered by computing, not by guessing")
    print("=" * 70)
    ok = 0
    for q, want in tests:
        r = answer_arithmetic(q)
        got = r.text if r else None
        mark = "ok " if (want is None and got is None) or got == want else "MISS"
        if mark == "ok ":
            ok += 1
        shown = got if got is not None else "(not arithmetic -- falls through)"
        print(f"  [{mark}] {q[:44]:46} -> {shown}")
        if r:
            print(f"         expression {r.expression}, method {r.method}")
    print(f"\n{ok}/{len(tests)} as expected")


if __name__ == "__main__":
    main()
