"""Check text and code against rules held outside the model.

The model does not carry Hindi grammar or programming syntax in its weights and
never will at this size. So the rules live in `rules/*.json`, are loaded once, and
are enforced on every answer. A rule is a fact about the text, not a preference:
either a postposition opens a clause or it does not.

This is the "control from outside" part. It is deterministic, it does not learn,
and it cannot be talked out of a violation by fluent prose.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

RULES_DIR = Path(__file__).resolve().parent.parent / "rules"
DEVANAGARI = re.compile(r"[\u0900-\u097f]")
LATIN = re.compile(r"[A-Za-z]")
MATRA = re.compile(r"[\u093e-\u094d]")
CODE_HINT = re.compile(r"^\s*(def |class |import |from |function |const |let |var |"
                       r"#include|public |private |fn |package )", re.M)


@dataclass
class Violation:
    rule: str
    severity: str
    detail: str

    def as_dict(self) -> dict:
        return {"rule": self.rule, "severity": self.severity, "detail": self.detail}


@dataclass
class RuleReport:
    violations: list[Violation] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"passed": self.passed,
                "errors": [v.as_dict() for v in self.errors],
                "warnings": [v.as_dict() for v in self.warnings]}


def load_rules(name: str) -> dict:
    path = RULES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"rule file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- Hindi ----

def check_hindi(text: str, rules: dict | None = None) -> RuleReport:
    rules = rules or load_rules("hindi_grammar")
    text = (text or "").strip()
    report = RuleReport()
    if not text:
        report.violations.append(Violation("hi.empty", "error", "no text"))
        return report

    sentences = [s.strip() for s in re.split(r"[।?!]\s*", text) if s.strip()]
    words = re.findall(r"[\u0900-\u097f]+", text)

    for rule in rules["rules"]:
        rid, sev = rule["id"], rule["severity"]

        if rid == "hi.ends_with_predicate":
            for s in sentences:
                tail = s.split()[-1] if s.split() else ""
                if not any(tail.endswith(e) for e in rule["endings"]):
                    # an answer that is a bare noun phrase is the common failure
                    report.violations.append(Violation(
                        rid, sev, f"does not end on a predicate: …{s[-28:]}"))
                    break

        elif rid == "hi.no_english_order":
            if re.search(rule["pattern"], text):
                report.violations.append(Violation(
                    rid, sev, "copula placed before its complement, English order"))

        elif rid == "hi.postposition_follows_noun":
            if re.search(rule["pattern"], text):
                report.violations.append(Violation(
                    rid, sev, "a postposition opens a clause"))

        elif rid in ("hi.plural_agreement", "hi.honorific_agreement"):
            hit = None
            for subj in rule["subjects"]:
                for cop in rule["bad_copulas"]:
                    gap = rule.get("max_gap", 40)
                    if re.search(rf"{re.escape(subj)}[^।?!]{{0,{gap}}}"
                                 rf"{re.escape(cop)}(?![\u0900-\u0903\u093a-\u094f])",
                                 text):
                        hit = f"{subj} … {cop}"
                        break
                if hit:
                    break
            if hit:
                report.violations.append(Violation(rid, sev, f"agreement: {hit}"))

        elif rid == "hi.no_double_postposition":
            stripped = text
            for allowed in rule.get("allow", []):
                stripped = stripped.replace(allowed, "")
            if re.search(rule["pattern"], stripped):
                report.violations.append(Violation(
                    rid, sev, "two postpositions with no noun between them"))

        elif rid == "hi.has_matra":
            if words:
                matras = len(MATRA.findall(text))
                letters = len(words) * 3 or 1
                if matras / letters < rule["min_matra_ratio"]:
                    report.violations.append(Violation(
                        rid, sev, "almost no vowel signs: unreadable Devanagari"))

        elif rid == "hi.no_repeated_word":
            seq = text.split()
            run = 1
            for i in range(1, len(seq)):
                run = run + 1 if seq[i] == seq[i - 1] else 1
                if run > rule["max_repeat"]:
                    report.violations.append(Violation(
                        rid, sev, f"repeats '{seq[i]}'"))
                    break

        elif rid == "hi.terminal_stop":
            if not text.endswith(tuple(rule["allow_terminal"])):
                report.violations.append(Violation(
                    rid, sev, "no terminal punctuation"))

        elif rid == "hi.no_latin_sentence":
            latin = len(LATIN.findall(text))
            if latin / max(len(text), 1) > rule["max_latin_ratio"]:
                report.violations.append(Violation(
                    rid, sev, "mostly Latin characters, not a Hindi answer"))

    return report


# ---------------------------------------------------------------- code ----

def detect_language(code: str) -> str:
    if re.search(r"^\s*(def |import |from \w+ import |class \w+:|print\()", code, re.M):
        return "python"
    if re.search(r"\b(function|const|let|=>)\b", code):
        return "javascript"
    if re.search(r"^\s*#include", code, re.M):
        return "c"
    return "unknown"


def _strip_strings(code: str) -> str:
    out, quote, i = [], None, 0
    while i < len(code):
        ch = code[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            out.append(" ")
        else:
            if ch in "\"'":
                quote = ch
                out.append(" ")
            else:
                out.append(ch)
        i += 1
    return "".join(out)


def check_code(code: str, rules: dict | None = None) -> RuleReport:
    rules = rules or load_rules("programming")
    code = code or ""
    report = RuleReport()
    if not code.strip():
        report.violations.append(Violation("code.empty", "error", "no code"))
        return report

    lang = detect_language(code)
    stripped = _strip_strings(code)

    for rule in rules["rules"]:
        rid, sev = rule["id"], rule["severity"]
        langs = rule.get("languages")
        if langs and lang not in langs:
            continue

        if rid == "code.balanced_brackets":
            for opener, closer in rule["pairs"]:
                if stripped.count(opener) != stripped.count(closer):
                    report.violations.append(Violation(
                        rid, sev, f"{opener}{closer} unbalanced "
                                  f"({stripped.count(opener)} vs {stripped.count(closer)})"))

        elif rid == "code.no_mixed_indentation":
            has_tab = any(line.startswith("\t") for line in code.splitlines())
            has_space = any(line.startswith("    ") for line in code.splitlines())
            if has_tab and has_space:
                report.violations.append(Violation(
                    rid, sev, "tabs and spaces both used for indentation"))

        elif rid == "code.python_keywords":
            for bad in rule["forbidden"]:
                if re.search(rf"(?<![A-Za-z_]){re.escape(bad)}", code):
                    report.violations.append(Violation(
                        rid, sev, f"non-Python construct: {bad!r}"))
                    break

        elif rid == "code.python_no_semicolon":
            if re.search(rule["pattern"], code, re.M):
                report.violations.append(Violation(
                    rid, sev, "line ends with a semicolon"))

        elif rid == "code.colon_after_block":
            for line in code.splitlines():
                if re.search(rule["pattern"], line):
                    report.violations.append(Violation(
                        rid, sev, f"block opener without a colon: {line.strip()[:40]}"))
                    break

        elif rid == "code.no_undefined_placeholder":
            for marker in rule["markers"]:
                if marker == "...":
                    if re.search(r"(^|\s)\.\.\.(\s|$)", code):
                        report.violations.append(Violation(
                            rid, sev, "ellipsis left where code should be"))
                        break
                elif marker in code:
                    report.violations.append(Violation(
                        rid, sev, f"placeholder left in code: {marker}"))
                    break

        elif rid == "code.has_entry_point":
            if not any(k in code for k in rule["any_of"]):
                report.violations.append(Violation(
                    rid, sev, "no function, class or entry point"))

        elif rid == "code.no_destructive_default":
            if re.search(rule["pattern"], code):
                report.violations.append(Violation(
                    rid, sev, "destructive shell command without a guard"))

    return report


# ------------------------------------------------------------ dispatch ----

def check(text: str) -> RuleReport:
    """Pick the rule set from the content: code, Hindi, or nothing to check."""
    if CODE_HINT.search(text) or re.search(r"^\s*```", text, re.M):
        return check_code(text)
    if len(DEVANAGARI.findall(text)) > len(text) * 0.2:
        return check_hindi(text)
    return RuleReport()


if __name__ == "__main__":
    samples_hi = [
        "भारत की राजधानी नई दिल्ली है।",
        "वे ठीक है।",
        "आप ठीक है।",
        "का पानी गरम करें।",
        "है यह बहुत अच्छा।",
        "भारत की राजधानी",
        "पानी पानी पानी पानी।",
        "नई दिल्ली है भारत की राजधानी।",
    ]
    print("--- Hindi ---")
    for s in samples_hi:
        r = check_hindi(s)
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark}  {s[:44]:46} {r.errors[0].detail if r.errors else ''}")

    samples_code = [
        "def add(a, b):\n    return a + b\n",
        "def add(a, b)\n    return a + b\n",
        "def add(a, b):\n    return a + b\n}",
        "function add(a, b) { return a + b }\n",
        "def f():\n    # TODO\n    pass\n",
        "rm -rf / --no-preserve-root",
    ]
    print("\n--- code ---")
    for c in samples_code:
        r = check_code(c)
        mark = "PASS" if r.passed else "FAIL"
        first = r.errors[0].detail if r.errors else ""
        print(f"{mark}  {c.splitlines()[0][:40]:42} {first}")
