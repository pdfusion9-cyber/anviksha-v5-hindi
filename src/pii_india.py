"""Indian identifiers, recognised and removed before they go anywhere.

A privacy layer is not a filter that hides things from a reader. It is a set of
decisions about what may leave a process at all. This one knows the identifiers
India actually uses, and it validates them rather than pattern-matching them,
because a pattern that fires on any twelve digits redacts invoice numbers and a
layer that redacts invoice numbers gets switched off.

The difference between this and a regex list is the checksum. Aadhaar carries a
Verhoeff digit; a twelve-digit number that fails it is not an Aadhaar and must not
be treated as one. The same reasoning applies to every identifier here: validate
first, then act.

What it does, in order of how much it matters:

    detect      find the identifier and say what it is
    redact      replace it with a stable token, so the text stays usable
    hash        keep a keyed digest for correlation without keeping the value
    refuse      stop the request entirely, for the identifiers that should never
                be in a prompt at all

Aadhaar is in the last category. It should not be redacted and passed along; a
request containing one has already gone wrong somewhere upstream, and passing a
tokenised version of it onward teaches the system that this is normal.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field


# ------------------------------------------------------------- validation --
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(number: str) -> bool:
    """The check digit that makes an Aadhaar an Aadhaar.

    Without this, any twelve-digit number looks like one. With it, a random
    twelve-digit number passes about one time in ten, and an invoice number
    almost never does.
    """
    digits = [int(c) for c in number if c.isdigit()]
    if not digits:
        return False
    c = 0
    for i, item in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][item]]
    return c == 0


def pan_valid(pan: str) -> bool:
    """PAN is five letters, four digits, one letter, and the fourth letter
    encodes the holder type. A well-formed string with a wrong fourth letter is
    not a PAN."""
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan.upper()):
        return False
    return pan[3].upper() in "PCHFATBLJG"


def ifsc_valid(code: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", code.upper()))


def gstin_valid(gstin: str) -> bool:
    g = gstin.upper()
    if not re.fullmatch(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]", g):
        return False
    return 1 <= int(g[:2]) <= 37          # state codes run 01..37


# ---------------------------------------------------------------- finding --
@dataclass
class Finding:
    kind: str
    value_span: tuple[int, int]
    severity: str
    action: str            # redact | hash | refuse
    valid: bool


@dataclass
class Result:
    findings: list[Finding] = field(default_factory=list)
    text: str = ""
    refused: bool = False
    reason: str = ""

    @property
    def clean(self) -> bool:
        return not self.findings


class IndiaPII:
    """Detects, validates and removes Indian personal identifiers."""

    # (name, regex, validator or None, severity, action)
    PATTERNS = [
        ("aadhaar", re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"),
         verhoeff_valid, "critical", "refuse"),
        ("pan", re.compile(r"\b[A-Za-z]{5}[0-9]{4}[A-Za-z]\b"),
         pan_valid, "critical", "refuse"),
        ("ifsc", re.compile(r"\b[A-Za-z]{4}0[A-Za-z0-9]{6}\b"),
         ifsc_valid, "medium", "redact"),
        ("gstin", re.compile(r"\b\d{2}[A-Za-z]{5}\d{4}[A-Za-z]\d[A-Za-z]\d[A-Za-z]\b"),
         gstin_valid, "high", "redact"),
        ("upi", re.compile(r"\b[\w.\-]{2,}@(?:ok(?:hdfc|icici|sbi|axis)|paytm|"
                           r"ybl|ibl|axl|apl|upi|bhim|jio|airtel|fbl|"
                           r"oksbi|okaxis|okicici|okhdfc)\b"), None, "high", "redact"),
        ("phone", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
         None, "medium", "redact"),
        ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), None, "medium", "redact"),
        ("voter", re.compile(r"\b[A-Za-z]{3}\d{7}\b"), None, "high", "redact"),
        ("driving_licence",
         re.compile(r"\b[A-Za-z]{2}[- ]?\d{2}[- ]?\d{4}[- ]?\d{7}\b"),
         None, "high", "redact"),
        ("passport", re.compile(r"\b[A-Za-z][0-9]{7}\b"), None, "high", "redact"),
        ("vehicle", re.compile(r"\b[A-Za-z]{2}[- ]?\d{1,2}[- ]?[A-Za-z]{1,3}[- ]?\d{4}\b"),
         None, "low", "redact"),
        ("pincode", re.compile(r"\b[1-9]\d{5}\b"), None, "low", "keep"),
        ("card", re.compile(r"\b(?:\d[ -]*?){13,19}\b"), None, "critical", "refuse"),
    ]

    # Things that look like an identifier but are not personal data. Keeping
    # these out of the redactor is what stops it eating version numbers, ports
    # and dates.
    NOT_PII = re.compile(
        r"\b(19|20)\d{2}[-/]\d{2}[-/]\d{2}\b"          # dates
        r"|\bv?\d+\.\d+\.\d+\b"                         # versions
        r"|\b\d{1,5}\b(?=\s*(?:port|ms|px|kb|mb|gb|%))"  # quantities
        r"|\b0x[0-9a-f]+\b"                             # hex
        r"|\b\d{4}-\d{2}-\d{2}T[\d:]+Z?\b",             # timestamps
        re.I)

    def __init__(self, key: bytes | None = None):
        # the key makes the token stable within a deployment and useless outside
        # it: the same phone number always becomes the same token here, and a
        # different token anywhere else
        self.key = key or b"kavach-pii-v1"

    def _token(self, kind: str, value: str) -> str:
        d = hmac.new(self.key, f"{kind}:{value}".encode(), hashlib.sha256)
        return f"<{kind}:{d.hexdigest()[:10]}>"

    def inspect(self, text: str, redact: bool = True) -> Result:
        result = Result(text=text)
        masked = text

        for name, pattern, validator, severity, action in self.PATTERNS:
            for m in pattern.finditer(text):
                value = m.group(0)
                if self.NOT_PII.fullmatch(value.strip()):
                    continue
                if validator is not None and not validator(value):
                    continue
                # a pincode or a card needs a second look before it is believed
                if name == "card":
                    digits = re.sub(r"\D", "", value)
                    if not (13 <= len(digits) <= 19):
                        continue
                    if not self._luhn(digits):
                        continue
                result.findings.append(Finding(name, m.span(), severity,
                                               action, True))
                if action == "refuse":
                    result.refused = True
                    result.reason = (
                        f"{name} present: this identifier must not enter a model "
                        f"context at all, so the request is refused rather than "
                        f"tokenised")
                elif action == "redact" and redact:
                    masked = masked.replace(value, self._token(name, value))

        result.text = masked
        return result

    @staticmethod
    def _luhn(digits: str) -> bool:
        total, alt = 0, False
        for c in reversed(digits):
            d = int(c)
            if alt:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
            alt = not alt
        return total % 10 == 0

    def summary(self, result: Result) -> dict:
        from collections import Counter
        return {"kinds": dict(Counter(f.kind for f in result.findings)),
                "refused": result.refused,
                "severity": max((f.severity for f in result.findings),
                                default="none")}


if __name__ == "__main__":
    pii = IndiaPII()
    samples = [
        ("valid Aadhaar", "My aadhaar is 2345 6789 0124 please verify"),
        ("invalid Aadhaar", "Invoice 2345 6789 0125 is due"),
        ("PAN", "PAN ABCPD1234E for the account"),
        ("invalid PAN", "Code ABCDX1234E is a product key"),
        ("phone", "call me on +91 98765 43210"),
        ("UPI", "send to ravi.kumar@okhdfc"),
        ("IFSC", "transfer to HDFC0001234"),
        ("email", "write to priya@example.in"),
        ("card", "card 4111 1111 1111 1111 expires soon"),
        ("not PII", "version 2.14.3 released on 2024-05-11, port 8080, 12.5%"),
    ]
    for name, text in samples:
        r = pii.inspect(text)
        s = pii.summary(r)
        print(f"{name:<16} {str(s['kinds']):<28} refused={s['refused']}")
        if r.findings and not r.refused:
            print(f"                 -> {r.text[:70]}")
