"""Output screening: decency, leakage and degeneracy checks.

This is a post-hoc filter, not a guarantee. It catches the failure modes a tiny
model actually produces — repetition collapse, empty output, stray template
tokens — and a short list of clearly abusive Hindi terms. It is deliberately
small and readable so its limits are obvious.
"""
from __future__ import annotations

import re

# deliberately short: only unambiguous abuse, no politics, no religion, no
# identity terms. A longer list would create false positives on ordinary text.
ABUSIVE = ["मूर्ख", "बेवकूफ", "गधा", "कमीना", "हरामी", "साला", "चूतिया"]
TEMPLATE_LEAK = ["<user>", "<assistant>", "<bos>", "<eos>", "<sep>", "<pad>"]


def screen(text: str) -> dict:
    issues: list[str] = []
    stripped = text.strip()

    if not stripped:
        issues.append("empty")
    if any(token in text for token in TEMPLATE_LEAK):
        issues.append("template_leak")
    if any(word in text for word in ABUSIVE):
        issues.append("abusive_language")

    # repetition collapse: same 8-character window repeated three or more times
    if len(stripped) > 24 and re.search(r"(.{8,})\1{2,}", stripped):
        issues.append("degenerate_repetition")

    # a reply that is mostly punctuation or whitespace carries no content
    alnum = sum(ch.isalnum() for ch in stripped)
    if stripped and alnum / len(stripped) < 0.4:
        issues.append("low_content")

    return {"clean": not issues, "issues": issues, "length": len(stripped)}
