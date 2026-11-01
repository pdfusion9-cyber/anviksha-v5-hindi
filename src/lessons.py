"""A permanent record of the model's own mistakes.

Three things happen to a generated reply:

    the best one        is kept for training and shown to the user
    the others          are not thrown away -- the best fragment inside each is
                        pulled out and filed for deeper analysis
    every flaw          is written down once, with a count, and kept on disk

The point of the third part is that a mistake should only be made once. The
store is a plain JSONL file, so it survives restarts, can be committed, and can
be read back into the prompt as a constraint the next time the model generates.

What this cannot do: it does not change the weights by itself. It is a memory
that is consulted at generation time and fed to training, not a learning rule.
Rewriting weights from it is what rft.py does, using the survivors it keeps.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _signature(note: str) -> str:
    """A stable id for a flaw, so the same mistake is counted, not duplicated."""
    return hashlib.sha1(note.strip().lower().encode("utf-8")).hexdigest()[:12]


@dataclass
class Lesson:
    signature: str
    flaw: str
    count: int = 1
    first_seen: str = ""
    last_seen: str = ""
    examples: list[dict] = field(default_factory=list)
    fragments: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class LessonStore:
    """Append-mostly store of learned flaws, deduplicated by signature."""

    def __init__(self, path: Path, max_examples: int = 5):
        self.path = path
        self.max_examples = max_examples
        self.lessons: dict[str, Lesson] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self.lessons[row["signature"]] = Lesson(**row)

    def record(self, flaw: str, prompt: str, reply: str,
               fragment: str | None = None) -> Lesson:
        sig = _signature(flaw)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        lesson = self.lessons.get(sig)
        if lesson is None:
            # created at 0 so the increment below makes the first sighting count 1
            lesson = Lesson(signature=sig, flaw=flaw, count=0, first_seen=now)
            self.lessons[sig] = lesson
        lesson.count += 1
        lesson.last_seen = now
        if len(lesson.examples) < self.max_examples:
            lesson.examples.append({"prompt": prompt[:200], "reply": reply[:200]})
        if fragment and len(lesson.fragments) < self.max_examples:
            lesson.fragments.append(fragment[:200])
        return lesson

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(self.lessons.values(), key=lambda l: -l.count)
        self.path.write_text("\n".join(r.to_json() for r in rows) + "\n",
                             encoding="utf-8")

    def top(self, k: int = 5) -> list[Lesson]:
        return sorted(self.lessons.values(), key=lambda l: -l.count)[:k]

    def constraint(self, k: int = 5) -> str:
        """Text to put in front of the model so it avoids the known mistakes."""
        top = self.top(k)
        if not top:
            return ""
        parts = [f"- {l.flaw} ({l.count} बार)" for l in top]
        return ("इन गलतियों से बचें, ये पहले हो चुकी हैं:\n" + "\n".join(parts))

    def summary(self) -> dict:
        return {
            "distinct_flaws": len(self.lessons),
            "total_occurrences": sum(l.count for l in self.lessons.values()),
            "top_flaws": [{"flaw": l.flaw, "count": l.count} for l in self.top(10)],
        }


def best_fragment(prompt: str, reply: str, min_words: int = 3) -> str | None:
    """The highest-scoring contiguous span of a reply that failed overall.

    A rejected reply is not uniformly bad. Cutting the span that still scores
    well on the path checks keeps whatever the model got right, which is what
    the deeper analysis is supposed to look at.
    """
    from path_reward import score_path

    words = reply.split()
    if len(words) < min_words:
        return None

    best_score, best_span = -1.0, None
    # sliding windows of increasing length; the path score is cheap enough that
    # an exhaustive scan over short replies is fine
    for size in range(min_words, min(len(words), 24) + 1):
        for start in range(0, len(words) - size + 1):
            span = " ".join(words[start:start + size])
            sc = score_path(prompt, span).score
            # prefer longer spans when scores tie
            if sc > best_score or (sc == best_score and best_span is not None
                                   and len(span) > len(best_span)):
                best_score, best_span = sc, span
    return best_span if best_score > 0 else None


if __name__ == "__main__":
    store = LessonStore(Path("/tmp/lessons.jsonl"))
    store.record("shares no content word with the prompt",
                 "भारत की राजधानी क्या है?", "आज मौसम अच्छा है और बाज़ार खुला है।",
                 fragment="आज मौसम अच्छा है")
    store.record("shares no content word with the prompt",
                 "नदी कहाँ बहती है?", "मुझे चाय पसंद है।", fragment="मुझे चाय पसंद है")
    store.record("question asked how-to but reply has no steps",
                 "पानी कैसे उबालें?", "पानी गरम करें।", fragment="पानी गरम करें")
    store.save()
    print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
    print("\nconstraint the model would see:\n" + store.constraint())

    frag = best_fragment("पानी कैसे उबालें?",
                         "पानी पानी गरम करें और फिर बर्तन में डालें और उबाल आने तक रखें")
    print(f"\nbest fragment of a mediocre reply: {frag!r}")
