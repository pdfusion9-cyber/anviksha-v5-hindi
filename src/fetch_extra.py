"""Add large Hindi instruction sources so the corpus can reach 100k+ pairs.

The existing seven sources yield about 18,700 usable pairs, which is the number
that has been limiting the model. Three public sets add far more:

    atharvanighot/Hindi-Instruct-500K      question/answer, six parquet shards
    guneetsk99/hindi_instruction_set_187K  instruction/output, one csv
    FreedomIntelligence/evol-instruct-hindi  conversations list, one json

Each is read with the same window and quality filter as everything else, so
adding a source cannot lower the bar -- it either produces pairs that fit or it
produces none.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_multi import CACHE, download, from_columns, from_messages  # noqa: E402

HF = "https://huggingface.co/datasets"


def from_question_answer(path: Path, source: str, max_prompt: int, max_reply: int,
                         cap: int) -> list[dict]:
    pairs, seen = [], set()
    handle = pq.ParquetFile(path)
    for batch in handle.iter_batches(batch_size=256,
                                     columns=["question", "answer"]):
        for row in batch.to_pylist():
            user = str(row.get("question", "") or "").strip()
            reply = str(row.get("answer", "") or "").strip()
            if not user or not reply or len(user) > max_prompt or len(reply) > max_reply:
                continue
            if len(reply) < 12:
                continue
            key = user[:100]
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"user": user, "assistant": reply, "source": source})
            if len(pairs) >= cap:
                return pairs
    return pairs


def from_csv_instruction(path: Path, source: str, max_prompt: int, max_reply: int,
                         cap: int) -> list[dict]:
    pairs, seen = [], set()
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            user = (row.get("instruction") or "").strip()
            extra = (row.get("input") or "").strip()
            if extra:
                user = f"{user}\n{extra}"
            reply = (row.get("output") or "").strip()
            if not user or not reply or len(user) > max_prompt or len(reply) > max_reply:
                continue
            if len(reply) < 12:
                continue
            key = user[:100]
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"user": user, "assistant": reply, "source": source})
            if len(pairs) >= cap:
                return pairs
    return pairs


def from_conversations(path: Path, source: str, max_prompt: int, max_reply: int,
                       cap: int) -> list[dict]:
    pairs, seen = [], set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data", data.get("rows", []))
    for row in data:
        turns = row.get("conversations") or []
        human = next((t.get("value", "") for t in turns if t.get("from") == "human"), "")
        bot = next((t.get("value", "") for t in turns if t.get("from") == "gpt"), "")
        user, reply = human.strip(), bot.strip()
        if not user or not reply or len(user) > max_prompt or len(reply) > max_reply:
            continue
        if len(reply) < 12:
            continue
        key = user[:100]
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"user": user, "assistant": reply, "source": source})
        if len(pairs) >= cap:
            return pairs
    return pairs


NEW_SOURCES = {
    "hindi500k": (from_question_answer,
                  f"{HF}/atharvanighot/Hindi-Instruct-500K/resolve/main/"
                  "data/train-00000-of-00006.parquet", "hindi500k_0.parquet"),
    "hindi500k_b": (from_question_answer,
                    f"{HF}/atharvanighot/Hindi-Instruct-500K/resolve/main/"
                    "data/train-00001-of-00006.parquet", "hindi500k_1.parquet"),
    "hindi187k": (from_csv_instruction,
                  f"{HF}/guneetsk99/hindi_instruction_set_187K/resolve/main/"
                  "hindi_Gemma_training187K.csv", "hindi187k.csv"),
    "evol_hindi": (from_conversations,
                   f"{HF}/FreedomIntelligence/evol-instruct-hindi/resolve/main/"
                   "evol-instruct-hindi.json", "evol_hindi.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-prompt", default=140, type=int)
    parser.add_argument("--max-reply", default=260, type=int)
    parser.add_argument("--cap", default=40000, type=int)
    parser.add_argument("--only", default="", help="comma separated subset of sources")
    args = parser.parse_args()

    wanted = [s.strip() for s in args.only.split(",") if s.strip()] or list(NEW_SOURCES)
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        if name not in NEW_SOURCES:
            print(f"{name}: unknown source, skipping")
            continue
        extractor, url, filename = NEW_SOURCES[name]
        path = download(url, filename)
        if path is None or not Path(path).exists():
            print(f"{name}: download failed")
            continue
        try:
            pairs = extractor(Path(path), name, args.max_prompt, args.max_reply, args.cap)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: skipped ({type(exc).__name__}: {str(exc)[:80]})")
            continue
        print(f"{name}: {len(pairs)} usable pairs")
        with (CACHE / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
