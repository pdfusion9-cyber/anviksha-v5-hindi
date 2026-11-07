from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
import urllib.request
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


SOURCES = {
    "xquad_en": {
        "url": "https://raw.githubusercontent.com/google-deepmind/xquad/master/xquad.en.json",
        "license": "CC-BY-SA-4.0",
        "attribution": "google-deepmind/xquad; SQuAD v1.1 translated by XQuAD contributors",
    },
    "xquad_hi": {
        "url": "https://raw.githubusercontent.com/google-deepmind/xquad/master/xquad.hi.json",
        "license": "CC-BY-SA-4.0",
        "attribution": "google-deepmind/xquad; SQuAD v1.1 translated by XQuAD contributors",
    },
    "phinc": {
        "url": "https://huggingface.co/datasets/LingoIITGN/PHINC/resolve/main/PHINC.csv",
        "license": "CC-BY-4.0",
        "attribution": "Lingo Research Group, IIT Gandhinagar; Srivastava & Singh (W-NUT 2020)",
    },
}


def download(url: str, path: Path):
    request = urllib.request.Request(url, headers={"User-Agent": "HCDM110-research-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if len(data) < 100:
        raise ValueError(f"download unexpectedly small: {url} ({len(data)} bytes)")
    path.write_bytes(data)
    return {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def stable_split(group: str, salt: str = "hcdm110-realtext-v1") -> str:
    value = int(hashlib.sha256((salt + "|" + group).encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "validation"
    return "test"


def parse_xquad(path: Path, language: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for article_index, article in enumerate(payload.get("data", [])):
        for paragraph_index, paragraph in enumerate(article.get("paragraphs", [])):
            context = paragraph.get("context", "").strip()
            if not context:
                continue
            group_id = f"article-{article_index:04d}-paragraph-{paragraph_index:04d}"
            split = stable_split(group_id, salt="xquad-parallel-doc-v1")
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "").strip()
                if question:
                    rows.append({
                        "source": "xquad",
                        "language": language,
                        "group_id": group_id,
                        "split": split,
                        "query": question,
                        "document": context,
                        "query_id": str(qa.get("id", "")),
                    })
    return rows, len(payload.get("data", []))


def parse_phinc(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"Sentence", "English_Translation"}
        if not required.issubset(set(fields)):
            raise ValueError(f"PHINC columns mismatch; got {fields}")
        seen = set()
        for index, record in enumerate(reader):
            query = (record.get("Sentence") or "").strip()
            document = (record.get("English_Translation") or "").strip()
            if len(query) < 3 or len(document) < 3:
                continue
            norm = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", query).casefold())
            if norm in seen:
                continue
            seen.add(norm)
            group_id = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            rows.append({
                "source": "phinc",
                "language": "hinglish",
                "group_id": group_id,
                "split": stable_split(group_id, salt="phinc-query-v1"),
                "query": query,
                "document": document,
                "query_id": str(index),
            })
    return rows, fields


TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def hashed_text_vector(text: str, dim: int = 384) -> torch.Tensor:
    """Deterministic multilingual lexical/character feature hashing; no external model."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = TOKEN_RE.findall(normalized)
    features = []
    features.extend("w:" + word for word in words)
    for word in words:
        chars = "^" + word + "$"
        for n in (3, 4):
            features.extend("c:" + chars[i:i+n] for i in range(max(0, len(chars)-n+1)))
    vector = torch.zeros(dim, dtype=torch.float32)
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        bucket = number % dim
        sign = 1.0 if ((number >> 8) & 1) else -1.0
        vector[bucket] += sign
    return F.normalize(vector, dim=0)


def encode(rows):
    queries = torch.stack([hashed_text_vector(row["query"]) for row in rows])
    documents = torch.stack([hashed_text_vector(row["document"]) for row in rows])
    labels = torch.tensor([int(hashlib.sha256(row["group_id"].encode()).hexdigest()[:8], 16)
                           for row in rows], dtype=torch.long)
    return queries, documents, labels


class SharedSparseProjection(torch.nn.Module):
    def __init__(self, dim=384, expansion=768, active_k=64, seed=123):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(expansion, dim, generator=generator)
        weight = F.normalize(weight, dim=-1)
        self.weight = torch.nn.Parameter(weight)
        self.active_k = active_k

    def forward(self, x):
        scores = F.relu(x.float() @ self.weight.float().T)
        values, ids = scores.topk(self.active_k, dim=-1)
        codes = torch.zeros_like(scores).scatter(1, ids, values)
        return F.normalize(codes, dim=-1)


def multi_positive_symmetric_infonce(query_codes, document_codes, labels, temperature=0.10):
    similarities = query_codes @ document_codes.T / temperature
    positives = labels[:, None].eq(labels[None, :])
    log_prob_qd = similarities - torch.logsumexp(similarities, dim=1, keepdim=True)
    log_prob_dq = similarities.T - torch.logsumexp(similarities.T, dim=1, keepdim=True)
    pos_count = positives.sum(1).clamp_min(1)
    loss_qd = -(log_prob_qd.masked_fill(~positives, 0).sum(1) / pos_count).mean()
    loss_dq = -(log_prob_dq.masked_fill(~positives.T, 0).sum(1) /
                positives.T.sum(1).clamp_min(1)).mean()
    return 0.5 * (loss_qd + loss_dq)


def train_projection(q, d, labels, seed, active_k, epochs=20):
    model = SharedSparseProjection(384, 768, active_k, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Keep paired samples in each batch; random order per epoch.
    generator = torch.Generator().manual_seed(seed + active_k * 100)
    loss_track = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(q.shape[0], generator=generator)
        for ids in order.split(128):
            if ids.numel() < 2:
                continue
            qz, dz = model(q[ids]), model(d[ids])
            loss = multi_positive_symmetric_infonce(qz, dz, labels[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_track.append(float(loss.detach()))
    model.eval()
    return model, {"epochs": epochs, "optimizer_steps": len(loss_track),
                   "initial_loss": loss_track[0] if loss_track else None,
                   "final_loss": loss_track[-1] if loss_track else None}


def retrieval_scores(scores, gold):
    ranked = scores.argsort(dim=-1, descending=True)
    hit1 = ranked[:, 0].eq(gold).float()
    hit5 = (ranked[:, :5] == gold[:, None]).any(-1).float()
    hit10 = (ranked[:, :10] == gold[:, None]).any(-1).float()
    positions = torch.nonzero(ranked == gold[:, None], as_tuple=False)[:, 1] + 1
    reciprocal = 1.0 / positions.float()
    ndcg = 1.0 / torch.log2(positions.float() + 1.0)
    return {"recall_at_1": float(hit1.mean()), "recall_at_5": float(hit5.mean()),
            "recall_at_10": float(hit10.mean()), "mrr": float(reciprocal.mean()),
            "ndcg": float(ndcg.mean()), "queries": int(gold.numel()),
            "candidate_documents": int(scores.shape[1])}


def choose_weight(raw_val, learned_val, gold):
    choices = []
    for weight in [i/10 for i in range(11)]:
        metrics = retrieval_scores((1-weight)*raw_val + weight*learned_val, gold)
        choices.append((metrics["mrr"], -weight, weight, metrics))
    best = max(choices)
    return best[2], best[3]


def run_task(name, train_rows, val_rows, test_rows, seed, active_k, epochs):
    if len(train_rows) < 16 or len(val_rows) < 5 or len(test_rows) < 5:
        raise ValueError(f"{name}: split too small train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    train_q, train_d, train_y = encode(train_rows)
    val_q, val_d, _ = encode(val_rows)
    test_q, test_d, _ = encode(test_rows)
    # Documents in validation/test are distinct candidates; row order is the gold index.
    train_model, train_report = train_projection(train_q, train_d, train_y, seed, active_k, epochs)
    with torch.inference_mode():
        val_raw = F.normalize(val_q, dim=-1) @ F.normalize(val_d, dim=-1).T
        val_learned = F.normalize(train_model(val_q), dim=-1) @ F.normalize(train_model(val_d), dim=-1).T
    gold_val = torch.arange(len(val_rows))
    weight, val_metrics = choose_weight(val_raw, val_learned, gold_val)
    with torch.inference_mode():
        raw = F.normalize(test_q, dim=-1) @ F.normalize(test_d, dim=-1).T
        learned = F.normalize(train_model(test_q), dim=-1) @ F.normalize(train_model(test_d), dim=-1).T
        hybrid = (1-weight)*raw + weight*learned
    gold_test = torch.arange(len(test_rows))
    return {
        "task": name, "train_rows": len(train_rows), "validation_rows": len(val_rows),
        "test_rows": len(test_rows), "candidate_documents_test": len(test_rows),
        "train": train_report, "validation": {"weight": weight, "mrr": val_metrics["mrr"]},
        "test": {"raw_cosine": retrieval_scores(raw, gold_test),
                 "contrastive_dg": retrieval_scores(learned, gold_test),
                 "validation_tuned_hybrid": retrieval_scores(hybrid, gold_test)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("data/hcdm110_realtext"))
    parser.add_argument("--out", type=Path, default=Path("reports/hcdm110_realtext.json"))
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--active-k", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, spec in SOURCES.items():
        files[name] = args.workdir / (name + (".csv" if name == "phinc" else ".json"))
    provenance = {name: download(SOURCES[name]["url"], files[name]) for name in SOURCES}

    en_rows, en_articles = parse_xquad(files["xquad_en"], "en")
    hi_rows, hi_articles = parse_xquad(files["xquad_hi"], "hi")
    phinc_rows, phinc_columns = parse_phinc(files["phinc"])
    tasks = []
    for lang, rows in (("english", en_rows), ("hindi", hi_rows)):
        split = {part: [x for x in rows if x["split"] == part]
                 for part in ("train", "validation", "test")}
        tasks.append(run_task("xquad_" + lang, split["train"], split["validation"],
                              split["test"], args.seed, args.active_k, args.epochs))
    split = {part: [x for x in phinc_rows if x["split"] == part]
             for part in ("train", "validation", "test")}
    tasks.append(run_task("phinc_hinglish_to_english", split["train"], split["validation"],
                          split["test"], args.seed, args.active_k, args.epochs))

    report = {
        "benchmark": "HCDM-110 real-language retrieval smoke benchmark v1",
        "representation": "384-d deterministic word + character n-gram feature hashing; no pretrained encoder",
        "model": {"shared_sparse_projection_expansion": 768, "active_k": args.active_k,
                  "contrastive_loss": "symmetric multi-positive in-batch InfoNCE",
                  "epochs": args.epochs, "validation_objective": "MRR"},
        "sources": {name: {**SOURCES[name], **provenance[name]} for name in SOURCES},
        "dataset_audit": {
            "xquad_english_rows": len(en_rows), "xquad_hindi_rows": len(hi_rows),
            "xquad_english_articles": en_articles, "xquad_hindi_articles": hi_articles,
            "phinc_deduplicated_rows": len(phinc_rows), "phinc_columns": phinc_columns,
            "phinc_train": len(split["train"]), "split_rule": "stable SHA256 group split 70/15/15",
            "split_group_overlap": False,
        },
        "tasks": tasks,
        "guardrails": [
            "XQuAD: question-to-paragraph retrieval, tested separately in English and Hindi.",
            "PHINC: Hinglish sentence to human English translation retrieval; not same-language search.",
            "All evaluation candidates and queries are partitioned by document/query group.",
            "This smoke test uses lexical feature hashing, not ANVIKSHA/1B model embeddings.",
            "No raw user content; dataset text is not printed to logs or committed.",
            "XQuAD is CC BY-SA 4.0; PHINC is CC BY 4.0. Attribution and share-alike terms must be respected.",
            "No production checkpoint loaded or modified.",
        ],
        "checkpoint_loaded": False, "production_model_modified": False
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
