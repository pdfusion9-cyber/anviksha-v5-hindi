from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
import unicodedata
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi


MIRACL_REPO = "miracl/miracl"
PHINC_URL = "https://huggingface.co/datasets/LingoIITGN/PHINC/resolve/main/PHINC.csv"
TEXT_DIM = 384
EXPANSION_DIM = 768
TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def stable_bucket(value: str, salt: str) -> int:
    digest = hashlib.sha256((salt + "|" + value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 100


def split_train_validation(qid: str, language: str) -> str:
    return "fit" if stable_bucket(qid, "miracl-" + language) < 80 else "validation"


def feature_vector(text: str, dim: int = TEXT_DIM) -> torch.Tensor:
    """Deterministic Unicode word + character n-gram hashing baseline."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = TOKEN_RE.findall(normalized)
    features = ["w:" + word for word in words]
    for word in words:
        bounded = "^" + word + "$"
        for n in (3, 4):
            features.extend("c:" + bounded[i:i+n]
                            for i in range(max(0, len(bounded) - n + 1)))
    vector = torch.zeros(dim, dtype=torch.float32)
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        bucket = number % dim
        vector[bucket] += 1.0 if ((number >> 8) & 1) else -1.0
    return F.normalize(vector, dim=0)


def encode_texts(texts):
    if not texts:
        raise ValueError("cannot encode empty text list")
    return torch.stack([feature_vector(text) for text in texts])


def normalize_passages(values):
    output = []
    seen = set()
    for item in values or []:
        docid = str(item.get("docid", "")).strip()
        text = str(item.get("text", "") or "").strip()
        title = str(item.get("title", "") or "").strip()
        if not docid or not text or docid in seen:
            continue
        seen.add(docid)
        output.append({"docid": docid, "text": (title + " " + text).strip()})
    return output


def normalize_miracl_row(record):
    qid = str(record.get("query_id", record.get("qid", ""))).strip()
    query = str(record.get("query", "") or "").strip()
    positive = normalize_passages(record.get("positive_passages"))
    negative = normalize_passages(record.get("negative_passages"))
    if not qid or not query or not positive:
        raise ValueError("MIRACL row missing query id, query, or positive passage")
    return {"qid": qid, "query": query, "positive": positive, "negative": negative}


def load_miracl(language: str, revision: str):
    dataset = load_dataset(
        MIRACL_REPO, language, revision=revision, trust_remote_code=True
    )
    train_rows = [normalize_miracl_row(row) for row in dataset["train"]]
    dev_rows = [normalize_miracl_row(row) for row in dataset["dev"]]
    return train_rows, dev_rows


def candidate_pool(rows):
    docs = {}
    qrels = {}
    for row in rows:
        positives = row["positive"]
        negatives = row["negative"]
        qrels[row["qid"]] = {item["docid"]: 1 for item in positives}
        for item in positives + negatives:
            docs.setdefault(item["docid"], item["text"])
    if not rows or not docs:
        raise ValueError("empty MIRACL evaluation split/candidate pool")
    doc_ids = sorted(docs)
    return {
        "query_ids": [row["qid"] for row in rows],
        "query_texts": [row["query"] for row in rows],
        "doc_ids": doc_ids,
        "doc_texts": [docs[key] for key in doc_ids],
        "qrels": qrels,
    }


def training_pairs(rows):
    pairs = []
    for row in rows:
        # A single positive per query avoids overweighting queries with many labels.
        positive = row["positive"][0]
        pairs.append({"qid": row["qid"], "query": row["query"],
                      "docid": positive["docid"], "document": positive["text"]})
    return pairs


def download_phinc(path: Path):
    request = urllib.request.Request(
        PHINC_URL, headers={"User-Agent": "HCDM110-licensed-retrieval-benchmark/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    if len(content) < 1000:
        raise ValueError("PHINC download unexpectedly small")
    path.write_bytes(content)
    return {"url": PHINC_URL, "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def load_phinc(path: Path):
    rows = []
    seen_queries = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not {"Sentence", "English_Translation"}.issubset(set(fields)):
            raise ValueError("PHINC CSV schema mismatch")
        for index, record in enumerate(reader):
            query = (record.get("Sentence") or "").strip()
            document = (record.get("English_Translation") or "").strip()
            if len(query) < 3 or len(document) < 3:
                continue
            qkey = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", query).casefold())
            if qkey in seen_queries:
                continue
            seen_queries.add(qkey)
            qid = hashlib.sha256(qkey.encode("utf-8")).hexdigest()
            docid = hashlib.sha256(
                unicodedata.normalize("NFKC", document).casefold().encode("utf-8")
            ).hexdigest()
            rows.append({"qid": qid, "query": query, "docid": docid,
                         "document": document})
    if not rows:
        raise ValueError("PHINC has no usable rows after deduplication")
    return rows, fields


def split_phinc(rows):
    # Split by the target translation document ID: duplicate translations cannot
    # cross fit/validation/test even if multiple Hinglish queries map to them.
    groups = {"fit": [], "validation": [], "test": []}
    for row in rows:
        bucket = stable_bucket(row["docid"], "phinc-target-document")
        split = "fit" if bucket < 70 else ("validation" if bucket < 85 else "test")
        groups[split].append(row)
    return groups


def phinc_pairs(rows):
    return [{"qid": row["qid"], "query": row["query"],
             "docid": row["docid"], "document": row["document"]} for row in rows]


def phinc_eval(rows):
    docs = {}
    qrels = {}
    for row in rows:
        docs.setdefault(row["docid"], row["document"])
        qrels[row["qid"]] = {row["docid"]: 1}
    ids = sorted(docs)
    return {"query_ids": [x["qid"] for x in rows],
            "query_texts": [x["query"] for x in rows],
            "doc_ids": ids, "doc_texts": [docs[x] for x in ids], "qrels": qrels}


class SparseProjection(torch.nn.Module):
    def __init__(self, active_k=32, seed=110):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = F.normalize(torch.randn(EXPANSION_DIM, TEXT_DIM, generator=generator), dim=-1)
        self.weight = torch.nn.Parameter(weight)
        self.active_k = active_k

    def forward(self, x):
        values = F.relu(x.float() @ self.weight.float().T)
        top_values, indices = values.topk(self.active_k, dim=-1)
        sparse = torch.zeros_like(values).scatter(1, indices, top_values)
        return F.normalize(sparse, dim=-1)


def contrastive_loss(qcodes, dcodes, groups, temperature=0.10):
    logits = (qcodes @ dcodes.T) / temperature
    positives = groups[:, None].eq(groups[None, :])
    log_qd = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    log_dq = logits.T - torch.logsumexp(logits.T, dim=1, keepdim=True)
    q_count = positives.sum(1).clamp_min(1)
    d_count = positives.sum(0).clamp_min(1)
    q_loss = -(log_qd.masked_fill(~positives, 0.0).sum(1) / q_count).mean()
    d_loss = -(log_dq.masked_fill(~positives.T, 0.0).sum(1) / d_count).mean()
    return 0.5 * (q_loss + d_loss)


def train_encoder(pairs, seed, epochs=8, active_k=32):
    if len(pairs) < 16:
        raise ValueError(f"not enough training pairs: {len(pairs)}")
    q = encode_texts([x["query"] for x in pairs])
    d = encode_texts([x["document"] for x in pairs])
    groups = torch.tensor([int(x["qid"][:15], 16) for x in pairs], dtype=torch.long)
    model = SparseProjection(active_k, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + active_k)
    losses = []
    model.train()
    started = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(len(pairs), generator=generator)
        for ids in order.split(128):
            qcodes, dcodes = model(q[ids]), model(d[ids])
            loss = contrastive_loss(qcodes, dcodes, groups[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - started
    model.eval()
    return model, {"pairs": len(pairs), "epochs": epochs, "steps": len(losses),
                   "initial_loss": losses[0], "final_loss": losses[-1],
                   "train_seconds": elapsed}


def score_matrix(eval_data, model=None):
    queries = encode_texts(eval_data["query_texts"])
    docs = encode_texts(eval_data["doc_texts"])
    with torch.inference_mode():
        if model is None:
            return F.normalize(queries, dim=-1) @ F.normalize(docs, dim=-1).T
        return model(queries) @ model(docs).T


def retrieval_metrics(scores, eval_data):
    doc_positions = {docid: i for i, docid in enumerate(eval_data["doc_ids"])}
    ranked = scores.argsort(dim=-1, descending=True)
    mrr, ndcg, precision = [], [], []
    recall = {1: [], 5: [], 10: []}
    for row_index, qid in enumerate(eval_data["query_ids"]):
        relevant = {doc_positions[x] for x in eval_data["qrels"].get(qid, {})
                    if x in doc_positions}
        if not relevant:
            continue
        order = ranked[row_index].tolist()
        hits = [int(index in relevant) for index in order[:10]]
        first = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
        mrr.append(1.0 / first if first else 0.0)
        dcg = sum(hit / torch.log2(torch.tensor(rank + 1.0)).item()
                  for rank, hit in enumerate(hits, 1))
        ideal = sum(1.0 / torch.log2(torch.tensor(rank + 1.0)).item()
                    for rank in range(1, min(len(relevant), 10) + 1))
        ndcg.append(dcg / ideal if ideal else 0.0)
        precision.append(sum(hits) / 10.0)
        for k in recall:
            recall[k].append(sum(1 for index in order[:k] if index in relevant) /
                             max(1, len(relevant)))
    if not mrr:
        raise ValueError("no evaluation query has a relevant candidate")
    return {
        "queries": len(mrr), "candidate_documents": len(eval_data["doc_ids"]),
        "mrr_at_10": statistics.mean(mrr), "ndcg_at_10": statistics.mean(ndcg),
        "precision_at_10": statistics.mean(precision),
        "recall_at_1": statistics.mean(recall[1]),
        "recall_at_5": statistics.mean(recall[5]),
        "recall_at_10": statistics.mean(recall[10]),
    }


def choose_fusion_weight(val_data, model):
    raw = score_matrix(val_data)
    learned = score_matrix(val_data, model)
    candidates = []
    for weight in [i / 10 for i in range(11)]:
        metrics = retrieval_metrics((1-weight)*raw + weight*learned, val_data)
        candidates.append((metrics["mrr_at_10"], -weight, weight, metrics))
    best = max(candidates, key=lambda x: (x[0], x[1]))
    return best[2], best[3]


def evaluate_task(name, fit_pairs, validation, test, seed, epochs, active_k):
    fit_qids = {x["qid"] for x in fit_pairs}
    val_qids = set(validation["query_ids"])
    test_qids = set(test["query_ids"])
    if fit_qids & val_qids or fit_qids & test_qids or val_qids & test_qids:
        raise ValueError(name + ": query split leakage")
    fit_docids = {x["docid"] for x in fit_pairs}
    val_docids = set(validation["doc_ids"])
    test_docids = set(test["doc_ids"])
    if fit_docids & val_docids or fit_docids & test_docids:
        raise ValueError(name + ": training passage overlaps evaluation candidates")
    model, training = train_encoder(fit_pairs, seed, epochs, active_k)
    weight, val_metrics = choose_fusion_weight(validation, model)
    raw_scores = score_matrix(test)
    learned_scores = score_matrix(test, model)
    hybrid_scores = (1-weight)*raw_scores + weight*learned_scores
    return {
        "task": name,
        "fit_pairs": len(fit_pairs),
        "validation_queries": len(validation["query_ids"]),
        "validation_candidates": len(validation["doc_ids"]),
        "test_queries": len(test["query_ids"]),
        "test_candidates": len(test["doc_ids"]),
        "training": training,
        "validation": {"objective": "MRR@10", "learned_fusion_weight": weight,
                       "mrr_at_10": val_metrics["mrr_at_10"]},
        "test": {"raw_cosine": retrieval_metrics(raw_scores, test),
                 "contrastive_dg": retrieval_metrics(learned_scores, test),
                 "validation_tuned_hybrid": retrieval_metrics(hybrid_scores, test)},
        "query_split_overlap": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/hcdm110_realtext_miracl.json"))
    parser.add_argument("--workdir", type=Path, default=Path("data/hcdm110_realtext"))
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--active-k", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.workdir.mkdir(parents=True, exist_ok=True)

    miracl_revision = HfApi().dataset_info(MIRACL_REPO).sha
    tasks, audits = [], {}
    for task_name, language in (("miracl_english", "en"), ("miracl_hindi", "hi")):
        train_rows, dev_rows = load_miracl(language, miracl_revision)
        fit_rows, validation_rows = [], []
        for row in train_rows:
            target = fit_rows if split_train_validation(row["qid"], language) == "fit" else validation_rows
            target.append(row)
        fit_pairs = training_pairs(fit_rows)
        validation = candidate_pool(validation_rows)
        test = candidate_pool(dev_rows)
        # Remove any fit passage that appears in either evaluation candidate pool.
        evaluation_docids = set(validation["doc_ids"]) | set(test["doc_ids"])
        fit_pairs = [pair for pair in fit_pairs if pair["docid"] not in evaluation_docids]
        tasks.append(evaluate_task(task_name, fit_pairs, validation, test,
                                   args.seed, args.epochs, args.active_k))
        audits[task_name] = {
            "train_queries_total": len(train_rows), "fit_queries": len(fit_rows),
            "validation_queries": len(validation_rows), "official_dev_test_queries": len(dev_rows),
            "fit_pairs_after_document_overlap_filter": len(fit_pairs),
            "validation_candidates": len(validation["doc_ids"]),
            "test_candidates": len(test["doc_ids"]),
            "official_split_use": "train qids split into fit/validation; official dev reserved for test",
            "fit_document_overlap_with_validation_or_test": 0,
        }

    phinc_path = args.workdir / "PHINC.csv"
    provenance = {
        "miracl": {"repo": MIRACL_REPO, "revision": miracl_revision,
                   "license": "Apache-2.0", "usage": "topics/qrels positive and negative passages only"},
        "phinc": {"url": PHINC_URL, "license": "CC-BY-4.0",
                  "attribution": "Lingo Research Group, IIT Gandhinagar",
                  "citation": "Srivastava & Singh, W-NUT 2020, DOI 10.18653/v1/2020.wnut-1.7"},
    }
    phinc_download = download_phinc(phinc_path)
    provenance["phinc"].update(phinc_download)
    phinc_rows, phinc_columns = load_phinc(phinc_path)
    phinc_splits = split_phinc(phinc_rows)
    fit_pairs = phinc_pairs(phinc_splits["fit"])
    validation = phinc_eval(phinc_splits["validation"])
    test = phinc_eval(phinc_splits["test"])
    tasks.append(evaluate_task("phinc_hinglish_to_english", fit_pairs, validation, test,
                               args.seed, args.epochs, args.active_k))
    audits["phinc_hinglish_to_english"] = {
        "rows_after_query_deduplication": len(phinc_rows),
        "fit_pairs": len(fit_pairs), "validation_queries": len(validation["query_ids"]),
        "test_queries": len(test["query_ids"]), "validation_candidates": len(validation["doc_ids"]),
        "test_candidates": len(test["doc_ids"]), "columns": phinc_columns,
        "split_rule": "stable hash of target translation document id; no target-doc overlap",
    }

    report = {
        "benchmark": "HCDM-110 real-language retrieval v2",
        "text_representation": "384-d Unicode word + character n-gram feature hashing; no pretrained model",
        "learned_encoder": {"architecture": "shared 384->768 trainable projection, ReLU top-k, L2 normalize",
                            "active_k": args.active_k, "contrastive_objective": "symmetric multi-positive InfoNCE",
                            "epochs": args.epochs, "batch_size": 128},
        "tasks": tasks, "dataset_audit": audits, "source_provenance": provenance,
        "guardrails": [
            "Hindi and English use MIRACL train/dev queries with native relevance judgments and annotated positive/negative passages.",
            "MIRACL official train queries are split by qid into fit and validation; official dev is reserved for final test.",
            "The MIRACL candidate pool is the union of annotated positive/negative passages for the evaluation split, not the full Wikipedia corpus.",
            "Hinglish uses PHINC human English translations as positive retrieval targets; this is cross-lingual sentence retrieval, not monolingual Hinglish search.",
            "PHINC is split by target translation document id to prevent exact target-document overlap across splits.",
            "Hybrid weight is selected only on validation MRR@10.",
            "This is lexical feature hashing, not ANVIKSHA/1B model embeddings.",
            "Only aggregate counts and metrics are printed; no source text is logged or committed.",
            "No production checkpoint loaded or modified.",
        ],
        "checkpoint_loaded": False, "production_model_modified": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
