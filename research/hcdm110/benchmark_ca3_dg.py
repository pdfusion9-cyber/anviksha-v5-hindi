from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from hcdm110_prototype import AssociativeMemory, DGPatternEncoder


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def metrics(top_ids: torch.Tensor, relevance: torch.Tensor, k: int = 10):
    # One query has exactly five relevant episodes (same topic + same facet).
    retrieved_relevance = relevance[top_ids]
    hits = (retrieved_relevance == relevance[:, None]).sum(-1).float()
    precision = hits / float(k)
    recall = hits / 5.0
    return {
        "precision_at_10": float(precision.mean()),
        "recall_at_10": float(recall.mean()),
        "top1_relevant_accuracy": float((retrieved_relevance[:, 0] == relevance).float().mean()),
        "mean_relevant_hits_at_10": float(hits.mean()),
    }


def timed_batches(fn, batches, warmups=2, repeats=3):
    for _ in range(warmups):
        for batch in batches:
            fn(batch)
    per_batch = []
    for _ in range(repeats):
        for batch in batches:
            start = time.perf_counter()
            fn(batch)
            per_batch.append((time.perf_counter() - start) * 1000.0)
    per_example = [ms / max(batch.shape[0], 1) for ms, batch in zip(per_batch, batches * repeats)]
    return {
        "batch_latency_ms_p50": statistics.median(per_batch),
        "batch_latency_ms_p95": percentile(per_batch, 0.95),
        "amortized_per_query_ms_p50": statistics.median(per_example),
        "amortized_per_query_ms_p95": percentile(per_example, 0.95),
        "measured_batches": len(per_batch),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/hcdm110_ca3_dg_benchmark.json"))
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--topics", type=int, default=100)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--expansion-dim", type=int, default=4096)
    parser.add_argument("--active-k", type=int, default=64)
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=110)
    args = parser.parse_args()

    if args.examples != 1000 or args.topics <= 0 or args.examples % (args.topics * 2):
        raise SystemExit("benchmark expects exactly 1,000 examples and equal topic/facet groups")
    per_facet = args.examples // (args.topics * 2)
    if per_facet != 5 or args.retrieve_k != 10:
        raise SystemExit("benchmark contract expects five relevant episodes and top-10 retrieval")
    torch.set_num_threads(2)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # 100 topics x 2 facets x 5 distinct episodes = 1,000 records.
    topic_vectors = F.normalize(torch.randn(args.topics, args.dim, generator=generator), dim=-1)
    facet_vectors = F.normalize(torch.randn(args.topics, 2, args.dim, generator=generator), dim=-1)
    rows, relevance, record_ids = [], [], []
    for topic in range(args.topics):
        for facet in range(2):
            for episode in range(per_facet):
                noise = torch.randn(args.dim, generator=generator)
                vector = topic_vectors[topic] + 0.30 * facet_vectors[topic, facet] + 0.025 * noise
                rows.append(F.normalize(vector, dim=-1))
                relevance.append(topic * 2 + facet)
                record_ids.append((topic, facet, episode))
    raw = torch.stack(rows)
    relevance = torch.tensor(relevance, dtype=torch.long)
    n = raw.shape[0]

    query_noise = torch.randn(raw.shape, generator=generator) * 0.012
    queries = F.normalize(raw + query_noise, dim=-1)

    dg = DGPatternEncoder(args.dim, args.expansion_dim, args.active_k, seed=args.seed)
    with torch.inference_mode():
        dg_keys = dg(raw)
        dg_queries = dg(queries)

    # Baseline: raw pooled-vector cosine retrieval.
    raw_sim = queries @ raw.T
    raw_top = raw_sim.topk(args.retrieve_k, dim=-1).indices
    raw_metrics = metrics(raw_top, relevance, args.retrieve_k)

    # DG + CA3: sparse-code cosine retrieval through the isolated memory engine.
    memory = AssociativeMemory(n, args.expansion_dim, value_dim=1)
    memory.add(dg_keys, relevance.float().unsqueeze(-1))
    dg_top_parts = []
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            retrieved, scores, slots = memory.retrieve(
                dg_queries[start:start + args.batch_size], top_k=args.retrieve_k
            )
            dg_top_parts.append(slots.cpu())
    dg_top = torch.cat(dg_top_parts, dim=0)
    dg_metrics = metrics(dg_top, relevance, args.retrieve_k)

    # Pattern-separation diagnostics over all 2,000 distinct pairs sharing
    # topic+facet. Exact collision = identical active sets; near collision =
    # Jaccard overlap >= 0.50. These thresholds are explicit engineering
    # diagnostics, not universal neuroscience standards.
    support = dg_keys > 0
    same_group_pairs = []
    for i in range(n):
        topic_i, facet_i, _ = record_ids[i]
        for j in range(i + 1, n):
            if record_ids[j][0] == topic_i and record_ids[j][1] == facet_i:
                same_group_pairs.append((i, j))
    if len(same_group_pairs) != args.topics * 2 * (per_facet * (per_facet - 1) // 2):
        raise RuntimeError("unexpected same-facet pair count")

    intersections, unions = [], []
    raw_cosines = []
    exact_collisions = 0
    near_collisions = 0
    for i, j in same_group_pairs:
        a, b = support[i], support[j]
        intersection = int((a & b).sum())
        union = int((a | b).sum())
        jaccard = intersection / max(union, 1)
        intersections.append(jaccard)
        raw_cosines.append(float(torch.dot(raw[i], raw[j])))
        exact_collisions += int(torch.equal(a, b))
        near_collisions += int(jaccard >= 0.50)
    collision = {
        "definition": "Distinct episodes from same topic+facet; exact collision means identical DG active-feature supports; near collision means support Jaccard >= 0.50.",
        "same_facet_distinct_pairs": len(same_group_pairs),
        "raw_cosine_mean_for_pairs": statistics.mean(raw_cosines),
        "dg_support_jaccard_mean": statistics.mean(intersections),
        "dg_support_jaccard_p95": percentile(intersections, 0.95),
        "exact_support_collision_count": exact_collisions,
        "exact_support_collision_rate": exact_collisions / len(same_group_pairs),
        "near_support_collision_count": near_collisions,
        "near_support_collision_rate": near_collisions / len(same_group_pairs),
        "near_collision_threshold": 0.50,
    }

    chunks_raw = [queries[i:i + args.batch_size] for i in range(0, n, args.batch_size)]
    chunks_code = [dg_queries[i:i + args.batch_size] for i in range(0, n, args.batch_size)]

    def encode_fn(batch):
        with torch.inference_mode():
            return dg(batch)

    def raw_retrieve_fn(batch):
        with torch.inference_mode():
            return (F.normalize(batch, dim=-1) @ F.normalize(raw, dim=-1).T).topk(
                args.retrieve_k, dim=-1
            )

    def dg_retrieve_fn(batch):
        with torch.inference_mode():
            return memory.retrieve(batch, top_k=args.retrieve_k)

    timing = {
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "dg_encode": timed_batches(encode_fn, chunks_raw, warmups=2, repeats=3),
        "raw_cosine_retrieval": timed_batches(raw_retrieve_fn, chunks_raw, warmups=1, repeats=3),
        "dg_ca3_retrieval": timed_batches(dg_retrieve_fn, chunks_code, warmups=1, repeats=3),
        "note": "Latency is CPU-only and host-specific; amortized per-query figures divide batch latency by batch size."
    }

    report = {
        "benchmark": "HCDM-110 isolated DG/CA3 synthetic retrieval benchmark v1",
        "seed": args.seed,
        "examples": n,
        "topics": args.topics,
        "facets_per_topic": 2,
        "episodes_per_facet": per_facet,
        "relevant_episodes_per_query": per_facet,
        "dimensions": {
            "pooled_input": args.dim,
            "dg_expansion": args.expansion_dim,
            "dg_active_features": args.active_k,
            "retrieval_k": args.retrieve_k,
        },
        "synthetic_generation": {
            "formula": "normalize(topic_vector + 0.30*facet_vector + 0.025*Gaussian_noise)",
            "query": "normalize(record_vector + 0.012*Gaussian_noise)",
            "purpose": "known-label retrieval stress test; not a natural-language or model-quality evaluation"
        },
        "raw_cosine_baseline": raw_metrics,
        "dg_ca3_sparse_retrieval": dg_metrics,
        "pattern_separation_collision": collision,
        "latency": timing,
        "interpretation_guardrails": [
            "Synthetic retrieval results do not establish improvement in language-model quality.",
            "Precision@10 and recall@10 are distinct because each query has five relevant episodes but retrieves ten.",
            "Collision thresholds are engineering diagnostics, not biological constants.",
            "No production checkpoint is loaded, modified, or saved."
        ],
        "checkpoint_loaded": False,
        "production_model_modified": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
