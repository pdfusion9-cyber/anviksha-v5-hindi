from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from hcdm110_prototype import DGPatternEncoder, LearnableDGEncoder


NEW_SEEDS = [101, 202, 303, 404, 505]
TOP_KS = [32, 64, 128]
NOISES = [0.05, 0.10, 0.20, 0.30]


def make_cues(topics, facets, samples_per_facet, noise, generator):
    xs, ys = [], []
    for topic in range(topics.shape[0]):
        for facet in range(facets.shape[1]):
            label = topic * facets.shape[1] + facet
            eps = torch.randn(samples_per_facet, topics.shape[-1], generator=generator)
            cue = topics[topic] + 0.30 * facets[topic, facet] + noise * eps
            xs.append(F.normalize(cue, dim=-1))
            ys.append(torch.full((samples_per_facet,), label, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def supervised_contrastive_loss(features, labels, temperature=0.10):
    features = F.normalize(features.float(), dim=-1)
    logits = (features @ features.T) / temperature
    batch = labels.shape[0]
    self_mask = torch.eye(batch, dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~self_mask
    denominator_logits = logits.masked_fill(self_mask, -1e9)
    log_prob = logits - torch.logsumexp(denominator_logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        raise ValueError("contrastive batch has no positive pairs")
    per_anchor = -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) /
                   positive_count.clamp_min(1))
    return per_anchor[valid].mean()


def train_encoder(train_x, train_y, expansion, active_k, seed, epochs):
    encoder = LearnableDGEncoder(384, expansion, active_k, seed=seed)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    class_to_rows = [torch.nonzero(train_y == c, as_tuple=False).flatten()
                     for c in range(int(train_y.max()) + 1)]
    gen = torch.Generator().manual_seed(seed + 91_000 + active_k)
    losses = []
    encoder.train()
    started = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(len(class_to_rows), generator=gen)
        for class_ids in order.split(16):
            batch_rows = []
            for class_id in class_ids.tolist():
                rows = class_to_rows[class_id]
                picks = torch.randperm(rows.numel(), generator=gen)[:8]
                batch_rows.append(rows[picks])
            ids = torch.cat(batch_rows)
            z = encoder(train_x[ids])
            loss = supervised_contrastive_loss(z, train_y[ids], temperature=0.10)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - started
    encoder.eval()
    return encoder, {"seconds": elapsed, "initial_loss": losses[0],
                     "final_loss": losses[-1], "optimizer_steps": len(losses)}


def rank_metrics(scores, labels, query_labels):
    top = scores.topk(10, dim=-1).indices
    retrieved = labels[top]
    relevant = retrieved.eq(query_labels[:, None])
    same_topic = torch.div(retrieved, 2, rounding_mode="floor").eq(
        torch.div(query_labels, 2, rounding_mode="floor")[:, None])
    wrong_facet = same_topic & ~relevant
    discounts = 1.0 / torch.log2(torch.arange(2, 12, dtype=torch.float32))
    dcg = (relevant.float() * discounts).sum(dim=-1)
    idcg = discounts[:5].sum()
    return {
        "precision_at_10": float(relevant.float().mean()),
        "recall_at_10": float(relevant.sum(dim=-1).float().mean() / 5.0),
        "top1_relevant_accuracy": float(relevant[:, 0].float().mean()),
        "ndcg_at_10": float((dcg / idcg).mean()),
        "wrong_facet_at_10_rate": float(wrong_facet.float().mean()),
        "unrelated_topic_at_10_rate": float((~same_topic).float().mean()),
    }


def cosine_scores(q, keys):
    return F.normalize(q.float(), dim=-1) @ F.normalize(keys.float(), dim=-1).T


def choose_fusion_weight(raw_scores, learned_scores, labels, query_labels):
    # Validation-only grid search; maximises NDCG@10, tie-breaks toward raw baseline.
    candidates = []
    for learned_weight in [i / 10 for i in range(11)]:
        fused = (1.0 - learned_weight) * raw_scores + learned_weight * learned_scores
        metrics = rank_metrics(fused, labels, query_labels)
        candidates.append((metrics["ndcg_at_10"], -learned_weight, learned_weight, metrics))
    best = max(candidates, key=lambda item: (item[0], item[1]))
    return best[2], best[3]


def build_large_memory(topics, facets, episodes_per_facet, noise, generator):
    return make_cues(topics, facets, episodes_per_facet, noise, generator)


def encode_in_batches(encoder, x, batch_size=2048):
    chunks = []
    with torch.inference_mode():
        for chunk in x.split(batch_size):
            chunks.append(encoder(chunk).cpu())
    return torch.cat(chunks, dim=0)


def memory_scale(encoder, topics, facets, capacities, seed, active_k, expansion):
    results = []
    for capacity in capacities:
        if capacity % 200:
            raise ValueError("scale capacity must be divisible by 200 classes")
        episodes = capacity // 200
        gen = torch.Generator().manual_seed(seed * 100_000 + active_k * 100 + capacity)
        memory_x, memory_y = build_large_memory(topics, facets, episodes, 0.025, gen)
        encoded = encode_in_batches(encoder, memory_x)
        query_x, query_y = make_cues(topics, facets, 1, 0.20, gen)
        query_x = query_x[:32]
        query_y = query_y[:32]
        with torch.inference_mode():
            query_codes = encoder(query_x)
            query_codes = F.normalize(query_codes, dim=-1)
            keys = F.normalize(encoded, dim=-1)
            scores = query_codes @ keys.T
            top = scores.topk(10, dim=-1).indices
            retrieved_labels = memory_y[top]
            relevant = retrieved_labels.eq(query_y[:, None])
            warm = (query_codes @ keys.T).topk(10, dim=-1).indices
            latencies = []
            for _ in range(5):
                t0 = time.perf_counter()
                scores = query_codes @ keys.T
                scores.topk(10, dim=-1)
                latencies.append((time.perf_counter() - t0) * 1000)
        results.append({
            "capacity": capacity,
            "episodes_per_facet": episodes,
            "encoder_active_k": active_k,
            "expansion_dim": expansion,
            "encoded_memory_bytes": int(encoded.numel() * encoded.element_size()),
            "mean_precision_at_10_batch32": float(relevant.float().mean()),
            "mean_top1_relevant_accuracy_batch32": float(relevant[:, 0].float().mean()),
            "retrieval_batch32_p50_ms": statistics.median(latencies),
            "retrieval_batch32_p95_ms": sorted(latencies)[-1],
            "queries": 32,
            "latency_repeats": len(latencies),
            "retrieval": "exact dense cosine + topk on CPU",
        })
        del memory_x, memory_y, encoded, query_x, query_y, query_codes, keys, scores, top
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/hcdm110_contrastive_dg.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=NEW_SEEDS)
    parser.add_argument("--top-k", type=int, nargs="+", default=TOP_KS)
    parser.add_argument("--noises", type=float, nargs="+", default=NOISES)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--expansion", type=int, default=1024)
    parser.add_argument("--scale-capacities", type=int, nargs="+", default=[10_000, 100_000])
    args = parser.parse_args()
    torch.set_num_threads(2)
    all_runs = []
    scaling = []

    for seed in args.seeds:
        gen = torch.Generator().manual_seed(seed)
        topics = F.normalize(torch.randn(100, 384, generator=gen), dim=-1)
        facets = F.normalize(torch.randn(100, 2, 384, generator=gen), dim=-1)
        train_x, train_y = make_cues(topics, facets, 12, 0.08, gen)
        memory_x, memory_y = make_cues(topics, facets, 5, 0.025, gen)
        validation_sets = {}
        for noise in (0.10, 0.20):
            validation_sets[noise] = make_cues(topics, facets, 3, noise, gen)
        test_sets = {noise: make_cues(topics, facets, 1, noise, gen)
                     for noise in args.noises}

        raw_memory = F.normalize(memory_x, dim=-1)
        fixed_by_k = {}
        for active_k in args.top_k:
            fixed = DGPatternEncoder(384, args.expansion, active_k, seed=seed)
            with torch.inference_mode():
                fixed_by_k[active_k] = F.normalize(encode_in_batches(fixed, memory_x), dim=-1)

        for active_k in args.top_k:
            encoder, train_report = train_encoder(
                train_x, train_y, args.expansion, active_k, seed, args.epochs)
            with torch.inference_mode():
                learned_memory = F.normalize(encode_in_batches(encoder, memory_x), dim=-1)

            val_raw, val_learned, val_labels = [], [], []
            for noise in (0.10, 0.20):
                vx, vy = validation_sets[noise]
                with torch.inference_mode():
                    val_raw.append(cosine_scores(vx, raw_memory))
                    val_learned.append(cosine_scores(encoder(vx), learned_memory))
                val_labels.append(vy)
            raw_val = torch.cat(val_raw)
            learned_val = torch.cat(val_learned)
            labels_val = torch.cat(val_labels)
            learned_weight, validation_metrics = choose_fusion_weight(
                raw_val, learned_val, memory_y, labels_val)

            seed_k_report = {
                "seed": seed, "active_k": active_k, "training": train_report,
                "validation": {"fusion_learned_weight": learned_weight,
                               "objective": "NDCG@10 over noise 0.10 and 0.20",
                               "best_validation_metrics": validation_metrics},
                "test": []
            }
            for noise, (qx, qy) in test_sets.items():
                with torch.inference_mode():
                    raw_scores = cosine_scores(qx, raw_memory)
                    fixed_scores = cosine_scores(
                        F.normalize(encode_in_batches(
                            DGPatternEncoder(384, args.expansion, active_k, seed=seed), qx), dim=-1),
                        fixed_by_k[active_k])
                    learned_scores = cosine_scores(encoder(qx), learned_memory)
                    hybrid_scores = (1.0 - learned_weight) * raw_scores + learned_weight * learned_scores
                seed_k_report["test"].append({
                    "noise": noise,
                    "raw_cosine": rank_metrics(raw_scores, memory_y, qy),
                    "fixed_dg": rank_metrics(fixed_scores, memory_y, qy),
                    "learned_contrastive_dg": rank_metrics(learned_scores, memory_y, qy),
                    "hybrid_validation_tuned": rank_metrics(hybrid_scores, memory_y, qy)
                })
            all_runs.append(seed_k_report)

            # Full 10K/100K retrieval scale is measured for each independently
            # trained seed/K encoder, using newly sampled memory episodes.
            scaling.extend([{"seed": seed, **item} for item in memory_scale(
                encoder, topics, facets, args.scale_capacities, seed, active_k, args.expansion)])
            del encoder, learned_memory

    summary = []
    for active_k in args.top_k:
        for noise in args.noises:
            row = {"active_k": active_k, "noise": noise}
            for method in ("raw_cosine", "fixed_dg", "learned_contrastive_dg",
                           "hybrid_validation_tuned"):
                metrics = [test[method] for run in all_runs if run["active_k"] == active_k
                           for test in run["test"] if test["noise"] == noise]
                for key in metrics[0]:
                    row[method + "_" + key] = sum(m[key] for m in metrics) / len(metrics)
            weights = [run["validation"]["fusion_learned_weight"] for run in all_runs
                       if run["active_k"] == active_k]
            row["mean_validation_tuned_learned_weight"] = sum(weights) / len(weights)
            summary.append(row)

    report = {
        "benchmark": "HCDM-110 supervised contrastive DG, top-k and validation-tuned hybrid v1",
        "seeds": args.seeds, "top_k_values": args.top_k, "noise_levels": args.noises,
        "training": {"loss": "supervised contrastive loss", "temperature": 0.10,
                     "epochs": args.epochs, "balanced_batch": "16 classes x 8 samples",
                     "train_examples_per_facet": 12},
        "dataset": {"topics": 100, "facets_per_topic": 2, "memory_episodes_per_facet": 5,
                    "memory_records": 1000, "validation_samples_per_facet_per_noise": 3,
                    "validation_noises": [0.10, 0.20], "test_samples_per_facet_per_noise": 1,
                    "test_noises": args.noises, "train_memory_validation_test_disjoint": True,
                    "fusion_weight_selection": "validation NDCG@10 grid 0.0..1.0 in 0.1 steps"},
        "summary_mean_over_new_seeds": summary,
        "per_seed_per_k": all_runs,
        "memory_scaling": scaling,
        "scaling_contract": {"capacities": args.scale_capacities,
                             "classes": 200, "episodes_per_class": "capacity / 200",
                             "batch_queries": 32, "key_dim": args.expansion,
                             "retrieval": "exact dense cosine + topk"},
        "guardrails": ["Five new seeds are used by default: 101, 202, 303, 404, 505.",
                       "Training, memory, validation, and test are independently sampled.",
                       "Fusion weight is selected only on validation queries.",
                       "Synthetic task results are not evidence of language-model quality.",
                       "No production checkpoint loaded or modified."],
        "checkpoint_loaded": False, "production_model_modified": False
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
