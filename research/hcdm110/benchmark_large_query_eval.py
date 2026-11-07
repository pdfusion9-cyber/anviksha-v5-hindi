from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_contrastive_dg import (
    make_cues, train_encoder, choose_fusion_weight, supervised_contrastive_loss,
)
from hcdm110_prototype import LearnableDGEncoder


SEEDS = [601, 702, 803, 904, 1005]
NOISES = [0.20, 0.30]
TOP_K = 32
EXPANSION = 1024
QUERIES_PER_FACET = 50
BATCH_SIZE = 512


def query_metrics(scores, labels, query_labels):
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
        "precision_at_10": relevant.float().mean(dim=-1),
        "recall_at_10": relevant.sum(dim=-1).float() / 5.0,
        "top1_relevant_accuracy": relevant[:, 0].float(),
        "ndcg_at_10": dcg / idcg,
        "wrong_facet_at_10_rate": wrong_facet.float().mean(dim=-1),
        "unrelated_topic_at_10_rate": (~same_topic).float().mean(dim=-1),
    }


def evaluate_method(method, qx, qy, memory_x, memory_y, encoder=None, weight=0.0):
    parts = {key: [] for key in (
        "precision_at_10", "recall_at_10", "top1_relevant_accuracy", "ndcg_at_10",
        "wrong_facet_at_10_rate", "unrelated_topic_at_10_rate")}
    raw_keys = F.normalize(memory_x.float(), dim=-1)
    with torch.inference_mode():
        if method in ("learned", "hybrid"):
            learned_keys = F.normalize(encoder(memory_x), dim=-1)
        for start in range(0, qx.shape[0], BATCH_SIZE):
            q = qx[start:start+BATCH_SIZE]
            qlabels = qy[start:start+BATCH_SIZE]
            raw_q = F.normalize(q.float(), dim=-1)
            raw_scores = raw_q @ raw_keys.T
            if method == "raw":
                scores = raw_scores
            else:
                learned_q = F.normalize(encoder(q), dim=-1)
                learned_scores = learned_q @ learned_keys.T
                scores = learned_scores if method == "learned" else (
                    (1.0-weight) * raw_scores + weight * learned_scores)
            batch_metrics = query_metrics(scores, memory_y, qlabels)
            for key, value in batch_metrics.items():
                parts[key].append(value.cpu())
    return {key: torch.cat(values) for key, values in parts.items()}


def timed_retrieval(method, qx, memory_x, encoder=None, weight=0.0, repeats=3):
    raw_keys = F.normalize(memory_x.float(), dim=-1)
    with torch.inference_mode():
        learned_keys = F.normalize(encoder(memory_x), dim=-1) if method != "raw" else None
        # Warmup is excluded from reported latency.
        q = qx[:min(BATCH_SIZE, qx.shape[0])]
        raw = F.normalize(q.float(), dim=-1) @ raw_keys.T
        if method != "raw":
            learned = F.normalize(encoder(q), dim=-1) @ learned_keys.T
            _ = learned if method == "learned" else (1-weight)*raw + weight*learned
        else:
            _ = raw
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for start in range(0, qx.shape[0], BATCH_SIZE):
                q = qx[start:start+BATCH_SIZE]
                raw = F.normalize(q.float(), dim=-1) @ raw_keys.T
                if method != "raw":
                    learned = F.normalize(encoder(q), dim=-1) @ learned_keys.T
                    scores = learned if method == "learned" else (
                        (1-weight)*raw + weight*learned)
                else:
                    scores = raw
                scores.topk(10, dim=-1)
            times.append((time.perf_counter()-t0)*1000)
    return {"full_query_set_p50_ms": statistics.median(times),
            "full_query_set_p95_ms": sorted(times)[-1],
            "query_count": int(qx.shape[0]), "batch_size": BATCH_SIZE,
            "repeats": repeats, "includes_query_encoding": method != "raw"}


def paired_bootstrap_ci(delta, seed, repeats=400, chunk=10):
    delta = delta.float().cpu()
    n = delta.numel()
    gen = torch.Generator().manual_seed(seed + n * 17)
    means = []
    for start in range(0, repeats, chunk):
        count = min(chunk, repeats-start)
        idx = torch.randint(0, n, (count, n), generator=gen)
        means.append(delta[idx].mean(dim=1))
    samples = torch.cat(means)
    return {"mean_difference": float(delta.mean()),
            "ci95_low": float(torch.quantile(samples, 0.025)),
            "ci95_high": float(torch.quantile(samples, 0.975)),
            "queries": n, "bootstrap_replicates": repeats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/hcdm110_large_query_eval.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--noises", type=float, nargs="+", default=NOISES)
    ap.add_argument("--queries-per-facet", type=int, default=QUERIES_PER_FACET)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--expansion", type=int, default=EXPANSION)
    ap.add_argument("--bootstrap-replicates", type=int, default=400)
    ap.add_argument("--timing-repeats", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(2)
    all_tests = {noise: {"raw": {k: [] for k in (
        "precision_at_10","recall_at_10","top1_relevant_accuracy","ndcg_at_10")},
        "learned": {k: [] for k in ("precision_at_10","recall_at_10","top1_relevant_accuracy","ndcg_at_10")},
        "hybrid": {k: [] for k in ("precision_at_10","recall_at_10","top1_relevant_accuracy","ndcg_at_10")}}
        for noise in args.noises}
    seed_reports = []

    for seed in args.seeds:
        gen = torch.Generator().manual_seed(seed)
        topics = F.normalize(torch.randn(100,384,generator=gen),dim=-1)
        facets = F.normalize(torch.randn(100,2,384,generator=gen),dim=-1)
        train_x, train_y = make_cues(topics,facets,12,0.08,gen)
        memory_x, memory_y = make_cues(topics,facets,5,0.025,gen)
        validation = {noise:make_cues(topics,facets,3,noise,gen)
                      for noise in (0.10,0.20)}
        train_seed_start = time.perf_counter()
        encoder, train_report = train_encoder(
            train_x,train_y,args.expansion,TOP_K,seed,args.epochs)
        train_report["wall_seconds"] = time.perf_counter()-train_seed_start

        with torch.inference_mode():
            raw_mem = F.normalize(memory_x,dim=-1)
            learned_mem = F.normalize(encoder(memory_x),dim=-1)
        vx = torch.cat([validation[n][0] for n in (0.10,0.20)])
        vy = torch.cat([validation[n][1] for n in (0.10,0.20)])
        with torch.inference_mode():
            raw_val = F.normalize(vx,dim=-1) @ raw_mem.T
            learned_val = F.normalize(encoder(vx),dim=-1) @ learned_mem.T
        weight, val_metrics = choose_fusion_weight(
            raw_val,learned_val,memory_y,vy)
        seed_result = {"seed":seed,"training":train_report,
            "validation":{"fusion_learned_weight":weight,
             "objective":"validation NDCG@10 only","metrics":val_metrics},
            "test":[]}

        for noise in args.noises:
            # 50 independent queries per each of 200 classes = 10,000 queries.
            qx,qy = make_cues(topics,facets,args.queries_per_facet,noise,gen)
            raw_metrics = evaluate_method("raw",qx,qy,memory_x,memory_y)
            learned_metrics = evaluate_method(
                "learned",qx,qy,memory_x,memory_y,encoder)
            hybrid_metrics = evaluate_method(
                "hybrid",qx,qy,memory_x,memory_y,encoder,weight)
            for method, metrics in (("raw",raw_metrics),("learned",learned_metrics),
                                   ("hybrid",hybrid_metrics)):
                for key in all_tests[noise][method]:
                    all_tests[noise][method][key].append(metrics[key])
            times = {
                "raw":timed_retrieval("raw",qx,memory_x,repeats=args.timing_repeats),
                "learned":timed_retrieval("learned",qx,memory_x,encoder,repeats=args.timing_repeats),
                "hybrid":timed_retrieval("hybrid",qx,memory_x,encoder,weight,repeats=args.timing_repeats)
            }
            seed_result["test"].append({
                "noise":noise,"queries":int(qx.shape[0]),
                "raw_cosine":{k:float(v.mean()) for k,v in raw_metrics.items()},
                "contrastive_dg":{k:float(v.mean()) for k,v in learned_metrics.items()},
                "validation_tuned_hybrid":{k:float(v.mean()) for k,v in hybrid_metrics.items()},
                "end_to_end_latency":times})
        seed_reports.append(seed_result)
        del encoder, train_x, train_y, memory_x, memory_y, topics, facets

    summary=[]
    for noise in args.noises:
        row={"noise":noise,"queries_per_seed":args.queries_per_facet*200,
             "total_queries":args.queries_per_facet*200*len(args.seeds)}
        metrics_by_method={}
        for method in ("raw","learned","hybrid"):
            metrics_by_method[method]={}
            for key,parts in all_tests[noise][method].items():
                merged=torch.cat(parts)
                metrics_by_method[method][key]=float(merged.mean())
                metrics_by_method[method][key+"_queries"]=int(merged.numel())
        row["raw_cosine"]=metrics_by_method["raw"]
        row["contrastive_dg"]=metrics_by_method["learned"]
        row["validation_tuned_hybrid"]=metrics_by_method["hybrid"]
        row["paired_bootstrap_vs_raw"]={}
        for other in ("learned","hybrid"):
            row["paired_bootstrap_vs_raw"][other]={}
            for metric in ("precision_at_10","recall_at_10","top1_relevant_accuracy","ndcg_at_10"):
                delta=torch.cat(all_tests[noise][other][metric])-torch.cat(all_tests[noise]["raw"][metric])
                row["paired_bootstrap_vs_raw"][other][metric]=paired_bootstrap_ci(
                    delta, int(noise*10000)+len(args.seeds), args.bootstrap_replicates)
        summary.append(row)

    report={
        "benchmark":"HCDM-110 large independent-query paired comparison v1",
        "seeds":args.seeds,"test_noises":args.noises,
        "dataset":{"topics":100,"facets_per_topic":2,"memory_episodes_per_facet":5,
          "memory_records_per_seed":1000,"queries_per_facet_per_noise":args.queries_per_facet,
          "queries_per_seed_per_noise":args.queries_per_facet*200,
          "total_queries_per_noise_all_seeds":args.queries_per_facet*200*len(args.seeds),
          "training_validation_memory_test_disjoint":True,
          "validation_noises":[0.10,0.20],"validation_samples_per_facet":3},
        "model":{"encoder":"supervised contrastive LearnableDGEncoder","top_k":TOP_K,
          "expansion_dim":args.expansion,"epochs":args.epochs},
        "summary":summary,"per_seed":seed_reports,
        "guardrails":["Five new seeds by default: 601, 702, 803, 904, 1005.",
          "Test queries are independent random draws, 50 per facet and noise per seed.",
          "Fusion weights selected only from validation NDCG@10.",
          "Paired bootstrap intervals are over per-query differences pooled across seeds.",
          "Synthetic task; not evidence of natural-language/model quality.",
          "No production checkpoint loaded or modified."],
        "checkpoint_loaded":False,"production_model_modified":False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
