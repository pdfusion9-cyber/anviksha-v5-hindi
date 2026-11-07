from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from hcdm110_prototype import AssociativeMemory, DGPatternEncoder


def pctl(xs, q):
    ys=sorted(float(x) for x in xs)
    return ys[min(len(ys)-1, max(0, round((len(ys)-1)*q)))]


def score(top_ids, labels, query_labels, k=10):
    found=labels[top_ids]
    rel=found.eq(query_labels[:,None])
    return {
        "precision_at_10":float(rel.float().mean()),
        "recall_at_10":float(rel.sum(-1).float().mean()/5.0),
        "top1_relevant_accuracy":float(rel[:,0].float().mean()),
        "cross_facet_false_retrieval_rate":float((~rel).float().mean()),
        "mean_relevant_hits_at_10":float(rel.sum(-1).float().mean())
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("reports/hcdm110_hard_dg_ca3.json"))
    ap.add_argument("--seeds",type=int,nargs="+",default=[11,22,33,44,55])
    ap.add_argument("--noises",type=float,nargs="+",default=[0.05,0.10,0.20,0.30])
    ap.add_argument("--scale-capacities",type=int,nargs="+",default=[1000,10000,100000])
    ap.add_argument("--dim",type=int,default=384)
    ap.add_argument("--expansion",type=int,default=4096)
    ap.add_argument("--active-k",type=int,default=64)
    args=ap.parse_args()
    torch.set_num_threads(2)
    results=[]
    for seed in args.seeds:
        g=torch.Generator().manual_seed(seed)
        topics=F.normalize(torch.randn(100,args.dim,generator=g),dim=-1)
        facets=F.normalize(torch.randn(100,2,args.dim,generator=g),dim=-1)
        rows=[]; labels=[]
        for t in range(100):
            for f in range(2):
                for _ in range(5):
                    noise=torch.randn(args.dim,generator=g)
                    rows.append(F.normalize(topics[t]+0.30*facets[t,f]+0.025*noise,dim=-1))
                    labels.append(t*2+f)
        keys=torch.stack(rows); labels=torch.tensor(labels)
        dg=DGPatternEncoder(args.dim,args.expansion,args.active_k,seed=seed)
        with torch.inference_mode():
            encoded=dg(keys)
        mem=AssociativeMemory(1000,args.expansion,1)
        mem.add(encoded,labels.float().unsqueeze(-1))
        seed_result={"seed":seed,"noise_levels":[]}
        for noise in args.noises:
            q=[]
            qlabels=[]
            # Independent partial cues: not copies of stored episodes.
            for t in range(100):
                for f in range(2):
                    q.append(F.normalize(topics[t]+0.30*facets[t,f]+noise*torch.randn(args.dim,generator=g),dim=-1))
                    qlabels.append(t*2+f)
            q=torch.stack(q); qlabels=torch.tensor(qlabels)
            with torch.inference_mode():
                raw_top=(q@keys.T).topk(10,dim=-1).indices
                qcodes=dg(q)
                _,_,dg_top=mem.retrieve(qcodes,10)
            seed_result["noise_levels"].append({
                "noise":noise,
                "raw_cosine":score(raw_top,labels,qlabels),
                "dg_ca3":score(dg_top.cpu(),labels,qlabels)
            })
        results.append(seed_result)

    # Scaling proxy: use 512-d keys and 64-d values so 100k entries fit
    # on ordinary CPU runners. This is data-structure scaling, not full DG cost.
    scaling=[]
    for capacity in args.scale_capacities:
        gen=torch.Generator().manual_seed(9000+capacity)
        key_dim,value_dim=512,64
        mem=AssociativeMemory(capacity,key_dim,value_dim)
        batch= min(capacity,10000)
        k=F.normalize(torch.randn(batch,key_dim,generator=gen),dim=-1)
        v=torch.randn(batch,value_dim,generator=gen)
        start=time.perf_counter(); mem.add(k,v); write_ms=(time.perf_counter()-start)*1000
        if mem.size < capacity:
            remain=capacity-mem.size
            for start_i in range(0,remain,10000):
                n=min(10000,remain-start_i)
                mem.add(F.normalize(torch.randn(n,key_dim,generator=gen),dim=-1),
                        torch.randn(n,value_dim,generator=gen))
        query=F.normalize(torch.randn(32,key_dim,generator=gen),dim=-1)
        samples=[]
        for _ in range(5):
            t0=time.perf_counter(); mem.retrieve(query,10); samples.append((time.perf_counter()-t0)*1000)
        scaling.append({
            "capacity":capacity,"key_dim":key_dim,"value_dim":value_dim,
            "estimated_tensor_storage_bytes":capacity*(key_dim+value_dim)*4+capacity,
            "initial_batch_write_ms":write_ms,
            "retrieval_batch_32_p50_ms":statistics.median(samples),
            "retrieval_batch_32_p95_ms":pctl(samples,.95),
            "storage_scope":"CPU float32 tensors + validity mask; excludes Python/runtime overhead"
        })
    report={
        "benchmark":"HCDM-110 harder independent-cue DG/CA3 benchmark v2",
        "seeds":args.seeds,"noise_levels":args.noises,
        "dataset":{"topics":100,"facets_per_topic":2,"stored_episodes_per_facet":5,
                   "queries_per_facet":1,"stored_records":1000,"queries_per_seed":200,
                   "query_is_independent_partial_cue":True,"exact_stored_query_reuse":False},
        "dg":{"input_dim":args.dim,"expansion_dim":args.expansion,"active_k":args.active_k},
        "results":results,"memory_scaling_proxy":scaling,
        "guardrails":["Synthetic results only; no language-model quality claim.",
                      "Each query is independently sampled from its topic/facet cue.",
                      "Scaling uses 512-d keys and 64-d values, not full 4096-d DG keys.",
                      "No production checkpoint loaded or modified."],
        "checkpoint_loaded":False,"production_model_modified":False
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
