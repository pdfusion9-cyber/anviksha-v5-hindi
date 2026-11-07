from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import torch
import torch.nn.functional as F
from hcdm110_prototype import DGPatternEncoder, LearnableDGEncoder

def metric(top, labels, qlabels):
    got=labels[top]
    relevant=got.eq(qlabels[:,None])
    same_topic=got.div(2,rounding_mode="floor").eq(qlabels.div(2,rounding_mode="floor")[:,None])
    wrong_facet=same_topic & ~relevant
    return {
      "precision_at_10":float(relevant.float().mean()),
      "recall_at_10":float(relevant.sum(-1).float().mean()/5.0),
      "top1_relevant_accuracy":float(relevant[:,0].float().mean()),
      "wrong_facet_at_10_rate":float(wrong_facet.float().mean()),
      "unrelated_topic_at_10_rate":float((~same_topic).float().mean())
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("reports/hcdm110_learnable_dg.json"))
    ap.add_argument("--seeds",type=int,nargs="+",default=[11,22,33,44,55])
    ap.add_argument("--noises",type=float,nargs="+",default=[0.05,0.10,0.20,0.30])
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--expansion",type=int,default=1024)
    ap.add_argument("--active-k",type=int,default=64)
    args=ap.parse_args()
    torch.set_num_threads(2)
    all_results=[]
    for seed in args.seeds:
      g=torch.Generator().manual_seed(seed)
      topics=F.normalize(torch.randn(100,384,generator=g),dim=-1)
      facets=F.normalize(torch.randn(100,2,384,generator=g),dim=-1)
      # Independent labelled training set; memory episodes and evaluation queries are separate draws.
      train_x=[]; train_y=[]
      for t in range(100):
       for f in range(2):
        for _ in range(12):
         z=torch.randn(384,generator=g)
         train_x.append(F.normalize(topics[t]+.30*facets[t,f]+.08*z,dim=-1)); train_y.append(t*2+f)
      train_x=torch.stack(train_x); train_y=torch.tensor(train_y)
      mem_x=[]; mem_y=[]
      for t in range(100):
       for f in range(2):
        for _ in range(5):
         z=torch.randn(384,generator=g)
         mem_x.append(F.normalize(topics[t]+.30*facets[t,f]+.025*z,dim=-1)); mem_y.append(t*2+f)
      mem_x=torch.stack(mem_x); mem_y=torch.tensor(mem_y)
      learned=LearnableDGEncoder(384,args.expansion,args.active_k,seed)
      head=torch.nn.Linear(args.expansion,200)
      opt=torch.optim.AdamW(list(learned.parameters())+list(head.parameters()),lr=2e-3,weight_decay=1e-4)
      started=time.perf_counter()
      learned.train()
      for epoch in range(args.epochs):
       order=torch.randperm(train_x.shape[0],generator=g)
       for ids in order.split(128):
        codes=learned(train_x[ids]); logits=head(codes)
        loss=F.cross_entropy(logits,train_y[ids])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(learned.parameters(),1.0); opt.step()
      train_seconds=time.perf_counter()-started
      learned.eval()
      with torch.no_grad():
       fixed=DGPatternEncoder(384,args.expansion,args.active_k,seed)
       fixed_mem=fixed(mem_x)
       learned_mem=learned(mem_x)
      per_seed={"seed":seed,"train_seconds":train_seconds,"epochs":args.epochs,"noise_levels":[]}
      for noise in args.noises:
       qx=[]; qy=[]
       for t in range(100):
        for f in range(2):
         qx.append(F.normalize(topics[t]+.30*facets[t,f]+noise*torch.randn(384,generator=g),dim=-1)); qy.append(t*2+f)
       qx=torch.stack(qx); qy=torch.tensor(qy)
       with torch.no_grad():
        raw=F.normalize(qx,dim=-1)@F.normalize(mem_x,dim=-1).T
        fixed_scores=F.normalize(fixed(qx),dim=-1)@F.normalize(fixed_mem,dim=-1).T
        learned_scores=F.normalize(learned(qx),dim=-1)@F.normalize(learned_mem,dim=-1).T
        # Rank fusion: equal-weight normalized cosine scores.
        hybrid=.5*raw+.5*learned_scores
        tops={"raw_cosine":raw.topk(10,dim=-1).indices,
              "fixed_dg":fixed_scores.topk(10,dim=-1).indices,
              "learned_dg":learned_scores.topk(10,dim=-1).indices,
              "hybrid":hybrid.topk(10,dim=-1).indices}
       per_seed["noise_levels"].append({"noise":noise,**{name:metric(ids,mem_y,qy) for name,ids in tops.items()}})
      all_results.append(per_seed)
    summary=[]
    for noise in args.noises:
      row={"noise":noise}
      for method in ("raw_cosine","fixed_dg","learned_dg","hybrid"):
       metrics=[m[method] for s in all_results for m in s["noise_levels"] if m["noise"]==noise]
       for key in metrics[0]:
        row[method+"_"+key]=sum(x[key] for x in metrics)/len(metrics)
      summary.append(row)
    report={"benchmark":"HCDM-110 learnable sparse DG controlled retrieval v1",
      "seeds":args.seeds,"noise_levels":args.noises,"epochs":args.epochs,
      "dataset":{"topics":100,"facets_per_topic":2,"training_examples_per_facet":12,
       "memory_episodes_per_facet":5,"independent_queries_per_facet":1,
       "train_memory_query_disjoint":True,"memory_records":1000,"queries_per_seed":200},
      "encoder":{"type":"trainable linear projection + ReLU top-k + L2 normalization",
       "expansion_dim":args.expansion,"active_k":args.active_k,"classifier_classes":200},
      "summary_mean_over_seeds":summary,"per_seed":all_results,
      "guardrails":["Synthetic labelled task only; not language-model quality evidence.",
       "Training examples, memory episodes and evaluation queries are independently sampled.",
       "No production checkpoint loaded or modified."],
      "checkpoint_loaded":False,"production_model_modified":False}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
