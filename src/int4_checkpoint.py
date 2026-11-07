"""Packed group-wise symmetric INT4 checkpoint storage (not INT4 arithmetic)."""
import argparse, hashlib, json, math, os, tempfile
from pathlib import Path
import torch

FORMAT = "kavach-moe-int4-v1"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def qtensor(x, group=128):
    if not x.is_floating_point(): raise TypeError("floating tensor required")
    dtype=str(x.dtype).split(".")[-1]
    flat=x.detach().cpu().float().contiguous().view(-1); n=flat.numel()
    ng=math.ceil(n/group) if n else 0
    pad=torch.zeros(ng*group); pad[:n]=flat
    m=pad.view(ng,group) if ng else pad.reshape(0,group)
    mx=m.abs().amax(1) if ng else torch.empty(0)
    scales=torch.where(mx>0,mx/7,torch.ones_like(mx))
    q=torch.round(m/scales[:,None]).clamp(-7,7).to(torch.int8).reshape(-1)[:n]
    nib=(q.to(torch.int16)&15).to(torch.uint8)
    if n%2: nib=torch.cat((nib,torch.zeros(1,dtype=torch.uint8)))
    packed=nib[::2] | (nib[1::2]<<4)
    err=(q.float()*scales.repeat_interleave(group)[:n]-flat) if n else flat
    return {"kind":"int4","packed":packed,"scales":scales,"shape":list(x.shape),"numel":n,"group_size":group,"dtype":dtype,"rmse":float(err.square().mean().sqrt()) if n else 0.0}

def qstate(state, group):
    out={}; aliases={}; summary={"tensor_count":0,"alias_count":0,"parameter_count":0,"packed_weight_bytes":0,"scale_bytes":0,"non_float_bytes":0,"worst_tensor_rmse":0.0}
    for name,t in state.items():
        if not isinstance(t,torch.Tensor): raise TypeError(f"{name} is not tensor")
        key=(t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()),str(t.dtype)) if t.device.type=="cpu" and t.layout==torch.strided else None
        if key is not None and key in aliases:
            out[name]={"kind":"alias","target":aliases[key]}; summary["alias_count"]+=1; continue
        if t.is_floating_point():
            item=qtensor(t,group); out[name]=item; summary["tensor_count"]+=1; summary["parameter_count"]+=t.numel()
            summary["packed_weight_bytes"]+=item["packed"].numel(); summary["scale_bytes"]+=item["scales"].numel()*4
            summary["worst_tensor_rmse"]=max(summary["worst_tensor_rmse"],item["rmse"])
            if key is not None: aliases[key]=name
        else:
            z=t.detach().cpu().contiguous(); out[name]={"kind":"raw","tensor":z}; summary["non_float_bytes"]+=z.numel()*z.element_size()
    summary["estimated_payload_bytes"]=summary["packed_weight_bytes"]+summary["scale_bytes"]+summary["non_float_bytes"]
    return out,summary

def dequantize(item):
    n=int(item["numel"]); g=int(item["group_size"]); p=item["packed"].byte().reshape(-1); s=item["scales"].float().reshape(-1)
    if p.numel()!=(n+1)//2 or s.numel()!=(math.ceil(n/g) if n else 0): raise ValueError("packed size/scale mismatch")
    if n==0: return torch.empty(item["shape"],dtype=torch.float32)
    nib=torch.stack((p&15,(p>>4)&15),1).reshape(-1)[:n].short()
    q=torch.where(nib>=8,nib-16,nib).float()
    y=q*s.repeat_interleave(g)[:n]
    if math.prod(item["shape"])!=n: raise ValueError("shape mismatch")
    return y.reshape(item["shape"])

def dequantize_state(state):
    out={}
    for k,v in state.items():
        kind=v["kind"]
        if kind=="int4": out[k]=dequantize(v).to(getattr(torch,v["dtype"],torch.float32))
        elif kind=="raw": out[k]=v["tensor"].clone()
        elif kind=="alias":
            if v["target"] not in out: raise ValueError("bad alias")
            out[k]=out[v["target"]]
        else: raise ValueError("unknown tensor kind")
    return out

def quantize(src,dst,tokenizer=None,group=128):
    src=Path(src); dst=Path(dst)
    if src.resolve()==dst.resolve(): raise ValueError("refusing source overwrite")
    ck=torch.load(src,map_location="cpu",weights_only=False,mmap=True)
    if not isinstance(ck,dict) or "model" not in ck or "config" not in ck: raise ValueError("expected {model, config} checkpoint")
    state,summary=qstate(ck["model"],group)
    meta={"format":FORMAT,"source_file":src.name,"source_sha256":sha256(src),"source_size_bytes":src.stat().st_size,"group_size":group,"summary":summary,"quantization":"symmetric signed INT4, group-wise, FP32 scales","runtime_note":"Storage only; existing MoE runtime dequantizes to FP32."}
    payload={"format":FORMAT,"config":ck["config"],"model":state,"metadata":meta}
    if isinstance(ck.get("sft"),dict): payload["sft"]=ck["sft"]
    if tokenizer:
        tp=Path(tokenizer); payload["tokenizer_json"]=tp.read_text(encoding="utf-8"); meta["tokenizer_sha256"]=sha256(tp)
    dst.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=dst.parent,prefix=dst.name); os.close(fd)
    try: torch.save(payload,tmp); os.replace(tmp,dst)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    report={"format":FORMAT,"source_size_bytes":src.stat().st_size,"output_size_bytes":dst.stat().st_size,"compression_ratio":round(src.stat().st_size/dst.stat().st_size,3),**summary,"source_sha256":meta["source_sha256"]}
    return report

def verify(path):
    p=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
    if p.get("format")!=FORMAT: raise ValueError("format mismatch")
    restored=dequantize_state(p["model"]); checked=0
    for k,t in restored.items():
        if t.is_floating_point() and not torch.isfinite(t).all(): raise ValueError("nonfinite: "+k)
        checked+=1
    return {"format":FORMAT,"tensor_keys":len(restored),"checked_keys":checked,"all_shapes_decoded":True,"finite_checked":True,"metadata":p["metadata"]}

def dequantize_checkpoint(src,dst):
    src=Path(src); dst=Path(dst)
    if src.resolve()==dst.resolve(): raise ValueError("refusing source overwrite")
    p=torch.load(src,map_location="cpu",weights_only=False,mmap=True)
    if p.get("format")!=FORMAT: raise ValueError("format mismatch")
    state=dequantize_state(p["model"])
    out={"model":state,"config":p["config"],"int4_source_metadata":p["metadata"]}
    if isinstance(p.get("sft"),dict): out["sft"]=p["sft"]
    dst.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=dst.parent,prefix=dst.name); os.close(fd)
    try:
        torch.save(out,tmp)
        os.replace(tmp,dst)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return {"source":str(src),"output":str(dst),"output_size_bytes":dst.stat().st_size,
            "tensor_keys":len(state),"format":"dequantized-fp32-from-int4"}

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("quantize"); q.add_argument("--input",required=True); q.add_argument("--output",required=True); q.add_argument("--tokenizer"); q.add_argument("--group-size",type=int,default=128)
    v=sub.add_parser("verify"); v.add_argument("--input",required=True)
    d=sub.add_parser("dequantize"); d.add_argument("--input",required=True); d.add_argument("--output",required=True)
    a=ap.parse_args()
    if a.cmd=="quantize": result=quantize(a.input,a.output,a.tokenizer,a.group_size)
    elif a.cmd=="verify": result=verify(a.input)
    else: result=dequantize_checkpoint(a.input,a.output)
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
