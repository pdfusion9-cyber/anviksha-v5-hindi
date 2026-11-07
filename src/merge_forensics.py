"""Merge forensics: why does the merged model route to fewer experts than a copy?

A single copy routes 15.1 of its 22 experts (69%). The merged model, after the
router-row bug was fixed, routes 28.9 of 110 (26%). So the bug was not the whole
story and something else is still lost.

The suspicion this tests: the merge averages the shared tensors across five
copies, but each copy's router rows were trained against that copy's own hidden
state. After averaging, the hidden state the router sees is a blend of five, and
each copy's rows are now reading inputs they were never trained on. If one copy's
rows happen to produce systematically larger logits under the blended state, that
copy wins every token and the others are never called.

Measured per layer, per copy block of 22 router rows:

    logit scale        mean and spread of h @ row for the same hidden states
    win rate           how often a row from this block is in the top 2
    expert output norm how large the experts themselves write into the residual
    weight norm        norm of the router rows themselves

If one block dominates on logit scale, that is the cause and the fix is to
re-scale each block, not to retrain.
"""
from __future__ import annotations

import io
import json
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np


class _R:
    __slots__ = ("key", "offset", "size")

    def __init__(self, k, o, n):
        self.key, self.offset, self.size = k, o, n


class U(pickle.Unpickler):
    def __init__(self, f):
        super().__init__(f, encoding="latin1")

    def find_class(self, m, n):
        if n == "OrderedDict":
            return OrderedDict
        if n == "_rebuild_tensor_v2":
            return lambda st, o, sz, *a, **k: _R(st, o, sz)
        return lambda *a, **k: None

    def persistent_load(self, pid):
        return pid[2] if isinstance(pid, tuple) and pid[:1] == ("storage",) else None


class Ckpt:
    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        root = self.z.namelist()[0].split("/")[0]
        self.root = root
        idx = U(io.BytesIO(self.z.read(f"{root}/data.pkl"))).load()
        self.model = idx["model"]
        self.config = idx["config"]
        self.merge = idx.get("merge")

    def t(self, name):
        r = self.model[name]
        flat = np.frombuffer(self.z.read(f"{self.root}/data/{r.key}"), dtype=np.float32)
        size = r.size if isinstance(r.size, (list, tuple)) else (r.size,)
        n = int(np.prod(size))
        return flat[int(r.offset):int(r.offset) + n].reshape([int(s) for s in size]).copy()


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def hidden_states(ck, tokens, layers):
    """Real hidden states at each layer's mixture input, from the real forward pass."""
    n, dim, heads = len(tokens), ck.config["dim"], ck.config["heads"]
    hd = dim // heads
    x = ck.t("embed.weight")[tokens] + ck.t("position.weight")[:n]
    out = []
    for L in range(layers):
        p = f"blocks.{L}"
        mu = x.mean(-1, keepdims=True); sd = np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        h = (x - mu) / sd * ck.t(f"{p}.attn.norm.weight") + ck.t(f"{p}.attn.norm.bias")
        qkv = h @ ck.t(f"{p}.attn.qkv.weight").T
        q, k, v = np.split(qkv, 3, axis=-1)
        f = lambda t: t.reshape(n, heads, hd).transpose(1, 0, 2)
        q, k, v = f(q), f(k), f(v)
        s = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)
        s = np.where(np.triu(np.ones((n, n), bool), 1)[None], -1e9, s)
        a = (softmax(s) @ v).transpose(1, 0, 2).reshape(n, dim)
        x = x + a @ ck.t(f"{p}.attn.proj.weight").T
        mu = x.mean(-1, keepdims=True); sd = np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        g = (x - mu) / sd * ck.t(f"{p}.moe.norm.weight") + ck.t(f"{p}.moe.norm.bias")
        out.append(g.copy())
        # advance through the mixture with the real router
        probs = softmax(g @ ck.t(f"{p}.moe.router.weight").T)
        idx = np.argsort(-probs, axis=-1)[:, : ck.config["top_k"]]
        w = np.take_along_axis(probs, idx, axis=-1)
        w = w / w.sum(-1, keepdims=True)
        o = np.zeros_like(g)
        for slot in range(ck.config["top_k"]):
            for e in np.unique(idx[:, slot]):
                rows = np.where(idx[:, slot] == e)[0]
                w1 = ck.t(f"{p}.moe.experts.{int(e)}.w1.weight")
                w2 = ck.t(f"{p}.moe.experts.{int(e)}.w2.weight")
                z = g[rows] @ w1.T
                z = 0.5 * z * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
                o[rows] += w[rows, slot][:, None] * (z @ w2.T)
        x = x + o
    return out


def main():
    merged_path = Path(sys.argv[1])
    copy_paths = [Path(p) for p in sys.argv[2:7]]

    m = Ckpt(merged_path)
    E, L, k = m.config["experts"], m.config["layers"], m.config["top_k"]
    per = E // 5
    print(f"merged: {E} experts, {L} layers, top_k {k}, {per} per copy block")
    print(f"merge record present: {m.merge is not None}")

    tokens = list(range(1, 257))
    hs = hidden_states(m, tokens, L)

    report = {"config": m.config, "per_block": per, "layers": []}
    print(f"\n{'layer':>5} {'block':>6} {'logit mean':>11} {'logit sd':>9} "
          f"{'wins':>6} {'share':>7} {'row norm':>9} {'expert out':>11}")
    for layer in range(L):
        g = hs[layer]
        router = m.t(f"blocks.{layer}.moe.router.weight")
        logits = g @ router.T
        probs = softmax(logits)
        idx = np.argsort(-probs, axis=-1)[:, :k]
        counts = np.bincount(idx.ravel(), minlength=E)
        row = {"layer": layer, "blocks": []}
        for c in range(5):
            sl = slice(c * per, (c + 1) * per)
            blk_logits = logits[:, sl]
            wins = int(counts[sl].sum())
            share = wins / counts.sum()
            rn = float(np.linalg.norm(router[sl], axis=1).mean())
            # how large do this block's experts write into the residual?
            outs = []
            for e in range(c * per, (c + 1) * per):
                w1 = m.t(f"blocks.{layer}.moe.experts.{e}.w1.weight")
                w2 = m.t(f"blocks.{layer}.moe.experts.{e}.w2.weight")
                outs.append(float(np.linalg.norm(w1) + np.linalg.norm(w2)))
            eo = float(np.mean(outs))
            print(f"{layer:>5} {c:>6} {blk_logits.mean():>11.4f} "
                  f"{blk_logits.std():>9.4f} {wins:>6} {share*100:>6.1f}% "
                  f"{rn:>9.4f} {eo:>11.1f}")
            row["blocks"].append({
                "copy": c, "logit_mean": float(blk_logits.mean()),
                "logit_sd": float(blk_logits.std()), "wins": wins,
                "share": float(share), "row_norm": rn, "expert_out_norm": eo})
        row["used"] = int((counts > 0).sum())
        report["layers"].append(row)

    used = [r["used"] for r in report["layers"]]
    print(f"\nmean experts used {np.mean(used):.1f} of {E} "
          f"({np.mean(used)/E*100:.0f}%)")
    report["mean_used"] = float(np.mean(used))

    # logit scale spread across blocks: the suspected cause
    spreads = []
    for r in report["layers"]:
        means = [b["logit_mean"] for b in r["blocks"]]
        spreads.append(max(means) - min(means))
    print(f"spread of block logit means, per layer: "
          f"{[round(x, 3) for x in spreads]}")
    print(f"mean spread {np.mean(spreads):.4f} nats")
    report["block_logit_spread"] = [float(x) for x in spreads]

    Path("merge_forensics.json").write_text(json.dumps(report, indent=2))
    print("\nwritten merge_forensics.json")


if __name__ == "__main__":
    main()
