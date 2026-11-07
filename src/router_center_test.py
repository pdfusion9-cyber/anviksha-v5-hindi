"""Does centring each copy's router block restore expert utilisation?

The forensics found that the five copies' router rows do not share a scale: in
layer 3 copy 4's rows average 2.77 against copy 3's 0.70, so copy 4 wins 87.5% of
tokens before the input is considered. That is a between-block offset, not a
within-block one.

Four variants, all on the same merged checkpoint and the same token sequence:

    none        as merged
    centre      subtract each block's own mean logit
    centre_sd   subtract the mean and divide by the block's standard deviation
    global      subtract one mean across all 110 rows (the control: if this works
                as well, the fault was a global offset and blocks are irrelevant)

Centre and global are deliberately close together. If centre wins and global does
not, the offset is per-block, which is what the forensics implies. If both win,
the diagnosis was wrong about where the offset lives.

Measured through the real forward pass, because centring changes which experts
fire and therefore changes every hidden state after it.
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


def gelu(z):
    return 0.5 * z * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def correct(logits, mode, per):
    """Apply the routing correction to raw router logits."""
    if mode == "none":
        return logits
    E = logits.shape[1]
    if mode == "global":
        return logits - logits.mean(axis=1, keepdims=True)
    out = logits.copy()
    for c in range(E // per):
        sl = slice(c * per, (c + 1) * per)
        blk = logits[:, sl]
        out[:, sl] = blk - blk.mean(axis=1, keepdims=True)
        if mode == "centre_sd":
            out[:, sl] /= (blk.std(axis=1, keepdims=True) + 1e-6)
    return out


def forward(ck, tokens, mode, per):
    cfg = ck.config
    n, dim, heads, layers = len(tokens), cfg["dim"], cfg["heads"], cfg["layers"]
    k, hd = cfg["top_k"], dim // cfg["heads"]
    x = ck.t("embed.weight")[tokens] + ck.t("position.weight")[:n]
    stats = []
    for L in range(layers):
        p = f"blocks.{L}"
        mu = x.mean(-1, keepdims=True); sd = np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        h = (x - mu) / sd * ck.t(f"{p}.attn.norm.weight") + ck.t(f"{p}.attn.norm.bias")
        qkv = h @ ck.t(f"{p}.attn.qkv.weight").T
        q, kk, v = np.split(qkv, 3, axis=-1)
        f = lambda t: t.reshape(n, heads, hd).transpose(1, 0, 2)
        q, kk, v = f(q), f(kk), f(v)
        s = (q @ kk.transpose(0, 2, 1)) / np.sqrt(hd)
        s = np.where(np.triu(np.ones((n, n), bool), 1)[None], -1e9, s)
        a = (softmax(s) @ v).transpose(1, 0, 2).reshape(n, dim)
        x = x + a @ ck.t(f"{p}.attn.proj.weight").T
        mu = x.mean(-1, keepdims=True); sd = np.sqrt(x.var(-1, keepdims=True) + 1e-5)
        g = (x - mu) / sd * ck.t(f"{p}.moe.norm.weight") + ck.t(f"{p}.moe.norm.bias")
        raw = g @ ck.t(f"{p}.moe.router.weight").T
        probs = softmax(correct(raw, mode, per))
        idx = np.argsort(-probs, axis=-1)[:, :k]
        counts = np.bincount(idx.ravel(), minlength=cfg["experts"])
        pn = counts[counts > 0] / counts.sum()
        stats.append({"used": int((counts > 0).sum()),
                      "top2": float(np.sort(counts)[-2:].sum() / counts.sum()),
                      "entropy": float(-(pn * np.log(pn)).sum())})
        w = np.take_along_axis(probs, idx, axis=-1)
        w = w / w.sum(-1, keepdims=True)
        o = np.zeros_like(g)
        for slot in range(k):
            for e in np.unique(idx[:, slot]):
                rows = np.where(idx[:, slot] == e)[0]
                w1 = ck.t(f"{p}.moe.experts.{int(e)}.w1.weight")
                w2 = ck.t(f"{p}.moe.experts.{int(e)}.w2.weight")
                o[rows] += w[rows, slot][:, None] * (gelu(g[rows] @ w1.T) @ w2.T)
        x = x + o
    return stats


def main():
    merged = Path(sys.argv[1])
    ck = Ckpt(merged)
    E = ck.config["experts"]
    per = E // 5
    tokens = list(range(1, 257))
    print(f"{E} experts, {ck.config['layers']} layers, {per} per block\n")

    out = {}
    print(f"{'variant':<10} {'used':>6} {'of':>4} {'pct':>6} {'top-2':>7} {'entropy':>8}")
    for mode in ("none", "centre", "centre_sd", "global"):
        st = forward(ck, tokens, mode, per)
        used = float(np.mean([s["used"] for s in st]))
        top2 = float(np.mean([s["top2"] for s in st]))
        ent = float(np.mean([s["entropy"] for s in st]))
        out[mode] = {"per_layer": st, "mean_used": used,
                     "mean_top2": top2, "mean_entropy": ent}
        print(f"{mode:<10} {used:>6.1f} {E:>4} {used/E*100:>5.0f}% "
              f"{top2*100:>6.1f}% {ent:>8.2f}")

    b = out["none"]["mean_used"]
    print()
    for mode in ("centre", "centre_sd", "global"):
        d = out[mode]["mean_used"]
        print(f"  {mode:<10} {b:.1f} -> {d:.1f}  ({d/b:.2f}x, "
              f"{(d-b)/E*100:+.0f} points of {E})")

    print("\nper layer, none against centre")
    print(f"{'layer':>5} {'none':>6} {'centre':>7} {'global':>7}")
    for L in range(ck.config["layers"]):
        print(f"{L:>5} {out['none']['per_layer'][L]['used']:>6} "
              f"{out['centre']['per_layer'][L]['used']:>7} "
              f"{out['global']['per_layer'][L]['used']:>7}")

    Path("router_center.json").write_text(json.dumps(out, indent=2))
    print("\nwritten router_center.json")


if __name__ == "__main__":
    main()
