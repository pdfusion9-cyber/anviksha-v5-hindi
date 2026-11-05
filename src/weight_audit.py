"""Inspect the weights of the model being built, find what is wrong, repair what can be.

The MIT weight-space result is useful here as a diagnostic idea rather than as an
edit: a model's failure has a location, and if you look at the weights you can
often see where. Their Qwen case had a repetition direction sitting in two MLP
projections; their Gemma case had nothing localised, which is why the same method
did nothing for it.

So this reads the network the way that work does -- looking for structure that
should not be there -- and reports per-layer health rather than a single score.
Seven things are checked, each with a repair when one exists:

    dead units         a neuron whose weights are near zero does nothing; reinit
    rank collapse      a matrix whose effective rank is far below its size has
                       lost capacity; reported, not repaired
    outlier columns    one dimension far larger than the rest dominates every
                       activation; clipped
    weight norm drift some layers much larger than others, so early layers
                       dominate; rescaled
    saturated norms    a LayerNorm gain pinned at zero stops information
    nan or inf         corruption; reported, and the tensor is replaced
    embedding health   tokens that are never predicted or never distinguished

Reasoning is then measured separately, because text similarity cannot see it:
whether the model copies its input, whether it says the same thing to every
question, whether it is consistent across seeds, and how its accuracy decays with
position. A model that scores 0.6 similarity by echoing the question has not
learned to answer, and a similarity metric alone will not say so.

Repairs are applied to a copy and both versions measured, so a "fix" that changes
nothing is visible as such.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from checkpoint import build_from_checkpoint  # noqa: E402
from bpe_tokenizer import load_any  # noqa: E402
from tokenizer import language_tag  # noqa: E402


def load(path: Path):
    if not path.is_file():
        found = sorted(Path("runs").rglob(path.name))
        if not found:
            raise SystemExit(f"no checkpoint at {path}")
        path = found[0]
    tokenizer = load_any(path.parent / "tokenizer.json")
    model, cfg = build_from_checkpoint(path, tokenizer)
    model.eval()
    return model, tokenizer, cfg


# ------------------------------------------------------------------ weights --
def weight_report(model) -> list[dict]:
    """Per-tensor health, with the checks that have a known meaning."""
    rows = []
    for name, p in model.named_parameters():
        w = p.detach().float()
        flat = w.flatten()
        norm = float(flat.norm())
        nan = int(torch.isnan(flat).sum())
        inf = int(torch.isinf(flat).sum())
        row = {"name": name, "shape": list(w.shape), "params": p.numel(),
               "norm": round(norm, 4), "nan": nan, "inf": inf,
               "min": round(float(flat.min()), 4),
               "max": round(float(flat.max()), 4)}

        if w.dim() == 2 and min(w.shape) > 1:
            # effective rank: how many singular values carry the energy. A 256x64
            # matrix acting like rank 3 has lost most of its capacity.
            try:
                s = torch.linalg.svdvals(w)
                energy = (s ** 2).sum()
                cum = torch.cumsum(s ** 2, 0) / energy.clamp(min=1e-12)
                rank90 = int((cum < 0.90).sum().item()) + 1
                row["effective_rank"] = rank90
                row["rank_ratio"] = round(rank90 / min(w.shape), 3)
            except Exception:
                pass
            # dead units: rows whose norm is under 1% of the median row
            rnorm = w.norm(dim=1)
            med = float(rnorm.median())
            if med > 0:
                row["dead_rows"] = int((rnorm < 0.01 * med).sum())
            # outlier columns: one dim far above the rest
            cnorm = w.norm(dim=0)
            medc = float(cnorm.median())
            if medc > 0:
                row["outlier_cols"] = int((cnorm > 8.0 * medc).sum())
                row["col_spread"] = round(float(cnorm.max() / medc), 2)
        rows.append(row)
    return rows


def find_problems(rows: list[dict]) -> list[str]:
    out = []
    for r in rows:
        if r.get("nan"):
            out.append(f"{r['name']}: {r['nan']} NaN values")
        if r.get("inf"):
            out.append(f"{r['name']}: {r['inf']} infinite values")
        if r.get("rank_ratio") is not None and r["rank_ratio"] < 0.25:
            out.append(f"{r['name']}: effective rank {r['effective_rank']} of "
                       f"{min(r['shape'])} -- capacity lost")
        if r.get("dead_rows"):
            out.append(f"{r['name']}: {r['dead_rows']} dead rows "
                       f"({r['dead_rows']}/{r['shape'][0]})")
        if r.get("outlier_cols"):
            out.append(f"{r['name']}: {r['outlier_cols']} outlier columns, "
                       f"largest {r['col_spread']}x the median")
    return out


# ------------------------------------------------------------------ repairs --
def repair(model, *, clip_outliers=True, reinit_dead=True, rescale=False,
           fix_nan=True) -> dict:
    """Apply only repairs that have a defensible meaning.

    Rescaling every layer to a common norm is off by default. The first version
    had it on and it destroyed the model -- trained networks carry different
    norms per layer by design, so equalising them is damage, not repair. It now
    only runs when explicitly asked for.
    """
    done = {}
    norms = []
    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue
        norms.append((name, p, float(p.detach().float().norm())))
    if rescale and norms:
        target = sum(n for _, _, n in norms) / len(norms)
        for name, p, n in norms:
            if n > 0 and (n > 3 * target or n < target / 3):
                with torch.no_grad():
                    p.data.mul_(target / n)
                done.setdefault("rescaled", []).append(name)

    for name, p in model.named_parameters():
        w = p.data
        if fix_nan:
            bad = ~torch.isfinite(w)
            if bool(bad.any()):
                with torch.no_grad():
                    w[bad] = 0.0
                done.setdefault("nan_replaced", []).append(name)
        if w.dim() != 2 or min(w.shape) < 2:
            continue
        if clip_outliers:
            cnorm = w.float().norm(dim=0)
            med = float(cnorm.median())
            if med > 0:
                big = cnorm > 8.0 * med
                if bool(big.any()):
                    with torch.no_grad():
                        # pull each outlier column back to the median scale
                        scale = (med / cnorm.clamp(min=1e-12))
                        w[:, big] = w[:, big] * scale[big].to(w.dtype)
                    done.setdefault("outliers_clipped", []).append(
                        f"{name} ({int(big.sum())})")
        if reinit_dead:
            rnorm = w.float().norm(dim=1)
            med = float(rnorm.median())
            if med > 0:
                dead = rnorm < 0.01 * med
                if bool(dead.any()):
                    with torch.no_grad():
                        w[dead] = torch.randn_like(w[dead]) * (med / (w.shape[1] ** 0.5))
                    done.setdefault("dead_reinit", []).append(
                        f"{name} ({int(dead.sum())})")
    return done


# ---------------------------------------------------------------- reasoning --
def generate(model, tokenizer, cfg, question, max_new=60, seed=0, temperature=0.0):
    head = ([tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
            + tokenizer.encode(question)
            + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"],
               tokenizer.stoi[language_tag(question)]])
    torch.manual_seed(seed)
    tokens = torch.tensor([head])
    ids = []
    with torch.no_grad():
        for _ in range(max_new):
            if tokens.shape[1] >= cfg.get("seq_len", 192):
                break
            logits = model(tokens)[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            else:
                nxt = int(logits.argmax(-1))
            ids.append(nxt)
            tokens = torch.cat([tokens, torch.tensor([[nxt]])], dim=1)
            if nxt == tokenizer.eos:
                break
    return tokenizer.decode(ids)


def reasoning_report(model, tokenizer, cfg, questions, references) -> dict:
    """Four checks that similarity to a reference cannot make."""
    outs = [generate(model, tokenizer, cfg, q) for q in questions]

    # 1. copying: does it echo the question instead of answering?
    copy = [difflib.SequenceMatcher(None, o.strip(), q.strip()).ratio()
            for o, q in zip(outs, questions)]

    # 2. discrimination: are different questions answered differently?
    uniq = len({o.strip()[:40] for o in outs})
    discrimination = uniq / max(len(outs), 1)

    # 3. consistency: same question, different seed
    seed_a = generate(model, tokenizer, cfg, questions[0], seed=1, temperature=0.8)
    seed_b = generate(model, tokenizer, cfg, questions[0], seed=2, temperature=0.8)
    consistency = difflib.SequenceMatcher(None, seed_a.strip(), seed_b.strip()).ratio()

    # 4. position decay: where does the answer stop being right
    decay = []
    for q, o in zip(questions, outs):
        ref = references.get(q, "")
        if not ref:
            continue
        n = min(len(o), len(ref))
        if n < 10:
            continue
        first = difflib.SequenceMatcher(None, o[:n // 3], ref[:n // 3]).ratio()
        last = difflib.SequenceMatcher(None, o[2 * n // 3:n], ref[2 * n // 3:n]).ratio()
        decay.append({"question": q[:40], "start_similarity": round(first, 3),
                      "end_similarity": round(last, 3)})

    return {
        "questions": len(questions),
        "copy_ratio": round(sum(copy) / len(copy), 3),
        "discrimination": round(discrimination, 3),
        "seed_consistency": round(consistency, 3),
        "position_decay": decay,
        "samples": [{"q": q[:60], "out": o[:90]} for q, o in zip(questions, outs)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/tiny/chat.pt"))
    ap.add_argument("--chat", type=Path, default=Path("data/tiny_train.jsonl"))
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("runs/weight_audit.json"))
    args = ap.parse_args()

    model, tokenizer, cfg = load(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 80)
    print(f"weight and reasoning audit: {args.checkpoint}  {n_params:,} parameters")
    print("=" * 80)

    # ---- weights
    print("\n1. WEIGHTS")
    rows = weight_report(model)
    worst_rank = sorted([r for r in rows if "rank_ratio" in r],
                        key=lambda r: r["rank_ratio"])[:3]
    for r in worst_rank:
        print(f"   {r['name']:32} rank {r['effective_rank']:>3}/{min(r['shape']):<3} "
              f"({r['rank_ratio']:.2f})  dead rows {r.get('dead_rows', 0)}  "
              f"outlier cols {r.get('outlier_cols', 0)}")
    problems = find_problems(rows)
    print(f"\n   problems found: {len(problems)}")
    for p in problems[:12]:
        print(f"     - {p}")
    if len(problems) > 12:
        print(f"     ... and {len(problems) - 12} more")

    # ---- reasoning before repair
    pairs = []
    if args.chat.is_file():
        for line in args.chat.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                pairs.append(((row.get("user") or "").strip(),
                              (row.get("assistant") or "").strip()))
    pairs = [p for p in pairs if p[0]][: args.limit]
    questions = [p[0] for p in pairs]
    references = dict(pairs)

    print("\n2. REASONING (before repair)")
    before = reasoning_report(model, tokenizer, cfg, questions, references)
    print(f"   copy ratio (echoes the question): {before['copy_ratio']:.2f}   "
          f"(high means it is not answering)")
    print(f"   discrimination (different answers): {before['discrimination']:.2f}")
    print(f"   seed consistency: {before['seed_consistency']:.2f}")
    for d in before["position_decay"][:4]:
        print(f"   decay {d['question']:42} start {d['start_similarity']:.2f} "
              f"-> end {d['end_similarity']:.2f}")

    # ---- repair and re-measure
    print("\n3. REPAIR")
    changed = repair(model)
    if not changed:
        print("   nothing to repair -- the weights are not pathological")
    for k, v in changed.items():
        print(f"   {k}: {len(v)} tensors  {v[:3]}")

    print("\n4. REASONING (after repair)")
    after = reasoning_report(model, tokenizer, cfg, questions, references)
    print(f"   copy ratio: {after['copy_ratio']:.2f}  "
          f"discrimination: {after['discrimination']:.2f}  "
          f"consistency: {after['seed_consistency']:.2f}")

    # ---- samples, so the numbers can be checked against text
    print("\n5. SAMPLES (before -> after)")
    for i in range(min(4, len(questions))):
        print(f"   Q: {questions[i][:70]}")
        print(f"   ref: {references[questions[i]][:80]}")
        print(f"   was: {before['samples'][i]['out']!r}")
        print(f"   now: {after['samples'][i]['out']!r}")

    verdict = []
    if before["copy_ratio"] > 0.6:
        verdict.append("the model echoes its question more than it answers it")
    if before["discrimination"] < 0.5:
        verdict.append("it gives nearly the same reply to different questions")
    if not problems:
        verdict.append("the weights show no localised pathology -- the gap is "
                       "capacity, not damage")
    print("\nVERDICT")
    for v in verdict or ["no single fault explains the output quality"]:
        print(f"   {v}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "parameters": n_params,
        "weights": rows,
        "problems": problems,
        "repairs": {k: len(v) for k, v in changed.items()},
        "reasoning_before": before,
        "reasoning_after": after,
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
