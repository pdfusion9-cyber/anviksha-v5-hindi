"""Test whether direct weight editing can fix this model, as the MIT result did for Qwen.

The reported result is real: MIT CSAIL used a frontier model to edit Qwen3.5-0.8B's
weights directly, no gradients, and MATH-500 went from 7.9% to 48.1% by changing
0.016% of the network. What the video leaves out is the condition that made it
work, and the authors state it plainly in their own write-up:

    the edit removed an obstacle that hid an ability the model already had; it
    could not add an ability the model lacked

Their evidence for that limit is the part that matters here. They tried the same
method on Gemma 3 270M and it did nothing -- 6.8% to 7.7%. Gemma did not loop; it
made ordinary reasoning mistakes spread across the whole network, with no single
localised fault to subtract. Qwen's failure was specific: 93% of its answers ran
into the 32,768-token limit, repeating themselves until the budget was gone, and
that repetition had a direction in two layers' MLP down-projections that could be
found and removed.

So the technique applies when three things hold:

    1  the failure is a loop or a stuck direction, not a general lack of skill
    2  it is localised enough to be a small number of weights
    3  the ability being hidden is already present underneath

This script tests whether any of that is true for our model instead of assuming
it either way. It measures repetition, tries the edit at several strengths, and
reports what changed. A negative result here is the useful one: it says the
remaining gap is capacity, and no weight surgery will close it.

Edit applied: take the top singular direction of a layer's weight matrix,
subtract a fraction of the projection onto it, and re-measure. This is the same
family of operation the paper describes, done at a size we can run on a CPU.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter
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


def generate(model, tokenizer, cfg, question, max_new=60, seed=0):
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
            nxt = int(model(tokens)[:, -1, :].argmax(-1))
            ids.append(nxt)
            tokens = torch.cat([tokens, torch.tensor([[nxt]])], dim=1)
            if nxt == tokenizer.eos:
                break
    return ids, tokenizer.decode(ids)


def loop_metrics(ids):
    """How much the output repeats. The Qwen fault would score high here."""
    if not ids:
        return {"longest_run": 0, "most_common_share": 0.0, "distinct_share": 0.0}
    longest = cur = 1
    for a, b in zip(ids, ids[1:]):
        cur = cur + 1 if a == b else 1
        longest = max(longest, cur)
    counts = Counter(ids)
    return {
        "longest_run": longest,
        "most_common_share": round(counts.most_common(1)[0][1] / len(ids), 3),
        "distinct_share": round(len(counts) / len(ids), 3),
    }


def editable_matrices(model):
    """Every 2-D weight, so the edit can be aimed rather than guessed."""
    out = []
    for name, param in model.named_parameters():
        if param.dim() == 2 and param.numel() > 64:
            out.append((name, param))
    return out


def top_direction(param):
    """Leading singular direction of the weight matrix."""
    w = param.detach().float()
    try:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
    except Exception:
        return None, None
    return vh[0], s


def apply_edit(param, direction, strength, columns=None):
    """Subtract `strength` times the projection onto `direction`."""
    with torch.no_grad():
        w = param.data
        d = direction.to(w.dtype)
        proj = torch.outer(w @ d, d)          # component along d, per row
        if columns is not None:
            # only the rows whose loading is largest, as the paper does
            load = (w @ d).abs()
            idx = torch.topk(load, min(columns, load.numel())).indices
            mask = torch.zeros_like(proj)
            mask[idx, :] = 1.0
            proj = proj * mask
        param.data = w - strength * proj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/tiny/chat.pt"))
    ap.add_argument("--questions", default=(
        "How does TLS work in practice?|What is AES used for?|"
        "Explain OAuth briefly.|Why does DMARC matter?|What is SBOM?"))
    ap.add_argument("--strengths", default="0.0,0.25,0.5,1.0,2.0")
    ap.add_argument("--chat", type=Path, default=Path("data/tiny_train.jsonl"),
                    help="pairs to draw reference answers from")
    ap.add_argument("--out", type=Path, default=Path("runs/weight_edit.json"))
    args = ap.parse_args()

    questions = [q.strip() for q in args.questions.split("|") if q.strip()]

    # reference answers, so quality can be measured rather than guessed from
    # token diversity. A distinctness gain with no similarity gain is noise.
    references = {}
    if args.chat.is_file():
        for line in args.chat.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            q = (row.get("user") or "").strip()
            if q:
                references[q] = (row.get("assistant") or "").strip()
    print("=" * 78)
    print("can direct weight editing fix this model?")
    print("=" * 78)

    model, tokenizer, cfg = load(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint {args.checkpoint}  {n_params:,} parameters  "
          f"vocab {tokenizer.size}\n")

    # ---- 1. is this the Qwen failure or the Gemma one?
    print("baseline: is the output looping?")
    baseline = []
    for q in questions:
        ids, text = generate(model, tokenizer, cfg, q)
        m = loop_metrics(ids)
        baseline.append((q, ids, text, m))
        print(f"  longest run {m['longest_run']}, "
              f"most common {m['most_common_share']:.0%}, "
              f"distinct {m['distinct_share']:.0%}  |  {text[:48]!r}")
    avg_run = sum(b[3]["longest_run"] for b in baseline) / len(baseline)
    print(f"\n  mean longest identical run: {avg_run:.2f}")
    print("  the Qwen fault ran to the token limit repeating; a run near 2 means")
    print("  the model is not stuck, it is simply under-trained\n")

    # ---- 2. pick the layer the paper found most useful: the last MLP projection
    mats = editable_matrices(model)
    target = None
    for name, param in mats:
        if "mlp" in name and "proj" in name or "down" in name:
            target = (name, param)
    if target is None and mats:
        target = mats[-1]
    if target is None:
        print("no editable 2-D weight found")
        return
    name, param = target
    direction, spectrum = top_direction(param)
    if direction is None:
        print("svd failed")
        return
    print(f"editing {name}  shape {tuple(param.shape)}  "
          f"top singular value {float(spectrum[0]):.3f}\n")

    # ---- 3. try it at several strengths
    original = param.data.clone()
    results = {}
    print(f"{'strength':>9} {'mean run':>9} {'distinct':>9} {'similarity':>11}  sample")
    print("-" * 88)
    for s in [float(x) for x in args.strengths.split(",")]:
        param.data = original.clone()
        if s > 0:
            apply_edit(param, direction, s)
        runs, distincts, sims, sample = [], [], [], ""
        for q in questions:
            ids, text = generate(model, tokenizer, cfg, q)
            m = loop_metrics(ids)
            runs.append(m["longest_run"])
            distincts.append(m["distinct_share"])
            ref = references.get(q, "")
            if ref:
                sims.append(difflib.SequenceMatcher(None, text.strip(),
                                                    ref.strip()).ratio())
            if not sample:
                sample = text[:34]
        results[s] = {"mean_run": sum(runs) / len(runs),
                      "mean_distinct": sum(distincts) / len(distincts),
                      "mean_similarity": sum(sims) / len(sims) if sims else None,
                      "sample": sample}
        sim_txt = (f"{results[s]['mean_similarity']:.3f}"
                   if results[s]['mean_similarity'] is not None else "-")
        print(f"{s:>9.2f} {results[s]['mean_run']:>9.2f} "
              f"{results[s]['mean_distinct']:>9.3f} {sim_txt:>11}  {sample!r}")

    param.data = original

    # ---- 4. verdict
    base_sim = results[0.0].get("mean_similarity")
    best_s = max((s for s in results if s > 0),
                 key=lambda s: results[s].get("mean_similarity") or 0.0, default=None)
    gain = ((results[best_s].get("mean_similarity") or 0.0) - (base_sim or 0.0)
            if best_s is not None else 0.0)
    verdict = ("the edit changed nothing useful"
               if gain <= 0.02 else
               f"the edit raised similarity to the trained answer by {gain:.1%}")
    print(f"\nverdict: {verdict}")
    print("reading: the paper's own control is the key here. Gemma 3 270M went")
    print("6.8% -> 7.7% under the same treatment because its errors were spread")
    print("across the network with no localised fault. A model whose output is")
    print("mangled rather than stuck is in that second group, and the fix for it")
    print("is more training, not an edit.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "parameters": n_params,
        "edited": name,
        "baseline_mean_run": avg_run,
        "results": {str(k): v for k, v in results.items()},
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
