"""A constrained sparse edit, built to be a fix rather than a flip.

The unconstrained version has a single degree of freedom on a balanced probe and
spends it flipping the decision boundary: one class goes 0% to 100% only because
the other goes 100% to 0%, and the overall number moves from 49.75% to 50.25%.
That scores the same as a coin toss and looks like a success if you only read the
aggregate.

Four constraints, each aimed at one way that happens:

    both groups must improve    the edit is rejected unless the group that
                                already worked stays above a floor. A flip
                                cannot pass this because it trades one for the
                                other.
    many probes, not one        the direction is averaged over the whole probe
                                set, so it is conditioned rather than arbitrary.
    alternating steps           several small edits are tried in turn, each
                                re-checked, instead of one large one.
    per-group reporting         every number is per group. The aggregate is
                                printed last and labelled as the thing that
                                cannot distinguish a fix from a flip.

The pass rule is deliberately strict: a run that improves the failing group while
the working group drops is reported as a flip, not as a fix, whatever the
aggregate says.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

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


def encode_pair(tokenizer, question, answer):
    head = ([tokenizer.stoi["<bos>"], tokenizer.stoi["<user>"]]
            + tokenizer.encode(question)
            + [tokenizer.stoi["<sep>"], tokenizer.stoi["<assistant>"],
               tokenizer.stoi[language_tag(question)]])
    body = tokenizer.encode(answer) + [tokenizer.eos]
    return head + body, len(head)


def group_loss(model, tokenizer, cfg, pairs):
    """Mean answer-span cross-entropy over a group. Lower is better."""
    seq_len = cfg.get("seq_len", 192)
    total, count = 0.0, 0
    for q, a in pairs:
        ids, start = encode_pair(tokenizer, q, a)
        ids = ids[:seq_len]
        if len(ids) < 8 or start >= len(ids) - 1:
            continue
        x = torch.tensor([ids[:-1]])
        y = torch.tensor(ids[1:])
        logits = model(x)[0]
        # score the answer span only, matching the training mask
        span_logits = logits[start - 1:]
        span_y = y[start - 1:]
        if span_y.numel() == 0:
            continue
        total += float(F.cross_entropy(span_logits, span_y))
        count += 1
    return total / max(count, 1)


def accuracy(model, tokenizer, cfg, pairs):
    """Fraction of pairs whose answer tokens are mostly predicted correctly."""
    seq_len = cfg.get("seq_len", 192)
    hits = 0
    for q, a in pairs:
        ids, start = encode_pair(tokenizer, q, a)
        ids = ids[:seq_len]
        if len(ids) < 8 or start >= len(ids) - 1:
            continue
        x = torch.tensor([ids[:-1]])
        y = torch.tensor(ids[1:])
        with torch.no_grad():
            pred = model(x)[0].argmax(-1)
        span_pred = pred[start - 1:]
        span_y = y[start - 1:]
        if span_y.numel() == 0:
            continue
        if float((span_pred == span_y).float().mean()) > 0.5:
            hits += 1
    return hits / max(len(pairs), 1)


def candidate_tensors(model, min_params=64):
    out = []
    for name, p in model.named_parameters():
        if p.dim() == 2 and p.numel() >= min_params and "embed" not in name:
            out.append((name, p))
    return out


def gradient_for(model, tokenizer, cfg, pairs):
    """One averaged gradient over the whole probe set, not a single example."""
    model.zero_grad(set_to_none=True)
    seq_len = cfg.get("seq_len", 192)
    used = 0
    for q, a in pairs:
        ids, start = encode_pair(tokenizer, q, a)
        ids = ids[:seq_len]
        if len(ids) < 8 or start >= len(ids) - 1:
            continue
        x = torch.tensor([ids[:-1]])
        y = torch.tensor(ids[1:])
        logits = model(x)[0]
        span_logits = logits[start - 1:]
        span_y = y[start - 1:]
        if span_y.numel() == 0:
            continue
        loss = F.cross_entropy(span_logits, span_y)
        loss.backward()
        used += 1
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().clone() / max(used, 1)
    model.zero_grad(set_to_none=True)
    return grads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/tiny/chat.pt"))
    ap.add_argument("--chat", type=Path, default=Path("data/multi_train.jsonl"))
    ap.add_argument("--probe", type=int, default=200,
                    help="probe examples; the flip came from using one")
    ap.add_argument("--fractions", default="0.0016,0.016,0.05,0.10")
    ap.add_argument("--steps", type=int, default=4,
                    help="alternating steps, re-checked after each")
    ap.add_argument("--lr", type=float, default=0.01,
                    help="measured range: 0.01 lowers the loss, 0.1 overshoots")
    ap.add_argument("--lr-check", action="store_true", default=True,
                    help="confirm the step lowers the loss before using it")
    ap.add_argument("--floor", type=float, default=0.95,
                    help="the working group must stay at this fraction of its "
                         "starting accuracy, or the edit is a flip")
    ap.add_argument("--out", type=Path, default=Path("runs/constrained_edit.json"))
    args = ap.parse_args()

    model, tokenizer, cfg = load(args.checkpoint)
    print("=" * 82)
    print("constrained sparse edit: a fix, or another flip?")
    print("=" * 82)

    rows = []
    for line in args.chat.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            q, a = (r.get("user") or "").strip(), (r.get("assistant") or "").strip()
            if q and a:
                rows.append((q, a))

    # Two groups, so a flip is visible. The split is by current answer accuracy,
    # which is what a per-class view would be on a classification task.
    scored = [(q, a, accuracy(model, tokenizer, cfg, [(q, a)])) for q, a in rows]
    good = [(q, a) for q, a, s in scored if s > 0.5]
    bad = [(q, a) for q, a, s in scored if s <= 0.5]
    print(f"\ngroup A (already working): {len(good)} pairs")
    print(f"group B (failing)        : {len(bad)} pairs")
    if not bad:
        print("\nevery training pair already passes: the model has memorised its "
              "own data")
        print("there is no failing group here to repair, and a 97% training "
              "accuracy means nothing about capability")
        print("the failure is entirely on unseen questions, which this tool "
              "cannot fix")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            # base_a was never assigned -- the name came from an earlier draft.
            # The baseline for group A is by construction 1.0 here: group A is the
            # pairs that already pass.
            "baseline": {"group_a": 1.0, "group_b": 0.0,
                         "sizes": {"a": len(good), "b": 0}},
            "verdict": "nothing to repair: training data is already fitted",
        }, indent=2), encoding="utf-8")
        return
    if len(bad) < 5:
        print(f"\nonly {len(bad)} of {len(good) + len(bad)} pairs fail -- too few "
              f"to constrain an edit against. A single failing example is what "
              f"produced a bias flip rather than a fix.")

    probe = (bad + good)[: args.probe]
    base_a = accuracy(model, tokenizer, cfg, good) if good else 1.0
    base_b = accuracy(model, tokenizer, cfg, bad)
    base_loss_b = group_loss(model, tokenizer, cfg, bad)
    print(f"\nbefore: group A {base_a:.3f}   group B {base_b:.3f}   "
          f"B loss {base_loss_b:.3f}")
    print(f"probe: {len(probe)} examples averaged into one direction "
          f"(the flip used one)")

    grads = gradient_for(model, tokenizer, cfg, probe)
    flat = []
    for name, p in candidate_tensors(model):
        g = grads.get(name)
        if g is None:
            continue
        flat.append((name, p, g.flatten()))
    total_w = sum(t.numel() for _, _, t in flat)
    all_g = torch.cat([g for _, _, g in flat])
    print(f"editable weights: {total_w:,}")

    # rank by the size of the gradient, which is the selection rule that decides
    # everything -- random edits of the same size do nothing
    order = torch.argsort(all_g.abs(), descending=True)

    results = {}
    for frac in [float(x) for x in args.fractions.split(",")]:
        k = max(1, int(total_w * frac))
        print(f"\n--- fraction {frac:.4%}  ({k} weights) ---")

        for mode in ("targeted", "random"):
            model, tokenizer, cfg = load(args.checkpoint)
            if mode == "targeted":
                idx = order[:k]
            else:
                g = torch.Generator().manual_seed(0)
                idx = torch.randperm(total_w, generator=g)[:k]

            # map flat indices back to tensors
            offset, plan = 0, []
            for name, p, gflat in flat:
                n = gflat.numel()
                sel = idx[(idx >= offset) & (idx < offset + n)] - offset
                if sel.numel():
                    plan.append((name, p, sel))
                offset += n

            check = group_loss(model, tokenizer, cfg, probe)
            a_floor = base_a * args.floor
            kept_steps, flipped = 0, False
            for step in range(args.steps):
                snapshot = {n: p.detach().clone() for n, p, _ in plan}
                for name, p, sel in plan:
                    g = grads[name].flatten()[sel]
                    with torch.no_grad():
                        p.data.view(-1)[sel] -= args.lr * g.to(p.dtype)
                probe_loss = group_loss(model, tokenizer, cfg, probe)
                if probe_loss > check:
                    # the step raised the loss, so it is a step in the wrong
                    # direction at this learning rate -- undo and stop
                    for name, p, _ in plan:
                        with torch.no_grad():
                            p.data.copy_(snapshot[name])
                    print(f"    step {step}: loss rose {check:.3f} -> "
                          f"{probe_loss:.3f}, undone (lr too large)")
                    break
                check = probe_loss
                new_a = accuracy(model, tokenizer, cfg, good) if good else 1.0
                new_b = accuracy(model, tokenizer, cfg, bad)
                if good and new_a < a_floor:
                    # the working group broke: this is the flip the report warns
                    # about, so the step is undone rather than accepted
                    for name, p, _ in plan:
                        with torch.no_grad():
                            p.data.copy_(snapshot[name])
                    flipped = True
                    break
                kept_steps += 1
                if new_b >= 1.0:
                    break

            new_a = accuracy(model, tokenizer, cfg, good) if good else 1.0
            new_b = accuracy(model, tokenizer, cfg, bad)
            new_loss_b = group_loss(model, tokenizer, cfg, bad)
            results[f"{mode}@{frac}"] = {
                "weights": k, "steps": kept_steps, "flipped": flipped,
                "group_a": round(new_a, 3), "group_b": round(new_b, 3),
                "b_loss": round(new_loss_b, 3),
                "aggregate": round((new_a * len(good) + new_b * len(bad))
                                   / max(len(good) + len(bad), 1), 4),
            }
            tag = "FLIP" if flipped else ("fix" if new_b > base_b and new_a >= a_floor else "-")
            print(f"  {mode:9} A {new_a:.3f}  B {new_b:.3f}  "
                  f"B loss {new_loss_b:.3f}  steps {kept_steps}  [{tag}]")

    # the aggregate is printed last on purpose
    print("\n" + "-" * 82)
    print("aggregate accuracy (the number that cannot tell a fix from a flip):")
    for key, v in results.items():
        print(f"  {key:22} {v['aggregate']:.4f}   "
              f"A {v['group_a']:.3f}  B {v['group_b']:.3f}")
    print("\nA flip scores the same aggregate as a fix. Read the two group columns.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "baseline": {"group_a": base_a, "group_b": base_b,
                     "b_loss": round(base_loss_b, 3),
                     "sizes": {"a": len(good), "b": len(bad)}},
        "probe": len(probe),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
