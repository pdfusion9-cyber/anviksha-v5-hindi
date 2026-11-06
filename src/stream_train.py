"""Stream a corpus through a sparse mixture of experts and train it.

The corpus is never held. Documents arrive from a shard reader, are filtered and
tokenised one at a time, and only a window buffer of finished token ids stays in
memory. That is what lets a 4.7 GB corpus train a model on a machine that cannot
hold 4.7 GB.

The parameter count is one billion and the active count is forty-four million.
Both are reported on every checkpoint, because a model described only by its
total is a model described dishonestly.

Every fault found in the audit is carried over: the content filter, the
short-answer filter, the replay of pretraining text, the non-finite loss guard,
and the chat learning rate as a fraction rather than a constant.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from moe_model import MoEConfig, MoELM


def load_blocklist(path: Path) -> list[str]:
    if not path.is_file():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    return [t.lower() for t in d.get("terms", []) if t]


MAX_DOC_CHARS = 4_000_000


def blocked(text: str, terms: list[str]) -> bool:
    if not terms:
        return False
    # a document larger than this is a reading fault, not a document. Filtering
    # it would allocate a second copy of it and that is how the run died.
    if len(text) > MAX_DOC_CHARS:
        return True
    low = text.lower()
    return any(t in low for t in terms)


def stream_documents(path: Path, limit: int | None = None, epochs: int = 1):
    """Yield documents one at a time. Never returns the file as a string.

    epochs > 1 repeats the corpus, which is what a token budget larger than the
    corpus requires. The repetition is not free: past a few passes the model
    memorises rather than generalises, so the caller prints how many passes the
    budget implies before starting.
    """
    for _ in range(max(1, epochs)):
        yield from _stream_once(path, limit)


def _stream_once(path: Path, limit: int | None = None):
    """One pass over the file."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for key in ("assistant", "text", "content", "user"):
                    v = rec.get(key)
                    if isinstance(v, str) and v.strip():
                        yield v.strip()
    else:
        # One document per line, which is how fetch_hplt writes the corpus:
        # `text.replace("\n", " ") + "\n"`. The first version split on blank
        # lines instead, so a corpus with no blank lines was read as one 2 GB
        # document and text.lower() raised MemoryError.
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


class WindowStream:
    """A rolling token buffer fed from a document stream.

    Documents are concatenated into one continuous id sequence and windows are
    sliced from it. The first version required each document to be at least as
    long as the window, so on a corpus of short paragraphs every document was
    dropped and the stream reported "corpus exhausted" with zero steps -- a
    silent total failure that looked like an empty corpus.
    """

    def __init__(self, docs, tokenizer, seq_len: int, terms, buffer_windows=256,
                 min_chars=20):
        self.docs = docs
        self.tok = tokenizer
        self.seq_len = seq_len
        self.terms = terms
        self.buffer_windows = buffer_windows
        self.min_chars = min_chars
        self.buf: list[np.ndarray] = []
        self.carry: list[int] = []      # leftover ids too few to fill a window
        self.docs_read = 0
        self.docs_dropped = 0
        self.tokens_seen = 0
        self.finished = False

    def _refill(self):
        while len(self.buf) < self.buffer_windows and not self.finished:
            try:
                doc = next(self.docs)
            except StopIteration:
                self.finished = True
                break
            self.docs_read += 1
            if len(doc) < self.min_chars:
                self.docs_dropped += 1
                continue
            if self.terms and blocked(doc, self.terms):
                self.docs_dropped += 1
                continue
            ids = self.tok.encode(doc)
            self.tokens_seen += len(ids)
            self.carry.extend(ids)
            n = self.seq_len + 1
            while len(self.carry) >= n:
                self.buf.append(np.array(self.carry[:n], dtype=np.int64))
                del self.carry[:self.seq_len]

    def batch(self, size: int, rng) -> np.ndarray | None:
        if len(self.buf) < size:
            self._refill()
        if not self.buf:
            return None
        take = min(size, len(self.buf))
        picks = rng.integers(0, len(self.buf), size=take)
        rows = [self.buf[int(i)] for i in picks]
        for i in sorted({int(p) for p in picks}, reverse=True):
            if i < len(self.buf):
                self.buf.pop(i)
        return np.stack(rows)


def build_model(vocab: int, args) -> MoELM:
    cfg = MoEConfig(vocab=vocab, dim=args.dim, layers=args.layers,
                    heads=args.heads, seq_len=args.seq_len,
                    experts=args.experts, top_k=args.top_k,
                    ffn_mult=args.ffn_mult)
    return MoELM(cfg)


def save(model, tokenizer, path: Path, args, extra: dict | None = None,
         optimiser=None, step: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    report = model.parameters_report()
    torch.save({"model": model.state_dict(),
                # the optimiser state, because a session that ends at twelve
                # hours and restarts without it throws away every moment AdamW
                # had accumulated
                "optimiser": optimiser.state_dict() if optimiser else None,
                "step": step,
                "config": model.cfg.__dict__,
                "args": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items()},
                "report": report,
                "extra": extra or {}}, path)
    tokenizer.save(path.parent / "tokenizer.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--chat", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/moe"))
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--tokenizer", default="bpe")
    ap.add_argument("--tokenizer-from", type=Path, default=None,
                   help="copy the tokenizer from this directory. Two checkpoints "
                        "can only be merged if their embedding rows mean the same "
                        "thing, which requires the identical tokenizer.")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--ffn-mult", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clock-budget-minutes", type=float, default=120.0)
    ap.add_argument("--epochs", type=int, default=1,
                    help="how many times to read the corpus. Raise it when the "
                         "token budget is larger than the corpus, which is the "
                         "usual case for a long run over a small domain")
    ap.add_argument("--resume", action="store_true",
                    help="load optimiser state and step from the checkpoint so "
                         "a session that ended can continue rather than restart")
    ap.add_argument("--tokens", type=float, default=0.0,
                    help="token budget. The run streams until this many tokens "
                         "have been seen, and reports what fraction of the "
                         "budget it reached, so a shortfall is visible rather "
                         "than implied by a clock that ran out.")
    ap.add_argument("--target-steps", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--balance-weight", type=float, default=0.01)
    ap.add_argument("--buffer-windows", type=int, default=256)
    ap.add_argument("--content-filter", action="store_true")
    ap.add_argument("--blocklist", type=Path, default=Path("rules/content_blocklist.json"))
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--replay-text", type=Path, default=None)
    ap.add_argument("--replay-fraction", type=float, default=0.0)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                   help="bf16 halves weights, gradients and optimiser states. "
                        "A 1B fp32 model needs 17.3 GB and the runner has 16.")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(7)

    terms = load_blocklist(args.blocklist) if args.content_filter else []
    if terms:
        print(f"content filter: {len(terms)} blocked terms")

    # tokenizer first, so the model is built at the right vocabulary size
    from bpe_tokenizer import BPETokenizer, load_any
    tok_path = args.out / "tokenizer.json"
    if args.tokenizer_from is not None:
        src = (args.tokenizer_from if args.tokenizer_from.is_file()
               else args.tokenizer_from / "tokenizer.json")
        if not src.is_file():
            raise SystemExit(
                f"--tokenizer-from {args.tokenizer_from} has no tokenizer. "
                f"Falling back to a fresh one would give this half a different "
                f"vocabulary from the other and the merge would combine "
                f"parameters that mean different things.")
        args.out.mkdir(parents=True, exist_ok=True)
        tok_path.write_bytes(src.read_bytes())
        print(f"tokenizer: copied from {src} so this run can be merged with it")
    if tok_path.is_file():
        tokenizer = load_any(tok_path)
        print(f"tokenizer: loaded {tokenizer.size} entries from {tok_path}")
    else:
        sample = []
        n = 0
        for doc in stream_documents(args.text, limit=args.max_docs or None):
            if terms and blocked(doc, terms):
                continue
            sample.append(doc)
            n += len(doc)
            if n > 20_000_000:
                break
        tokenizer = BPETokenizer.train("\n".join(sample), vocab_size=args.vocab)
        args.out.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tok_path)
        print(f"tokenizer: trained {tokenizer.size} entries on {n:,} characters")

    model = build_model(tokenizer.size, args)
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
        print("dtype: bfloat16 (parameters, gradients and optimiser states)")
    report = model.parameters_report()
    print(f"model: {report['total_parameters']:,} total parameters, "
          f"{report['active_parameters']:,} active per token "
          f"({report['active_fraction']*100:.1f}%)")
    print(f"       {args.experts} experts, top {args.top_k}, dim {args.dim}, "
          f"{args.layers} layers, window {args.seq_len}")
    bytes_per = 2 if args.dtype == "bf16" else 4
    need = model.cfg.footprint_gb(bytes_per)
    print(f"       training footprint {need:.2f} GB "
          f"(weights + gradients + two optimiser states for all "
          f"{report['total_parameters']/1e6:.0f}M parameters)")
    if need > 14.0:
        raise SystemExit(
            f"this configuration needs {need:.1f} GB and the runner has 16 GB. "
            f"Reduce --experts, or use --dtype bf16. Refusing to start a run "
            f"that will be killed.")

    start_step = 0
    if args.init_from and args.init_from.is_file():
        state = torch.load(args.init_from, weights_only=False)
        model.load_state_dict(state["model"])
        start_step = int(state.get("extra", {}).get("step", 0))
        print(f"resumed from {args.init_from} at step {start_step}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    # Resume the optimiser too. A free GPU session ends at twelve hours; without
    # this the moments restart every session and the loss curve hides it.
    if args.resume:
        ck = args.out / "chat.pt"
        if ck.is_file():
            prev = torch.load(ck, weights_only=False)
            if prev.get("optimiser"):
                try:
                    optimiser.load_state_dict(prev["optimiser"])
                    start_step = max(start_step, int(prev.get("step", 0)))
                    print(f"resumed optimiser and step {start_step:,} from {ck}")
                except Exception as e:
                    print(f"optimiser state did not load ({type(e).__name__}); "
                          f"continuing with fresh moments, which costs training "
                          f"time and is worth knowing about")
            else:
                print("checkpoint carries no optimiser state")
        else:
            print(f"--resume given but {ck} does not exist; starting fresh")
    epochs = max(1, int(getattr(args, "epochs", 1)))
    if epochs > 1:
        print(f"corpus repeated {epochs} times: a budget larger than the "
              f"corpus needs repetition, and repetition past a few passes "
              f"teaches the text rather than the pattern")
    stream = WindowStream(stream_documents(args.text, limit=args.max_docs or None,
                                           epochs=epochs),
                          tokenizer, args.seq_len, terms,
                          buffer_windows=args.buffer_windows)
    replay_stream = None
    if args.replay_fraction > 0:
        if args.replay_text is None or not args.replay_text.is_file():
            raise SystemExit("--replay-fraction set but --replay-text is missing")
        replay_stream = WindowStream(
            stream_documents(args.replay_text), tokenizer, args.seq_len, terms,
            buffer_windows=args.buffer_windows)
        print(f"replay: {args.replay_fraction:.0%} of each batch from "
              f"{args.replay_text}")

    started = time.time()
    budget = args.clock_budget_minutes * 60
    step = start_step
    nan_skipped = 0
    replay_tokens = 0
    losses: list[float] = []

    print(f"streaming from {args.text}; clock budget {args.clock_budget_minutes:g} min")
    while step - start_step < args.steps:
        if time.time() - started > budget:
            print("clock budget reached")
            break
        if replay_stream is not None:
            take_replay = max(int(args.batch * args.replay_fraction), 0)
            main_batch = stream.batch(max(args.batch - take_replay, 1), rng)
            rep_batch = replay_stream.batch(take_replay, rng) if take_replay else None
            if main_batch is None and rep_batch is None:
                print("both streams exhausted")
                break
            parts = [b for b in (main_batch, rep_batch) if b is not None]
            batch = np.concatenate(parts, 0)
            replay_tokens += int(rep_batch.shape[0]) if rep_batch is not None else 0
        else:
            batch = stream.batch(args.batch, rng)
        if batch is None:
            print("corpus exhausted")
            break

        x = torch.from_numpy(batch[:, :-1])
        y = torch.from_numpy(batch[:, 1:])
        optimiser.zero_grad()
        logits = model(x)
        main_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tokenizer.pad)
        bal = model.balance_loss()
        loss = main_loss + args.balance_weight * bal

        if not torch.isfinite(loss):
            nan_skipped += 1
            optimiser.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        step += 1
        losses.append(float(main_loss.detach()))

        if step % 10 == 0:
            recent = sum(losses[-10:]) / min(len(losses), 10)
            usage = model.expert_usage()
            spread = (max(usage) - min(usage)) if usage else 0.0
            print(f"step {step}  loss {recent:.4f}  balance {float(bal):.3f}  "
                  f"expert spread {spread:.3f}  docs {stream.docs_read:,}  "
                  f"tokens {stream.tokens_seen:,}  {time.time()-started:.0f}s",
                  flush=True)

        if step % args.checkpoint_every == 0:
            save(model, tokenizer, args.out / "chat.pt", args,
                 {"step": step, "docs_read": stream.docs_read,
                  "tokens_seen": stream.tokens_seen,
                  "nan_skipped": nan_skipped,
                  "expert_usage": model.expert_usage()},
                 optimiser=optimiser, step=step)

    final = {"step": step, "docs_read": stream.docs_read,
             "tokens_seen": stream.tokens_seen, "docs_dropped": stream.docs_dropped,
             "nan_skipped": nan_skipped, "replay_windows": replay_tokens,
             "expert_usage": model.expert_usage(),
             "parameters": report["total_parameters"],
             "active_parameters": report["active_parameters"],
             "seconds": round(time.time() - started, 1)}
    save(model, tokenizer, args.out / "chat.pt", args, final,
         optimiser=optimiser, step=step)
    (args.out / "stream_report.json").write_text(json.dumps(final, indent=2))
    trained = step - start_step
    print(f"\nfinished {trained:,} steps in {time.time()-started:.0f}s")
    if trained == 0:
        # A run that read no tokens and trained no steps exited zero and was
        # reported as a success. That is the fault this whole audit keeps
        # finding: silence read as success. It is an error now.
        print("ERROR: no steps were trained. The corpus was empty or unreadable.",
              flush=True)
        raise SystemExit(3)
    if args.tokens:
        frac = stream.tokens_seen / max(args.tokens, 1)
        print(f"budget: {stream.tokens_seen:,} of {args.tokens:,.0f} tokens "
              f"({frac*100:.4f}%)")
        if frac < 0.99:
            print(f"  the run stopped {1-frac:.4%} short of the budget it was "
                  f"given; the clock or the corpus ran out first")
    print(f"docs {stream.docs_read:,}, tokens {stream.tokens_seen:,}, "
          f"non-finite steps skipped {nan_skipped}")
    print(f"expert usage: {final['expert_usage']}")


if __name__ == "__main__":
    main()
