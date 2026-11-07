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
import hashlib
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
                 min_chars=20, state: dict | None = None):
        self.docs = docs
        self.tok = tokenizer
        self.seq_len = seq_len
        self.terms = terms
        self.buffer_windows = buffer_windows
        self.min_chars = min_chars
        self.buf: list[np.ndarray] = []
        self.carry: list[int] = []
        self.docs_read = 0
        self.docs_dropped = 0
        self.tokens_seen = 0
        self.finished = False
        self._skip_remaining = 0

        if state is not None:
            if int(state.get("seq_len", -1)) != self.seq_len:
                raise ValueError("saved stream state has a different sequence length")
            if int(state.get("buffer_windows", -1)) != self.buffer_windows:
                raise ValueError("saved stream state has a different buffer size")
            self.buf = [
                np.asarray(row, dtype=np.int64)
                for row in state.get("buffer", [])
            ]
            if any(row.shape != (self.seq_len + 1,) for row in self.buf):
                raise ValueError("saved stream buffer contains an invalid window")
            self.carry = [int(token) for token in state.get("carry", [])]
            self.docs_read = int(state.get("docs_read", 0))
            self.docs_dropped = int(state.get("docs_dropped", 0))
            self.tokens_seen = int(state.get("tokens_seen", 0))
            self.finished = bool(state.get("finished", False))
            # The buffered windows and carry are restored verbatim. Skip only
            # documents already consumed into those buffers before reading new
            # input from the same fingerprinted corpus.
            self._skip_remaining = 0 if self.finished else self.docs_read

    def state_dict(self) -> dict:
        return {
            "seq_len": self.seq_len,
            "buffer_windows": self.buffer_windows,
            "buffer": [row.tolist() for row in self.buf],
            "carry": list(self.carry),
            "docs_read": self.docs_read,
            "docs_dropped": self.docs_dropped,
            "tokens_seen": self.tokens_seen,
            "finished": self.finished,
        }

    def _refill(self):
        while len(self.buf) < self.buffer_windows and not self.finished:
            while self._skip_remaining:
                try:
                    next(self.docs)
                except StopIteration:
                    self._skip_remaining = 0
                    self.finished = True
                    break
                self._skip_remaining -= 1
            if self.finished:
                break
            try:
                doc = next(self.docs)
            except StopIteration:
                self.finished = True
                break
            self.docs_read += 1
            if len(doc) < self.min_chars:
                self.docs_dropped += 1
                continue
            if isinstance(doc, (list, tuple)):
                ids = list(doc)
            else:
                if self.terms and blocked(doc, self.terms):
                    self.docs_dropped += 1
                    continue
                ids = self.tok.encode(doc)

            # Every independent source document ends with EOS. This teaches
            # the model where one document stops instead of blending adjacent
            # documents into one uninterrupted stream.
            eos_id = getattr(self.tok, "eos", None)
            if eos_id is None:
                raise ValueError("tokenizer must expose an EOS token id")
            if not ids or int(ids[-1]) != int(eos_id):
                ids.append(int(eos_id))

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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(state: dict) -> int:
    top_level = state.get("step")
    extra_step = (state.get("extra") or {}).get("step")
    if top_level is not None and extra_step is not None:
        if int(top_level) != int(extra_step):
            raise ValueError(
                f"checkpoint step fields disagree: top-level={top_level}, "
                f"extra.step={extra_step}"
            )
    value = top_level if top_level is not None else extra_step
    return int(value or 0)


def restore_training_checkpoint(
    payload: dict,
    model: MoELM,
    optimiser: torch.optim.Optimizer,
    require_optimiser_state: bool,
) -> int:
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("checkpoint does not contain a model state")
    saved_config = payload.get("config") or {}
    model_config = getattr(model, "cfg", None)
    expected_config = vars(model_config) if model_config is not None else {}
    for key in (
        "vocab", "dim", "layers", "heads", "seq_len", "experts",
        "top_k", "ffn_mult",
    ):
        if key in saved_config and saved_config[key] != expected_config[key]:
            raise ValueError(
                f"checkpoint configuration mismatch for {key}: "
                f"{saved_config[key]} != {expected_config[key]}"
            )
    model.load_state_dict(payload["model"], strict=True)
    step = checkpoint_step(payload)

    optimiser_state = payload.get("optimiser")
    if require_optimiser_state and not isinstance(optimiser_state, dict):
        raise ValueError(
            "resume checkpoint has no optimiser state; refusing to silently "
            "restart AdamW moments"
        )
    if optimiser_state is not None:
        try:
            optimiser.load_state_dict(optimiser_state)
        except Exception as exc:
            raise ValueError(
                f"optimiser state could not be restored: {type(exc).__name__}: {exc}"
            ) from exc
    return step


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
    ap.add_argument("--pretokenized", action="store_true",
                    help="the corpus holds space-separated token ids, one "
                         "sequence per line, not text. Used for vocabulary "
                         "warm-up, where the ids are the lesson")
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
                   help="bf16 halves weights and gradients; factored Adafactor keeps small FP32 moments.")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"],
                   help="Adafactor is required for low-memory merged-model continuation.")
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
    if args.optimizer == "adafactor":
        from factorized_adafactor import FactorizedAdafactor, estimate_state_bytes
        optimiser_state_bytes = estimate_state_bytes(model.parameters())
        estimated = report["total_parameters"] * bytes_per * 2 + optimiser_state_bytes
        print(f"       factored optimizer state estimate {optimiser_state_bytes / 1e9:.3f} GB")
        print(f"       weights + gradients + factored state {estimated / 1e9:.2f} GB")
        if estimated > 13.5e9:
            raise SystemExit(
                f"estimated training tensors {estimated / 1e9:.2f} GB exceed the 13.5 GB safety cap")
    else:
        need = model.cfg.footprint_gb(bytes_per)
        print(f"       AdamW training footprint {need:.2f} GB "
              f"(weights + gradients + two optimiser states)")
        if need > 14.0:
            raise SystemExit(
                f"this configuration needs {need:.1f} GB and the runner has 16 GB. "
                f"Reduce --experts, or use --dtype bf16. Refusing to start a run "
                f"that will be killed.")

    if args.optimizer == "adafactor":
        from factorized_adafactor import FactorizedAdafactor
        optimiser = FactorizedAdafactor(
            model.parameters(), lr=args.lr, weight_decay=0.0,
            chunk_rows=64,
        )
    else:
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.95), weight_decay=0.1)
    start_step = 0
    resume_payload = None
    resume_path = None

    if args.resume:
        output_checkpoint = args.out / "chat.pt"
        if output_checkpoint.is_file():
            resume_path = output_checkpoint
        elif args.init_from is not None and args.init_from.is_file():
            resume_path = args.init_from
        else:
            raise SystemExit(
                "--resume requires either --out/chat.pt or an existing --init-from checkpoint"
            )
        sidecar_tokenizer = resume_path.parent / "tokenizer.json"
        if sidecar_tokenizer.is_file() and sidecar_tokenizer.read_bytes() != tok_path.read_bytes():
            raise SystemExit(
                f"checkpoint tokenizer does not match the active tokenizer: {sidecar_tokenizer}"
            )
        resume_payload = torch.load(
            resume_path, map_location="cpu", mmap=True, weights_only=False
        )
        try:
            start_step = restore_training_checkpoint(
                resume_payload, model, optimiser, require_optimiser_state=True
            )
        except Exception as exc:
            raise SystemExit(
                f"refusing unsafe resume from {resume_path}: {type(exc).__name__}: {exc}"
            ) from exc
        print(
            f"resumed model + AdamW state + global step {start_step:,} "
            f"from {resume_path}"
        )
    elif args.init_from is not None and args.init_from.is_file():
        resume_path = args.init_from
        sidecar_tokenizer = resume_path.parent / "tokenizer.json"
        if sidecar_tokenizer.is_file() and sidecar_tokenizer.read_bytes() != tok_path.read_bytes():
            raise SystemExit(
                f"checkpoint tokenizer does not match the active tokenizer: {sidecar_tokenizer}"
            )
        resume_payload = torch.load(
            resume_path, map_location="cpu", mmap=True, weights_only=False
        )
        try:
            start_step = restore_training_checkpoint(
                resume_payload, model, optimiser, require_optimiser_state=False
            )
        except Exception as exc:
            raise SystemExit(
                f"refusing incompatible initialization from {resume_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        print(f"initialized weights from {resume_path} at global step {start_step:,}")
    epochs = max(1, int(getattr(args, "epochs", 1)))
    if epochs > 1:
        print(f"corpus repeated {epochs} times: a budget larger than the "
              f"corpus needs repetition, and repetition past a few passes "
              f"teaches the text rather than the pattern")
    def pretokenized_documents(path, limit=None, epochs=1):
        for _ in range(max(1, epochs)):
            with path.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        return
                    line = line.strip()
                    if line:
                        yield [int(x) for x in line.split()]

    source = (pretokenized_documents(args.text, limit=args.max_docs or None,
                                     epochs=epochs)
              if args.pretokenized
              else stream_documents(args.text, limit=args.max_docs or None,
                                    epochs=epochs))
    corpus_sha256 = sha256_file(args.text)
    resume_extra = (resume_payload.get("extra", {}) if resume_payload else {}) or {}
    saved_stream_state = resume_extra.get("window_stream")
    saved_corpus_sha256 = resume_extra.get("corpus_sha256")
    if args.resume and (saved_stream_state is None) != (saved_corpus_sha256 is None):
        raise SystemExit(
            "resume checkpoint has incomplete stream metadata; refusing a partial data resume"
        )

    rng = np.random.default_rng(7)
    if args.resume and saved_stream_state is not None:
        if saved_corpus_sha256 != corpus_sha256:
            raise SystemExit(
                "training corpus fingerprint changed since the checkpoint; "
                "refusing to resume on a different document stream"
            )
        if int(resume_extra.get("stream_epochs", -1)) != epochs:
            raise SystemExit(
                "training epoch count changed since the checkpoint; refusing an inexact resume"
            )
        if "rng_state" not in resume_extra:
            raise SystemExit("resume checkpoint is missing the sampler RNG state")
        rng.bit_generator.state = resume_extra["rng_state"]
        stream = WindowStream(
            source, tokenizer, args.seq_len, terms,
            buffer_windows=args.buffer_windows, state=saved_stream_state
        )
        resume_policy = "exact_stream_and_optimizer_resume"
    else:
        stream = WindowStream(
            source, tokenizer, args.seq_len, terms,
            buffer_windows=args.buffer_windows
        )
        resume_policy = (
            "legacy_checkpoint_weights_optimizer_resumed_data_restarts_at_corpus_start"
            if args.resume else "fresh_stream"
        )

    replay_stream = None
    replay_corpus_sha256 = None
    if args.replay_fraction > 0:
        if args.replay_text is None or not args.replay_text.is_file():
            raise SystemExit("--replay-fraction set but --replay-text is missing")
        replay_corpus_sha256 = sha256_file(args.replay_text)
        saved_replay_state = resume_extra.get("replay_stream")
        saved_replay_sha = resume_extra.get("replay_corpus_sha256")
        if args.resume and saved_replay_state is not None:
            if saved_replay_sha != replay_corpus_sha256:
                raise SystemExit("replay corpus fingerprint changed since checkpoint")
            replay_stream = WindowStream(
                stream_documents(args.replay_text), tokenizer, args.seq_len, terms,
                buffer_windows=args.buffer_windows, state=saved_replay_state
            )
        elif args.resume and resume_extra.get("replay_fraction", 0):
            raise SystemExit("resume checkpoint has no replay stream state")
        else:
            replay_stream = WindowStream(
                stream_documents(args.replay_text), tokenizer, args.seq_len, terms,
                buffer_windows=args.buffer_windows
            )
        print(f"replay: {args.replay_fraction:.0%} of each batch from "
              f"{args.replay_text}")

    def checkpoint_extra():
        extra = {
            "step": step,
            "docs_read": stream.docs_read,
            "tokens_seen": stream.tokens_seen,
            "nan_skipped": nan_skipped,
            "expert_usage": model.expert_usage(),
            "window_stream": stream.state_dict(),
            "corpus_sha256": corpus_sha256,
            "stream_epochs": epochs,
            "rng_state": rng.bit_generator.state,
            "resume_policy": resume_policy,
            "replay_fraction": args.replay_fraction,
        }
        if replay_stream is not None:
            extra["replay_stream"] = replay_stream.state_dict()
            extra["replay_corpus_sha256"] = replay_corpus_sha256
        return extra

    started = time.time()
    budget = args.clock_budget_minutes * 60
    step = start_step
    nan_skipped = 0
    replay_tokens = 0
    losses: list[float] = []

    print(f"streaming from {args.text}; clock budget {args.clock_budget_minutes:g} min")
    while True:
        if args.target_steps > 0:
            if step >= args.target_steps:
                print(f"absolute target reached: {step:,}/{args.target_steps:,}")
                break
        elif step - start_step >= args.steps:
            break
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
                 checkpoint_extra(), optimiser=optimiser, step=step)

    final = checkpoint_extra()
    final.update({
        "docs_dropped": stream.docs_dropped,
        "replay_windows": replay_tokens,
        "parameters": report["total_parameters"],
        "active_parameters": report["active_parameters"],
        "seconds": round(time.time() - started, 1),
        "corpus_path": str(args.text),
    })
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
