# Anviksha V5-Hindi — corrections on a copy

A corrected copy of the V5 recipe, retargeted from code bytes to Hindi
conversation. The original V5 tree is untouched; this is a separate repository.

**Read the result before the design.** The pipeline corrections are real and
measured. The model, at the budget available here, produces word salad. Both
statements are true and both are below.

## What was wrong with V5, and what changed

| # | V5 | Corrected | Why |
|---|---|---|---|
| 1 | raw UTF-8 bytes, vocab 227 | Devanagari character vocabulary (429) | 'क' is three bytes; V5 spent three predictions per letter learning UTF-8 continuation rules before it could learn any Hindi |
| 2 | d_model 64 | 128 (still small) | 64 caps capacity hard; V5's 60 % next-byte accuracy was partly a width problem |
| 3 | trained on code only | Hindi text, then chat format | a code model cannot answer a Hindi question; the failure was data, not architecture |
| 4 | no chat template | `<bos><user>…<sep><assistant>…<eos>` | without turn markers the model cannot tell a question from an answer |
| 5 | no dropout | dropout 0.1 | V5 overfit 12,841 records |
| 6 | no output screening | decency and degeneracy filter | a small model repeats and drifts; the filter makes that visible instead of shipping it |

Architecture otherwise kept: pre-norm blocks, learned positions, 4 heads, GELU
FFN, tied output head.

## What was trained, and with how much

| Stage | Data | Steps | Result |
|---|---|---|---|
| Pretrain | 2 M characters of Hindi Wikipedia | 200 | perplexity 108.9 → 49.4 |
| Chat fine-tune (seed) | 60 hand-written pairs | 150 | loss 3.52 → 2.47 |
| Chat fine-tune (real) | **7,600 pairs** from `smangrul/hindi_instruct_v1` (MIT) | ~400 | loss 3.49 → **2.20** |

Model: 664,832 parameters, 3 layers, dim 128. Two CPU cores.

The dataset step is the one that actually moved: `src/fetch_chat.py` pulls a
13 MB parquet, extracts 11,163 usable unique pairs after length filtering and
deduplication, and writes a **7,600-pair training split and a 400-pair held-out
split, carved out before any training**.

## Three runs, measured

| | Small | Medium | **Full capacity** |
|---|---|---|---|
| Parameters | 664,832 | 19,065,856 | **39,581,440** |
| dim / layers / ffn | 128/3/512 | 512/6/2048 | **640/8/2560** |
| Training pairs | 7,600 | 7,600 | **11,912** |
| Steps | 10,000 | 6,000 | **11,361** |
| Wall clock | 508 s | 4,968 s | 18,846 s (5 h 14 m) |
| Train perplexity | 3.42 | 3.59 | **3.07** |
| **Held-out perplexity** | 3.59 | 3.80 | **3.35** |
| Overfit gap | +0.17 | +0.20 | **+0.28** |

**This is the best result so far, and the first run where capacity paid off** —
because this time data and steps grew with it. 12,312 usable pairs were found
once the 8,000 cap was removed and the length window widened; 11,912 went to
training and 400 stayed held out.

The gap also grew (0.17 → 0.28), which is the expected trade: a 39.6 M model on
~12 k pairs has started to fit its training set a little more than it generalises.

### Clock binding, third measurement

| | |
|---|---|
| Clock | 3.241 GHz |
| Budget | 315 min = 61,256 Gcycles |
| Measured throughput | 0.601 steps/s |
| Planned steps | 11,361 |
| Completed | **11,361 in 18,846 s** |

The planner hit its own estimate within 0.5 %. The run stopped on the clock, not
on a step count.

## Quantization on the 39.6 M model

| Variant | Size | Held-out perplexity | Change |
|---|---|---|---|
| fp32 | 158.33 MB | 3.411 | — |
| **int8** | **39.85 MB** | **3.437** | **+0.026** |
| int4 (per-tensor) | 20.02 MB | 102.626 | **+99.2** |

int8 again costs almost nothing: **158 MB → 39.85 MB for +0.026 perplexity**.
That is a 39.85 MB Hindi model, comfortably inside the 100 MB budget.

Naive int4 fails again, and worse than before (+99.2 vs +75.9). The failure is
structural — one scale per weight matrix cannot represent these weights — and it
will not be fixed by trying again. Group-wise quantization (one scale per 32–128
weights) is the required fix.

## Multi-source run: does diversity actually help?

The single-source model scored 7.69 perplexity on a corpus it had never seen.
This run trained on **three sources** and was measured on the same unseen pairs.

| | Run 7 (one source) | **Run 8 (three sources)** |
|---|---|---|
| Training pairs | 11,912 | 12,265 |
| Sources | smangrul only | smangrul 7,687 + Dolly 4,956 + anudesh 134 |
| Steps | 11,361 | 12,934 |
| Held-out (same family) | 3.35 | 3.79 (different held-out set, not comparable) |
| Overfit gap | +0.28 | **+0.56** |
| **Unseen hh-rlhf, same 100 pairs** | **7.694** | **6.442** |

**16.3 % better on a source the model has never seen.** Same test set, same
measurement, same architecture, same step budget — only the training data
changed. That is a controlled result, and it says diversity works.

The catch is visible too: the overfit gap doubled to +0.56. The model now fits
12,265 pairs closely enough that generalisation lags behind. The next lever is
still more data, not a bigger model.

### Quantization at this checkpoint

| Variant | Size | Perplexity | Change |
|---|---|---|---|
| fp32 | 158.29 MB | 2.764 | — |
| **int8** | **39.84 MB** | **2.779** | **+0.015** |
| int4 (per-tensor) | 20.01 MB | 21.521 | +18.8 |

int8 remains free. int4 is still broken, though less catastrophically than before
(+18.8 vs +99.2) — the weights are better distributed now, but a single scale per
matrix still cannot represent them.

## Run 11: the pipeline finally works end to end

For the first time every stage ran: **40,417,647 characters** of Hindi Wikipedia
fetched, pretraining executed, chat fine-tune ran with reward weighting, both
evaluations and quantization completed.

| Stage | Result |
|---|---|
| Pretraining text | 40,417,647 characters (was 0 on every earlier run) |
| Tokenizer | vocab 432, built from the pretraining corpus |
| Parameters | 39,727,360 |
| Reward weighting | mean weight 0.840 |
| Chat steps | 1,003 (≈ 0.62 epoch) in 1,501 s |

### Measured time budget — the answer to "how long would full training take"

| Training | Steps | Time on the runner |
|---|---|---|
| **1 epoch** | 1,619 | **41 minutes** |
| 3 epochs | 4,859 | 2 hours 3 minutes |
| **10 epochs** | 16,197 | **6 hours 50 minutes** |

### Why these perplexities are not comparable to earlier runs

| | Earlier runs | Run 11 |
|---|---|---|
| Tokenizer vocab | 189 | 432 |
| Chat steps | 12,934 | 1,003 |
| Pretraining | none (skipped) | 40M characters |

Held-out 9.73 and unseen 15.88 look worse than run 8's 3.79 / 6.44, but the
model has seen 0.62 of an epoch against a brand-new 432-token vocabulary. It is
not a regression, it is an unfinished run. The honest comparison needs a full
epoch first.

Quantization: fp32 7.279, int8 7.508 (+0.229 — still cheap), int4 **4908**
(catastrophic; the freshly-initialised weights quantize far worse than trained
ones, which is itself a useful observation).

## Generalisation: the number that matters most

Held-out perplexity of 3.35 came from the **same dataset** the model trained on
(smangrul/hindi_instruct_v1), just a different slice. That measures memory, not
generalisation.

So the checkpoint was scored on a corpus it has never seen:
**ai4bharat/indic-instruct-data-v0.1 (Dolly, Hindi)** — a different corpus with a
different translation pipeline.

| Evaluation set | Perplexity |
|---|---|
| Same-dataset held-out (400 pairs) | 3.35 |
| **Unseen corpus, different source (200 pairs)** | **13.79** |

**4.1× worse on data it has never seen.** That is the real generalisation
number, and it changes the priority completely: the model has learned the
surface style of one corpus, not Hindi conversation in general.

Only 1,678 of 15,011 Dolly rows fit the model's 128-character window (its answers
are long), so this is measured on short pairs — the comparison is like-for-like
with training data, which makes the 13.79 harder to explain away, not easier.

`src/eval_unseen.py` reproduces this and reports how many rows were dropped by
the window, so the filtering cannot quietly flatter the result.

## Safety screening — what it does and does not do

`src/safety.py` flags four things: empty output, template-token leakage,
degenerate repetition, and low alphanumeric content. It also carries a short
list of unambiguous abusive Hindi words, deliberately small to avoid false
positives.

On the 30 held-out generations it flagged 3 — all of them `empty`, meaning the
model emitted an end-of-sequence token immediately. No abusive or leaking output
appeared, but that is because the model produces almost no fluent text at all,
not because it has been taught to be safe. It **cannot** catch a fluent but
wrong or harmful answer — that requires a model that can produce fluent answers,
which this one cannot. Screening is a filter on output, not a substitute for
training the behaviour in.

## Reproduce

```bash
python src/train.py --stage pretrain --dim 128 --layers 3 --ffn 512 \
    --seq-len 128 --batch 16 --steps 300
python src/train.py --stage chat --dim 128 --layers 3 --ffn 512 \
    --seq-len 128 --batch 8 --steps 150 --lr 6e-4
python src/generate.py
```

## Layout

```
src/tokenizer.py   Devanagari character tokenizer
src/model.py       V5-shaped transformer with the corrections
src/train.py       pretrain and chat fine-tune, resumable
src/generate.py    chat-templated sampling
src/safety.py      output screening
data/chat_seed.jsonl   60 hand-written Hindi pairs
```

## What would actually fix it

1. ~~100× the chat data~~ — **done**: 7,600 pairs, 126× the seed set.
2. **100× the steps**, on a GPU or a long Actions run. The +0.32 overfit gap
   shows this is now the binding constraint.
3. ~~Evaluate on held-out prompts~~ — **done**: `src/eval_chat.py`, 400 held-out
   pairs, perplexity and screened generations.
4. **Then re-run the safety screen** on a model that can actually produce
   sentences — that is when decency screening starts to matter.

Until then, any claim that this copy "talks in Hindi" would be false.


## Clock-bound training

A fixed "10,000 steps" means different things on different machines. The run is
therefore bound to the CPU clock, and two numbers are measured rather than
assumed:

- **clock speed** — read from `/proc/cpuinfo`, or calibrated from a tight loop
  when the container hides it;
- **steps per second** — timed on the real model, batch and data path, over the
  first 15 steps.

The step target is then `min(target_steps, steps_per_second × budget_seconds)`.
A slow machine stops at the budget with fewer steps; a fast one reaches the
target and stops early. `runs/clock_report.json` records what actually happened.

Measured on this 2-core container:

| | |
|---|---|
| Clock | 2.532 GHz (`cpuinfo`) |
| Throughput | 1.576 steps/s |
| 3-minute budget | planned 283 steps, completed 283 in 204 s |
| **10,000 steps** | **105.8 min ≈ 1.76 h, 16.1 Tcycles** |

A GitHub Actions runner (2 vCPU, 6 h limit) fits roughly 34,000 steps, so the
10,000-step target completes in a single job with room to spare.

```bash
python src/train.py --stage chat --target-steps 10000 --clock-budget-minutes 300
gh workflow run train-hindi-chat -f target_steps=10000 -f clock_budget_minutes=300
```
