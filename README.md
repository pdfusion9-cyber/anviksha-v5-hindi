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

## Held-out evaluation

This is the number that matters, and it is not the loss curve.

| Metric | Value |
|---|---|
| Train perplexity (200 pairs) | 7.71 |
| **Held-out perplexity (200 pairs)** | **8.04** |
| Overfit gap | **+0.32** |
| Generations screened | 30 |
| Flagged by the safety screen | 3 (all `empty`) |

**A gap of 0.32 means the model is underfitting, not overfitting.** It has not
memorised the 7,600 pairs; it has not yet learned them. That is a useful
diagnosis: the bottleneck is optimisation steps and capacity, not regularisation
and not data volume. Adding more data would not help until the model can fit
what it already has.

### The generations, unedited

Sampled at temperature 0.7, top-k 20, from held-out prompts:

```
U: Tiramisu is my favorite dessert. Where does it originate from?
A: Saaliy the kay ae sha ke ke t karaate ke he ki

U: What planet has the most moons? Hindi mein jawab dena.
A: SSehais painste me kaan kar e t har kaaiea k k

U: Bertie, I'm having such a bad day. Translation of the above into Hindi
A: न कें ka सe म क0
```

Still not Hindi conversation. Note what the held-out set actually contains: many
of these prompts are translation and transliteration tasks written in Latin
script, which is why the model's output drifts into Latin characters. The seed
set was clean Hindi chat; this corpus is broader and harder, and the model is
far too small for it.

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
