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
| Pretrain | 2 M characters of Hindi Wikipedia | 200 | perplexity 108.9 → **49.4** |
| Chat fine-tune | 60 hand-written Hindi pairs | 150 | loss 3.52 → **2.47** |

Model: 664,832 parameters, 3 layers, dim 128. Two CPU cores.

**60 chat pairs is a seed, not a dataset.** It is included so the format and the
evaluation are reproducible, not because it is enough to teach conversation.

## The generations, unedited

Sampled at temperature 0.8, top-k 20, from the chat checkpoint:

```
U: नमस्ते
A: कत पराया है।मिनहैंकत ल हिं हैनहै?मे

U: आपका नाम क्या है?
A: पतात बातार मे कि ब बता ब।

U: आप कैसे हैं?
A: नहिखा।मे हैंद हैं कर है। र क बाइरेर हि

U: भारत की राजधानी क्या है?
A: मने है।

U: मुझे उदासी है।
A: मे स।मस्यात को र ही हूँ। का कोता ना बा      [FLAG: low_content]

U: धन्यवाद
A: क सा ही है।हूँ।नक्या।                        [FLAG: low_content]
```

This is word salad. It is recognisably Devanagari, it has learned the sentence
terminator `।`, the copula `है`, and the first-person `हूँ` — and nothing else
holds together.

### Why, stated plainly

- **350 total optimisation steps.** A model needs orders of magnitude more.
  V5 itself ran three generations over 12,841 records on a phone; that is a
  larger budget than this run, and V5 also could not converse.
- **60 chat pairs.** The fine-tune can only memorise the surface form of a
  conversation, not learn to answer.
- **665 k parameters.** Roughly 1/1000th of the 100 MB target discussed earlier
  (200 M at 4-bit).

The corrections removed the *structural* blockers — byte tokenisation, missing
turn markers, code-only data. What remains is a compute and data gap, and no
architecture change closes that.

## Safety screening — what it does and does not do

`src/safety.py` flags four things: empty output, template-token leakage,
degenerate repetition, and low alphanumeric content. It also carries a short
list of unambiguous abusive Hindi words, deliberately small to avoid false
positives.

It caught the two degenerate replies above. It **cannot** catch a fluent but
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

1. **100× the chat data.** Thousands of Hindi instruction-response pairs, not 60.
2. **100× the steps**, on a GPU or a long Actions run. This is a compute gap.
3. **Evaluate on held-out prompts**, not the six above. A 6-prompt read is a
   sanity check, not an evaluation.
4. **Then re-run the safety screen** on a model that can actually produce
   sentences — that is when decency screening starts to matter.

Until then, any claim that this copy "talks in Hindi" would be false.
