# Model 1 — गुणवत्ता और bilingual readiness roadmap

**Baseline artifact:** `hindi-five-folded`  
**Merge run:** [42233090777](https://github.com/pdfusion9-cyber/anviksha-v5-hindi/actions/runs/42233090777)  
**Status:** architecture merge and one-token-batch inference smoke test passed. This is not yet a bilingual-quality certification.

## 1. Verified architecture baseline

- 574,391,040 unique parameters; 8 transformer layers; 60 experts per layer; top-2 routing.
- Active-per-token estimate: 27,034,368 parameters.
- Strict state-dict loading passed.
- Finite forward-pass smoke test passed.
- Artifact SHA-256: `d3f855c3f0921846d2295166b7ef8eec6e6375d03dd8ad0c7d98f0a29c9a799d`.

These checks establish that the merged checkpoint can be loaded and run. They do not establish instruction following, factual accuracy, safety, or Hindi/English fluency.

## 2. Baseline data audit — blocking findings

The five source corpora were inspected at document level. Summed across the five exported domain files they contain 14,169,858 characters, but only 15 Devanagari characters were found in the inspected files (about 0.000106% of all characters). The current base is therefore effectively English-only.

| Source domain | Documents in local export | Devanagari characters | Docs passing a stricter 2-term domain-keyword floor | Spam/adult review flags |
|---|---:|---:|---:|---:|
| Crypto | 3,601 | 15 | 485 | 504 |
| Network | 3,497 | 0 | 135 | 760 |
| Application | 580 | 0 | 20 | 100 |
| Incident | 187 | 0 | 6 | 14 |
| Governance | 1,570 | 0 | 288 | 166 |

The domain-keyword floor is a screening heuristic, not a semantic relevance classifier. A passing document still requires quality sampling. A flag is a human-review request, not an automatic deletion instruction.

Five-document random samples also showed clear topic leakage: travel/betting/promotional text under crypto, public-health/social-policy articles under network, university and gaming material under application, and unrelated architecture/education material under incident. The incident sample included adult/spam promotion. Do not use these exports as the next clean training mixture without quarantine and re-review.

Detailed machine-readable audits are in `reports/quality_<domain>.json`.

## 3. Data policy before more training

Every source record must have: source URL or dataset ID, dataset revision, license, language, domain, source-document ID, split, content-filter result and SHA-256. Preserve one source document per JSONL record.

Required gates:
1. No unknown or incompatible license in the accepted manifest.
2. Train/validation/test split by source document (and entity where possible), not by sentence/window.
3. Exact duplicate overlap across splits = 0.
4. Hindi, English and Hinglish counts are measured by tokens and documents; the accepted mixture is intentionally balanced rather than assumed from a corpus name.
5. Domain relevance is checked by keyword screen plus a human-reviewed stratified sample. High-risk spam/adult/promotional contamination is quarantined.
6. The validation/test sets are frozen before training and never fed back into training.

## 4. Candidate public data sources

- **AI4Bharat IndicCorp v2:** 24-language monolingual pretraining corpus; its dataset card states CC0 for datasets created in the work. Acquire only the Hindi and English subsets needed; do not pull the whole multi-hundred-gigabyte collection.
  - https://huggingface.co/datasets/ai4bharat/IndicCorpV2
- **AI4Bharat IndicTrans2 / BPCC:** English–Indic parallel data useful for language alignment and translation-style tasks. The repository distinguishes CC0 and CC-BY-4.0 subsets; retain each subset's own license metadata and attribution.
  - https://github.com/AI4Bharat/IndicTrans2
- **AI4Bharat IN22-Conv:** 1,503 conversational translation examples across 22 Indic languages, marked CC-BY-4.0; reserve for evaluation rather than training.
  - https://huggingface.co/datasets/ai4bharat/IN22-Conv
- **AI4Bharat IndicInstruct/Airavata data collection:** candidate for instruction-format examples, but its collection contains multiple source datasets and translations. Verify each component's card/license and provenance before importing.
  - https://github.com/AI4Bharat/IndicInstruct
  - https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1

## 5. Training sequence

### Stage A — clean corpus and tokenizer
1. Build a source/license manifest and stream only selected Hindi and English datasets.
2. Add verified Hindi and English cybersecurity material; keep the five security domains balanced.
3. Build a byte-BPE tokenizer from a balanced Hindi/English/Hinglish sample at the same 8,192-token vocabulary size.
4. Remap the existing input/output embedding rows by token byte-string identity; initialize genuinely new merges from their constituent byte/token embeddings, then validate token round-trips and coverage before training. Do not simply replace the tokenizer while keeping old embedding row IDs unchanged.

### Stage B — continued pretraining
1. Continue from the merged 60-expert checkpoint with a fresh AdamW optimizer.
2. Use the corrected resumable stream trainer: persist optimizer, corpus fingerprint, byte cursor, prefetched windows, NumPy RNG and Torch RNG; fail closed if any required state is absent or the corpus/tokenizer changes.
3. Log per-language loss/perplexity, per-domain loss, token counts, expert usage and router balance every checkpoint.
4. Compare against the frozen baseline; keep the baseline artifact untouched.

### Stage C — supervised instruction tuning
Create a chat-formatted bilingual mixture with native Hindi, English and code-switched Hinglish examples. Separate user/assistant turns, mask prompt tokens from the loss, and retain document-level provenance and license metadata. Mix general tasks with all five cybersecurity domains; do not train only on long raw documents and then claim it is a conversational assistant.

### Stage D — evaluation and release gates
- Hindi, English and Hinglish: held-out perplexity plus generation quality reviewed by native speakers.
- Security: domain accuracy, grounded answer checks, abstention when evidence is missing, and technical correctness review.
- General instruction following: held-out instruction benchmark not seen during training.
- Safety: harmful-request refusal, prompt injection, credential handling, privacy, and unsafe cyber assistance tests.
- Routing: per-layer expert utilization, dead experts, load concentration and domain-to-expert diagnostics.
- Regression: compare every candidate against the frozen baseline; no release on a parameter-count claim alone.

## 6. Release rule

Do not label Model 1 “bilingual”, “reliable”, or “production-ready” until separate Hindi, English and Hinglish evaluations pass agreed thresholds, the training/validation split is audited, and the safety regression suite passes. Model size and structural loadability are necessary engineering checks, not quality scores.
