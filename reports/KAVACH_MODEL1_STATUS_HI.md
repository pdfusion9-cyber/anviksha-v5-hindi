# ANVIKSHA KAVACH Model 1 — स्थिति और अगला विकास चरण

## पहचान

इस branch पर पाँच-domain cybersecurity MoE checkpoint को KAVACH Model 1 का unified base माना जा रहा है। पाँच source copies के domains: cryptography, network, application, incident response और governance. यह निर्णय उपयोगकर्ता के निर्देश के अनुसार है; Hindi/English अलग base-model project इस roadmap का target नहीं है।

## वर्तमान सत्यापित checkpoint

- 574,391,040 unique parameters; 60 experts/layer; 8 layers; top-2 routing.
- Merge run: https://github.com/pdfusion9-cyber/anviksha-v5-hindi/actions/runs/42233090777
- Merge output passed strict state-dict load and one finite forward-pass smoke test.
- The five source copies were short and unevenly trained. Their runs establish architecture/merge mechanics, not a mature cybersecurity assistant.
- The checkpoint was trained on the existing five security-domain corpus. Do not silently replace its tokenizer or embedding indices.

## KAVACH protection stack

The repository has a deterministic guard independent of the neural model:
1. capability policy
2. India PII handling
3. static code sandbox analysis
4. pattern and behavior detectors
5. learned/adversarial rules

The guard is not itself the language model. A guarded inference adapter now exists in `src/kavach_model1.py`; it performs input screening, strict checkpoint loading, token generation, and output screening. This adapter has only prompt-construction/policy tests so far; it has **not** yet been run against the 2.14 GB production artifact.

## Least-privilege profile

`cybersecurity_assistant` grants only:
- `answer_question`
- `read_document`

It does not grant:
- code execution
- secret access
- outbound network
- policy modification
- tool invocation

Hindi and Hinglish question recognition was added so safe explanatory questions are not treated as executable payloads merely because they mention code/security terminology.

## Regression tests

Branch-scoped workflow:
https://github.com/pdfusion9-cyber/anviksha-v5-hindi/actions/workflows/kavach-quality.yml

The workflow runs:
- Python compilation of guard + adapter modules
- least-privilege profile tests
- Hindi, English and Hinglish defensive-question policy tests
- blocked code execution and outbound network tests
- existing adversarial security regression cases

A first CI run failed because the workflow pointed to a non-existent test path. The path was corrected. The next run passed 7 tests. The expanded run including the inference adapter tests is being checked separately.

## Important limitations

- The current five-copy pretraining runs were short; the merged checkpoint has not had supervised instruction fine-tuning.
- A forward-pass smoke test does not establish factual accuracy, robust Hindi/English fluency, safe cyber reasoning, or useful chat responses.
- The adapter uses the existing chat special tokens, but the five source copies were trained primarily with streamed raw text. Chat-format response quality must be measured and will likely require instruction fine-tuning.
- Do not execute generated commands, scan external targets, or grant network/tool capabilities to the model. Keep analysis in a sandbox and use synthetic/authorized test data.

## Next gates (in order)

1. Finish deterministic KAVACH CI and fix every regression.
2. Run the guarded adapter against the exact merged artifact on a runner; capture Hindi/English prompts, raw generations, guard decisions and runtime. Do not call this a quality pass.
3. Build a held-out, licensed cybersecurity evaluation set with benign and adversarial examples across all five domains and both languages. Freeze it before tuning.
4. Evaluate domain accuracy, answer grounding, abstention, PII, prompt injection, safe code explanation vs execution, and false-block rate.
5. Only then prepare instruction fine-tuning from reviewed, licensed data. Preserve the original 574M checkpoint as immutable baseline.
6. Re-run the entire evaluation after fine-tuning and compare against baseline before release.

## Current release status

**Engineering integration in progress; not production-ready.** The architecture merge is complete. KAVACH policy/guard improvements and a guarded adapter are in the separate safe branch. Actual artifact-backed inference, instruction tuning and independent quality benchmarks remain outstanding.
