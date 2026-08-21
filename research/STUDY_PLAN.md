# Study Plan: Nano-Scale Language Models From Scratch

## Working Thesis

This project should be framed as an empirical study of nano-scale transformer language models, not as a frontier-model contribution. The strongest contribution is a controlled analysis of which design choices still matter when the model is under roughly 1M-17M parameters, trained on a small local corpus, and evaluated under laptop-scale compute constraints.

## Recommended Research Question

Primary question:

> Under severe scale constraints, which design choices most improve the tradeoff between validation loss, generation quality, and training stability in decoder-only transformer language models trained from scratch?

More focused version for a short paper:

> How do tokenization choice, vocabulary restriction, context length, and positional encoding affect nano-scale transformer language models trained from scratch on WikiText-like data?

This is the best fit for the current repo because the experiments already compare:

- Character tokenization vs BPE.
- 500-token vs 1000-token BPE vocabularies.
- General BPE vs English-restricted BPE.
- Learned positional embeddings, RoPE, and ALiBi-style attention bias.
- Shorter vs longer context windows.
- Small vs scaled model configurations.

## Candidate Paper Titles

- Tokenization and Positional Encoding Effects in Nano-Scale Transformer Language Models
- What Matters at Nano Scale? A Small-Compute Study of Transformer Language Models
- From Scratch Under Constraint: Empirical Lessons from Sub-20M Parameter Language Models
- Failure Modes of Tiny Decoder-Only Language Models Trained on WikiText

## Niche Contribution

The niche should be:

> A reproducible, small-compute study of training tiny language models from scratch, showing that conclusions from larger LLMs do not always transfer cleanly to nano-scale settings.

The paper should not claim:

- State-of-the-art language modeling.
- General LLM capability.
- Broad conclusions about production LLMs.

The paper can defensibly claim:

- Certain tokenization choices reduce validation loss but can worsen qualitative generations.
- Larger context windows helped sentence-like structure more than raw loss alone suggested.
- Small BPE vocabularies caused strong generation degradation even when loss looked competitive.
- ALiBi-style models were promising at small scale but still suffered from tokenization artifacts.
- Validation loss alone was insufficient for model selection in these nano-scale runs.

## Existing Evidence

Current dataset sizes:

- WikiText input: 12,099,569 bytes.
- Shakespeare input: 1,115,394 bytes.

Current saved checkpoint metadata:

| Artifact | Iteration | Approx. State Params |
| --- | ---: | ---: |
| `wikitext/best_model.pth` | 4600 | 7,076,096 |
| `wikitext/checkpoint.pth` | 5000 | 7,076,096 |
| `wikitext/v1_models/best_model.pth` | 1700 | 8,459,870 |
| `wikitext/v1_models/best_model_v1_1.pth` | 9500 | 8,519,262 |
| `wikitext/v2/v2_1/best_model.pth` | 4800 | 14,073,332 |
| `wikitext/v2/v2_2/best_model.pth` | 4800 | 7,357,172 |
| `wikitext/v2/v2_3/best_model.pth` | 5000 | 16,736,756 |
| `wikitext/v3/v3_0/best_modelv3_0.pth` | 5000 | 6,991,451 |
| `wikitext/v3/v3_1/best_modelv3_1.pth` | 2100 | 8,171,099 |

Current written results:

- `wikitext/results/v1.md`: character-token baselines and scaling.
- `wikitext/results/v2.md`: BPE experiments.
- `wikitext/results/v3.md`: English-restricted BPE and ALiBi/RoPE comparison.

## Metrics To Add

The current metrics are mostly train loss, validation loss, and qualitative notes. That is useful, but too thin for a paper. Add the following metrics because they are cheap enough for CPU-only evaluation.

### Core Metrics

1. Validation loss
   - Use a fixed validation split.
   - Evaluate with a fixed number of batches, such as 50 or 100, instead of 200 if CPU time is high.

2. Perplexity
   - Compute as `exp(validation_loss)`.
   - Report only when the tokenization is the same, or clearly warn that cross-tokenizer perplexity is not directly comparable.

3. Parameter count
   - Report trainable parameters and checkpoint state parameters.

4. Training stability
   - Best validation loss.
   - Iteration of best validation loss.
   - Train-val gap at best checkpoint.
   - Plateau iteration, if visible from logs.

5. Inference speed
   - Tokens generated per second for a fixed prompt and fixed `max_new_tokens`.
   - Use CPU-only as the reference hardware unless MPS becomes available.

### Tokenization Metrics

6. Token fertility
   - Average tokens per whitespace-delimited word on the same text sample.
   - This helps explain why some vocabularies produce weird spacing or poor words.

7. Unknown-token rate
   - Percentage of generated tokens that decode to `<unk>` or equivalent unknown markers.
   - Especially important for English-restricted BPE.

8. Punctuation spacing error rate
   - Count patterns like `" ,"`, `" ."`, `" @.@"`, or spaces around punctuation.
   - This directly measures a recurring qualitative failure mode in your notes.
   - Important caveat: the WikiText source text already contains some tokenization artifacts, so report this metric as a generated-text rate compared against the validation/source-text baseline rather than as an absolute model error count.

### Generation Metrics

9. Fixed-prompt samples
   - Use the same prompts for every model:
     - `The history of language`
     - `In the early twentieth century`
     - `The city was known for`
     - `According to the study`
     - empty prompt

10. Repetition and diversity
   - Distinct-1 and Distinct-2 over generated text.
   - Repeated bigram/trigram rate.

11. Qualitative rubric
   - Score each generated sample manually from 1-5 for:
     - word validity
     - sentence structure
     - topical consistency
     - punctuation quality
   - This is acceptable if the rubric is disclosed and applied consistently.

## Metrics To Avoid For Now

Avoid these unless the project grows:

- MMLU, HellaSwag, ARC, GSM8K, or other benchmark suites. These models are too small and not instruction-tuned.
- Human preference studies. Too much overhead for the current stage.
- Large-scale multi-seed sweeps. CPU-only training makes this expensive.
- Claims based on comparing perplexity across character tokenization and BPE without caveats.

## Machine-Constrained Evaluation Plan

Current observed environment:

- PyTorch is available through `/opt/anaconda3/bin/python`.
- CUDA is not available.
- MPS is not available from the current PyTorch build.
- Treat all evaluations as CPU-only unless the environment changes.

Recommended protocol:

1. Do not retrain every historical model.
2. Load saved checkpoints where possible.
3. Evaluate each checkpoint on a small fixed validation subset.
4. Generate fixed-prompt samples.
5. Add one optional controlled rerun only if needed.

CPU-friendly settings:

- `eval_batches = 50` for paper-draft iteration.
- `eval_batches = 100` for final reported numbers.
- `max_new_tokens = 200` for generated samples.
- `num_prompts = 5`.
- `temperature = 1.0` for default sampling.
- Also include one deterministic-ish setting if implemented later, such as `temperature = 0.8`.

## Minimal Additional Experiments

To make the paper stronger without turning it into a massive project:

1. Re-evaluate existing best checkpoints with the same fixed prompts and metric script.
2. Add tokenization diagnostics for the three vocabularies:
   - `bpe_vocab.json`
   - `bpe_vocab_500.json`
   - `bpe_vocab_english_500.json`
3. Run one controlled comparison between v3.0 scaled and v3.1 scaled if both can be loaded cleanly:
   - Same prompt set.
   - Same generation length.
   - Same metric extraction.
4. Optionally rerun v3.1 with dropout `0.2` vs `0.1` only if training time is acceptable.

## Proposed Experiment Table Schema

Every row in the final paper should have:

| Model | Tokenizer | Vocab | Positional Method | Params | Context | Layers | Heads | Embedding | Dropout | Best Val Loss | Perplexity | Train-Val Gap | Notes |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## Paper Argument

A defensible argument could be:

1. Scaling improves tiny models, but architecture and tokenizer choices dominate the usefulness of samples.
2. Character-level models can produce recognizable words and punctuation but struggle with longer semantic structure.
3. BPE improves word-level fluency but introduces tokenization-specific artifacts, especially punctuation spacing and unknown-token behavior.
4. The best validation loss does not always correspond to the best qualitative output under nano-scale constraints.
5. For small-compute research, reporting only validation loss hides important failure modes; lightweight generation diagnostics are necessary.

## Draft Abstract

We present a small-compute empirical study of decoder-only transformer language models trained from scratch at nano scale. Across a sequence of sub-20M parameter models, we compare character tokenization, byte-pair encoding, vocabulary restrictions, context length, model scaling, and positional encoding choices. Although larger configurations generally reduce validation loss, we find that validation loss alone is a weak proxy for generation quality in this regime. In particular, small BPE vocabularies and English-restricted tokenization can produce competitive losses while introducing visible punctuation, unknown-token, and word-formation artifacts. Our results suggest that nano-scale language model studies should report lightweight generation diagnostics alongside intrinsic loss metrics, especially when models are used for educational or low-compute experimentation.

## Implemented Tooling

Current research tooling:

- `research/metrics.py`: tokenizer, diversity, repetition, and punctuation-spacing diagnostics.
- `research/evaluate_fixed_prompts.py`: fixed-prompt generation evaluation for the v3.1 checkpoint.
- `research/results/metrics.json`: tokenizer diagnostics on the first 50,000 characters of `input.txt`.
- `research/results/fixed_prompt_eval_v3_1.json`: v3.1 fixed-prompt generation metrics.

Run from the `wikitext/` repository root:

```bash
/opt/anaconda3/bin/python research/metrics.py --max-chars 50000
/opt/anaconda3/bin/python research/evaluate_fixed_prompts.py --python /opt/anaconda3/bin/python
```

## Next Step

Build a small evaluation script that:

- Loads one model checkpoint at a time.
- Generates from the fixed prompt list.
- Computes generation diagnostics.
- Writes JSON/CSV results into `research/results/`.

Start with v3.1 because it currently loads cleanly after path fixes. Then backfill v3.0 and v2 checkpoints as import/path issues are cleaned up.

## Initial Tokenizer Diagnostics

The first CPU-cheap diagnostic run used the first 50,000 characters of `wikitext/input.txt`.

| Tokenizer | Vocab Entries | Tokens | Tokens/Word | Unknown Rate |
| --- | ---: | ---: | ---: | ---: |
| `bpe_vocab.json` | 1268 | 18,289 | 2.007 | 0.0000 |
| `bpe_vocab_500.json` | 768 | 21,251 | 2.332 | 0.0000 |
| `bpe_vocab_english_500.json` | 603 | 21,244 | 2.331 | 0.0052 |

Interpretation:

- The 500-token vocabularies require about 16% more tokens per word than the 1000-token vocabulary on the sample.
- The English-restricted vocabulary introduces measurable unknown-token behavior even on the source corpus.
- This supports the paper's qualitative claim that smaller/restricted vocabularies create representational pressure that is not visible from validation loss alone.

## Initial v3.1 Fixed-Prompt Results

The first fixed-prompt pass evaluated `v3/v3_1/best_modelv3_1.pth` on CPU with five prompts and 300 generated tokens per run.

| Prompt | Seconds | Continuation Words | Words/Sec | Space Before `.` | Space Before `,` | `@.@` Artifacts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `The history of language` | 5.41 | 131 | 24.20 | 7 | 5 | 1 |
| `In the early twentieth century` | 4.75 | 148 | 31.13 | 3 | 6 | 0 |
| `The city was known for` | 4.63 | 115 | 24.86 | 4 | 6 | 0 |
| `According to the study` | 4.68 | 136 | 29.06 | 6 | 8 | 0 |
| empty prompt | 4.60 | 114 | 24.77 | 4 | 6 | 0 |

Interpretation:

- The model reliably produces paragraph-like and WikiText-like surface structure.
- It frequently uses dates, named-entity-like strings, headings, and clauses.
- Word formation remains weak, with many plausible-looking but invalid words.
- Punctuation spacing artifacts are measurable and should be compared against the source-text baseline.
- CPU generation speed is usable for fixed-prompt evaluation, so this metric can be reported without GPU access.
