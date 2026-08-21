# Research

This folder turns the exploratory WikiText experiments into a reproducible study of nano-scale language models trained from scratch.

The current paper direction is:

> How do tokenization choice, vocabulary restriction, context length, and positional encoding affect nano-scale transformer language models trained from scratch on WikiText-like data?

## Files

- `STUDY_PLAN.md`: research question, claims, metric protocol, and current findings.
- `metrics.py`: cheap tokenizer and text diagnostics.
- `model_registry.py`: loadable model definitions for the important historical checkpoints.
- `pre_sweep.py`: checkpoint loadability and short-generation smoke tests.
- `evaluate_fixed_prompts.py`: fixed-prompt generation evaluation for saved checkpoints.
- `build_comparison_csv.py`: flattens fixed-prompt JSON into a per-model/per-prompt CSV.
- `build_summary_csv.py`: aggregates the comparison CSV into one row per model.
- `results/`: generated JSON outputs from the evaluation scripts.

## Run

Use the Anaconda Python because the local `.venv` does not currently have PyTorch installed.

```bash
/opt/anaconda3/bin/python research/metrics.py --max-chars 50000
/opt/anaconda3/bin/python research/pre_sweep.py --max-new-tokens 40
/opt/anaconda3/bin/python research/evaluate_fixed_prompts.py --all --max-new-tokens 40 --output research/results/fixed_prompt_eval_presweep.json
```

For the final comparison run, increase `--max-new-tokens` to `200` or `300`.

```bash
/opt/anaconda3/bin/python research/evaluate_fixed_prompts.py --all --max-new-tokens 200 --output research/results/fixed_prompt_eval_full.json
/opt/anaconda3/bin/python research/build_comparison_csv.py --input research/results/fixed_prompt_eval_full.json --output research/results/comparison.csv
/opt/anaconda3/bin/python research/build_summary_csv.py --input research/results/comparison.csv --output research/results/comparison_summary.csv
```

## Publication Boundary

This folder is suitable for the public GitHub repository. It contains experiment code, aggregate metrics, and generated samples. Keep private material elsewhere if it includes reviewer responses, private venue correspondence, unfinished anonymous submissions, or notes you do not want publicly associated with the project.
